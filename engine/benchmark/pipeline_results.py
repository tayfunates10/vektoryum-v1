"""Convert production pipeline output into strict benchmark result records.

Unavailable metrics remain ``None``. The adapter never fabricates quality scores and
never changes the pipeline-selected artifact.

RFV-3D2: every result now carries ``metric_provenance`` proving at runtime which
path produced the metrics (exact final-artifact evaluation vs partial
quality-report fallback) and, when the exact path could not run, the exact
fail-closed reason class. Fallback is never silent and missing metrics stay
``None``; no value is fabricated and no threshold changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.final_artifact_evaluator import evaluate_final_svg
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba
from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS

PROVENANCE_SCHEMA = "rfv3d2-metric-provenance-v1"
# Windows drive or POSIX absolute path tokens — redacted from published evidence.
_ABS_PATH_TOKEN = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-~+]+){2,}")
_SANITIZED_MESSAGE_LIMIT = 200


def _sanitize_failure_message(message: object) -> str:
    """Redact filesystem paths/addresses and cap length; evidence stays publishable."""
    text = str(message)
    redacted = _ABS_PATH_TOKEN.sub("<redacted-path>", text)
    redacted = re.sub(r"0x[0-9a-fA-F]+", "<addr>", redacted)
    return redacted[:_SANITIZED_MESSAGE_LIMIT]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _get(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first(result: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _get(result, *path)
        if value is not None:
            return value
    return None


def extract_metrics(
    result: dict[str, Any], *, elapsed_ms: float, peak_rss_mb: float | None
) -> dict[str, float | int | None]:
    exact = _get(result, "final_artifact", "exact_metrics") or {}
    metrics: dict[str, float | int | None] = {
        "fidelity": _first(result, (("legacy_candidate_report", "metrics", "fidelity_score"), ("quality_report", "fidelity"), ("final_artifact", "exact_metrics", "fidelity"))),
        "ssim": _first(result, (("quality_report", "metrics", "B_appearance", "ssim"), ("final_artifact", "exact_metrics", "ssim"))),
        "edge_f1": _first(result, (("quality_report", "metrics", "B_appearance", "edge_f1"), ("final_artifact", "exact_metrics", "edge_f1"))),
        "alpha_iou": _first(result, (("quality_report", "metrics", "G_gradient_alpha", "alpha_iou"), ("final_artifact", "exact_metrics", "alpha_iou"))),
        "delta_e00": _first(result, (("quality_report", "metrics", "C_color", "de00_p95"), ("final_artifact", "exact_metrics", "delta_e00"))),
        "path_count": exact.get("path_count"),
        "svg_bytes": exact.get("svg_bytes"),
        "render_ms": round(float(elapsed_ms), 6),
        "peak_rss_mb": None if peak_rss_mb is None else round(float(peak_rss_mb), 6),
    }
    return {name: metrics.get(name) for name in REQUIRED_METRICS}


def _white_composite(rgba: Image.Image) -> np.ndarray:
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return np.asarray(background.convert("RGB"), dtype=np.uint8).copy()


def _new_provenance() -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "metric_source": "partial_quality_report",
        "exact_evaluator_attempted": False,
        "exact_evaluator_completed": False,
        "exact_evaluator_failure_class": None,
        "exact_evaluator_failure_message_sanitized": None,
        "selected_svg_path_present": False,
        "selected_svg_file_present": False,
        "selected_svg_sha256": None,
        "fallback_used": False,
        "fallback_source": None,
    }


def _fallback(
    output: dict[str, Any],
    provenance: dict[str, Any],
    *,
    failure_class: str,
    message: object | None,
    elapsed_ms: float,
    peak_rss_mb: float | None,
) -> tuple[dict[str, float | int | None], str | None, dict[str, Any]]:
    """Record a non-silent fallback; missing metrics stay ``None`` (no guessing)."""
    provenance["metric_source"] = "partial_quality_report"
    provenance["exact_evaluator_failure_class"] = failure_class
    if message is not None:
        provenance["exact_evaluator_failure_message_sanitized"] = _sanitize_failure_message(message)
    provenance["fallback_used"] = True
    provenance["fallback_source"] = "partial_quality_report"
    return (
        extract_metrics(output, elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb),
        None,
        provenance,
    )


def _exact_winner_metrics(
    output: dict[str, Any],
    source_rgba: Image.Image,
    *,
    elapsed_ms: float,
    peak_rss_mb: float | None,
) -> tuple[dict[str, float | int | None], str | None, dict[str, Any]]:
    provenance = _new_provenance()
    best = output.get("best") or {}
    raw_path = best.get("svg_path")
    if not raw_path:
        return _fallback(
            output, provenance, failure_class="selected_svg_path_missing", message=None,
            elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb,
        )
    provenance["selected_svg_path_present"] = True

    svg_path = Path(raw_path)
    if not svg_path.is_file():
        return _fallback(
            output, provenance, failure_class="selected_svg_file_missing", message=None,
            elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb,
        )
    provenance["selected_svg_file_present"] = True
    provenance["selected_svg_sha256"] = hashlib.sha256(svg_path.read_bytes()).hexdigest()

    provenance["exact_evaluator_attempted"] = True
    try:
        rgba = np.asarray(source_rgba, dtype=np.uint8).copy()
        source_rgb = _white_composite(source_rgba)
        source_alpha = rgba[:, :, 3]
        report = evaluate_final_svg(
            svg_path,
            source_rgb,
            source_alpha=source_alpha,
            image_class="clean_logo",
            required_metrics={"alpha_fidelity"},
        )
        exact = report.metrics
        visual = exact.get("B_visual") or {}
        color = exact.get("C_color") or {}
        edge = exact.get("D_edge_geometry") or {}
        editability = exact.get("H_editability") or {}
        gradient_alpha = exact.get("G_gradient_alpha") or {}

        rendered_rgba = render_svg_to_rgba(svg_path, source_rgba.width, source_rgba.height)
        alpha_iou: float | None = gradient_alpha.get("alpha_iou")
        if alpha_iou is None and rendered_rgba is not None:
            alpha_iou = alpha_plane_metrics(source_alpha, rendered_rgba[:, :, 3])["alpha_iou"]

        metrics: dict[str, float | int | None] = {
            "fidelity": best.get("fidelity_score"),
            "ssim": visual.get("ms_ssim") if visual.get("ms_ssim") is not None else visual.get("ssim"),
            "edge_f1": edge.get("edge_f1_1px"),
            "alpha_iou": alpha_iou,
            "delta_e00": color.get("de00_mean"),
            "path_count": editability.get("path_count"),
            "svg_bytes": svg_path.stat().st_size,
            "render_ms": round(float(elapsed_ms), 6),
            "peak_rss_mb": None if peak_rss_mb is None else round(float(peak_rss_mb), 6),
        }
    except Exception as exc:  # noqa: BLE001 — recorded, never silent
        return _fallback(
            output, provenance, failure_class="evaluator_exception", message=exc,
            elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb,
        )

    # The exact path only counts as completed when SSIM, edge F1 and delta_e00
    # are finite together. Anything else stays fail-closed with an explicit class.
    component = {name: metrics.get(name) for name in ("ssim", "edge_f1", "delta_e00")}
    if all(_finite(value) for value in component.values()):
        provenance["exact_evaluator_completed"] = True
        provenance["metric_source"] = "exact_final_artifact"
        return {name: metrics.get(name) for name in REQUIRED_METRICS}, report.sha256, provenance

    hard_codes = set(getattr(report, "hard_fail_codes", None) or [])
    failure_class = "render_failure" if "render_failed" in hard_codes else "exact_metrics_incomplete"
    missing = sorted(name for name, value in component.items() if not _finite(value))
    return _fallback(
        output, provenance, failure_class=failure_class,
        message=f"non-finite exact component metrics: {missing}",
        elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb,
    )


def run_case(
    case: BenchmarkCase,
    *,
    corpus_root: Path,
    work_root: Path,
    pipeline: Callable[..., dict[str, Any]],
    engine_version: str,
    trace_mode: str = "auto",
    peak_rss_mb: float | None = None,
) -> BenchmarkResult:
    source = (corpus_root / case.source_path).resolve()
    root = corpus_root.resolve()
    if root not in source.parents and source != root:
        raise ValueError(f"source_path escapes benchmark root: {case.source_path}")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != case.source_sha256:
        raise ValueError(f"source sha256 mismatch: {case.case_id}")

    job_dir = work_root / case.case_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        started = time.perf_counter()
        output = pipeline(image, source, trace_mode, job_dir)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics, exact_sha, provenance = _exact_winner_metrics(
            output,
            image,
            elapsed_ms=elapsed_ms,
            peak_rss_mb=peak_rss_mb,
        )

    artifact_sha = exact_sha or output.get("final_svg_sha256") or _get(output, "final_artifact", "final_svg_sha256")
    # Bind provenance to the artifact digest so evidence rows cannot be
    # reinterpreted against a different artifact later.
    provenance["artifact_sha256"] = artifact_sha
    return BenchmarkResult(
        case_id=case.case_id,
        engine_version=engine_version,
        metrics=metrics,
        artifact_sha256=artifact_sha,
        metric_provenance=provenance,
    )


def write_results(
    path: Path,
    results: list[BenchmarkResult],
    *,
    measurement_method: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": "benchmark-results-v1",
        "case_count": len(results),
        "measurement_method": measurement_method,
        "results": [item.to_dict() for item in sorted(results, key=lambda item: item.case_id)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")