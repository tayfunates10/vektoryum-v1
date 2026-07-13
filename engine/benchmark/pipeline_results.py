"""Convert production pipeline output into strict benchmark result records.

Unavailable metrics remain ``None``. The adapter never fabricates quality scores and
never changes the pipeline-selected artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS


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

    artifact_sha = output.get("final_svg_sha256") or _get(output, "final_artifact", "final_svg_sha256")
    return BenchmarkResult(
        case_id=case.case_id,
        engine_version=engine_version,
        metrics=extract_metrics(output, elapsed_ms=elapsed_ms, peak_rss_mb=peak_rss_mb),
        artifact_sha256=artifact_sha,
    )


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    payload = {
        "schema_version": "benchmark-results-v1",
        "case_count": len(results),
        "results": [item.to_dict() for item in sorted(results, key=lambda item: item.case_id)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
