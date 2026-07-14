"""Generate and validate the deterministic CVE-4 explicit-mode release corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
from typing import Any

import numpy as np
from defusedxml import ElementTree as SafeET
from PIL import Image, ImageDraw
from svgpathtools import parse_path

from app.fidelity import render_svg_to_rgb
from app.final_artifact_evaluator import evaluate_final_svg
from app.pipeline_entry import run_pipeline
from app.quality import basic_svg_quality_check
from app.scoring import _parse_svg_stats
from core_release_contract import (
    PRODUCTION_MODES,
    REPEAT_COUNT,
    REQUIRED_WORKFLOWS,
    SCHEMA_VERSION,
    validate_release_report,
)
from regression.artifact_probe import halo_ratio, ink_metrics, seam_ratio

REPEAT_TIMEOUT_SECONDS = 1800
_IMAGE_CLASS = {
    "geometric_logo": "geometric",
    "minimal_ai": "clean_logo",
    "logo_color": "clean_logo",
    "flat_logo": "clean_logo",
    "single_color": "clean_logo",
    "lineart": "lineart",
    "centerline": "lineart",
    "photo_poster": "photo",
}
_STYLE_FILL = re.compile(r"(?:^|;)\s*fill\s*:\s*([^;]+)", re.I)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _white_composite(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3]
    af = alpha.astype(np.float32)[:, :, None] / 255.0
    rgb = np.clip(
        np.rint(rgba[:, :, :3].astype(np.float32) * af + 255.0 * (1.0 - af)),
        0,
        255,
    ).astype(np.uint8)
    return rgb, alpha.copy() if bool((alpha < 255).any()) else None


def _fixture(mode: str, path: Path, size: int = 192) -> dict[str, Any]:
    """Create one small deterministic source tailored to an explicit mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    white = (255, 255, 255, 255)
    image = Image.new("RGBA", (size, size), white)
    draw = ImageDraw.Draw(image)
    palette: list[tuple[int, int, int]] = [(255, 255, 255)]

    if mode == "geometric_logo":
        draw.rectangle((18, 18, 174, 174), outline=(255, 0, 0, 255), width=8)
        draw.rectangle((42, 54, 82, 138), fill=(0, 0, 0, 255))
        draw.polygon([(108, 48), (160, 96), (108, 144), (86, 96)], fill=(0, 0, 0, 255))
        palette += [(255, 0, 0), (0, 0, 0)]
    elif mode == "minimal_ai":
        draw.ellipse((30, 30, 162, 162), fill=(20, 32, 48, 255))
        draw.polygon([(96, 48), (142, 136), (50, 136)], fill=(245, 175, 35, 255))
        palette += [(20, 32, 48), (245, 175, 35)]
    elif mode == "logo_color":
        colors = [(220, 40, 50), (255, 150, 20), (55, 145, 80), (40, 105, 195)]
        for index, color in enumerate(colors):
            x0 = 16 + index * 40
            draw.rectangle((x0, 34, x0 + 40, 154), fill=(*color, 255))
        draw.ellipse((58, 58, 134, 134), fill=(255, 255, 255, 255))
        palette += colors
    elif mode == "flat_logo":
        draw.rounded_rectangle((22, 36, 170, 156), radius=22, fill=(34, 86, 165, 255))
        draw.polygon([(48, 132), (96, 54), (144, 132)], fill=(245, 145, 25, 255))
        palette += [(34, 86, 165), (245, 145, 25)]
    elif mode == "single_color":
        draw.polygon([(28, 156), (96, 24), (164, 156)], fill=(0, 0, 0, 255))
        draw.ellipse((72, 86, 120, 134), fill=white)
        palette += [(0, 0, 0)]
    elif mode == "lineart":
        draw.rectangle((24, 24, 168, 168), outline=(0, 0, 0, 255), width=4)
        draw.line((34, 148, 92, 54, 158, 142), fill=(0, 0, 0, 255), width=4, joint="curve")
        draw.ellipse((66, 66, 126, 126), outline=(0, 0, 0, 255), width=4)
        palette += [(0, 0, 0)]
    elif mode == "centerline":
        draw.line((32, 96, 160, 96), fill=(0, 0, 0, 255), width=1)
        palette += [(0, 0, 0)]
    elif mode == "photo_poster":
        yy, xx = np.mgrid[0:size, 0:size]
        rng = np.random.default_rng(20260714)
        arr = np.empty((size, size, 4), dtype=np.uint8)
        arr[:, :, 0] = np.clip(45 + 180 * xx / max(1, size - 1), 0, 255)
        arr[:, :, 1] = np.clip(65 + 150 * yy / max(1, size - 1), 0, 255)
        arr[:, :, 2] = np.clip(120 + 60 * np.sin((xx + yy) / 22.0), 0, 255)
        noise = rng.normal(0, 12, (size, size, 3))
        arr[:, :, :3] = np.clip(arr[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        arr[:, :, 3] = 255
        image = Image.fromarray(arr, "RGBA")
        palette = [(45, 65, 120), (225, 215, 180), (255, 255, 255)]
    else:
        raise ValueError(f"unsupported production mode: {mode}")

    image.save(path, format="PNG", optimize=False)
    return {
        "mode": mode,
        "source_sha256": _sha256(path),
        "palette": palette,
        "image_class": _IMAGE_CLASS[mode],
    }


def _path_fill(element: Any) -> str:
    direct = element.attrib.get("fill")
    if direct is not None:
        return str(direct).strip().lower()
    style = str(element.attrib.get("style") or "")
    match = _STYLE_FILL.search(style)
    return match.group(1).strip().lower() if match else "black"


def _has_open_required_cycle(svg_path: Path) -> bool:
    """Reject geometrically open filled subpaths while allowing open strokes.

    A literal ``Z`` command is not required: several production tracers serialize
    a closed curve by returning its final endpoint to the first point.  The
    geometric path contract, not command spelling, is authoritative.
    """
    try:
        root = SafeET.parse(str(svg_path)).getroot()
    except Exception:
        return True
    for element in root.iter():
        if str(element.tag).split("}")[-1].lower() != "path":
            continue
        if _path_fill(element) == "none":
            continue
        d = str(element.attrib.get("d") or "")
        try:
            subpaths = parse_path(d).continuous_subpaths()
        except Exception:
            return True
        if not subpaths or any(not subpath.isclosed() for subpath in subpaths):
            return True
    return False


def _score_snapshot_match(best: dict[str, Any], svg_path: Path) -> bool:
    scored = best.get("score_details")
    if not isinstance(scored, dict):
        return False
    actual = _parse_svg_stats(svg_path)
    return all(
        scored.get(name) == actual.get(name)
        for name in ("path_count", "node_count", "unique_colors", "has_bitmap")
    )


def _quality_verdict(output: dict[str, Any], final_verdict: str) -> tuple[str, list[str]]:
    best = output.get("best") or {}
    report = basic_svg_quality_check(
        score_details=best.get("score_details", {}),
        mode=str(output.get("mode_used") or ""),
        geometry_report=best.get("cleanup_report", {}).get("report", {}),
        total_score=float(best.get("total_score") or 0.0),
        fidelity_score=best.get("fidelity_score"),
        structure_report=output.get("structure_report"),
    )
    reasons = [f"quality_warning:{index}" for index, _warning in enumerate(report.get("warnings") or [], start=1)]
    if report.get("status") == "failed":
        return "failed", reasons
    if final_verdict in {"failed", "needs_review"} or report.get("status") == "needs_review":
        return "needs_review", reasons
    return "production_ready", reasons


def _no_candidate_sample(mode: str, repeat_index: int, output: dict[str, Any]) -> dict[str, Any]:
    errors = [str(item.get("error") or "").lower() for item in output.get("results", [])]
    optional_only = bool(errors) and all(
        any(token in error for token in ("not found", "unavailable", "missing dependency", "yok"))
        for error in errors
    )
    return {
        "repeat_index": repeat_index,
        "status": "unavailable" if optional_only else "failed",
        "verdict": "unavailable" if optional_only else "failed",
        "reason_codes": ["optional_backend_unavailable" if optional_only else "no_vector_candidate"],
        "artifact_sha256": None,
    }


def _run_sample_inner(mode: str, repeat_index: int, source_path: Path, job_dir: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = opened.convert("RGBA")
        source_rgb, source_alpha = _white_composite(image)
        output = run_pipeline(image, source_path, mode, job_dir)

    if output.get("mode_used") != mode:
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["explicit_mode_drift"],
            "artifact_sha256": None,
        }
    best = output.get("best")
    if not isinstance(best, dict):
        return _no_candidate_sample(mode, repeat_index, output)

    svg_path = Path(str(best.get("svg_path") or ""))
    if not svg_path.is_file() or svg_path.stat().st_size <= 0:
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["winner_svg_missing"],
            "artifact_sha256": None,
        }

    artifact_sha = _sha256(svg_path)
    final = evaluate_final_svg(
        svg_path,
        source_rgb,
        source_alpha=source_alpha,
        image_class=str(fixture["image_class"]),
        required_metrics={"alpha_fidelity"} if source_alpha is not None else set(),
    )
    rendered = render_svg_to_rgb(svg_path, source_rgb.shape[1], source_rgb.shape[0])
    if rendered is None:
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["release_render_failed"],
            "artifact_sha256": artifact_sha,
            "evaluator_sha256": final.sha256,
        }

    ink = ink_metrics(source_rgb, rendered)
    metrics = {
        "ink_recall": float(ink["ink_recall"]),
        "ink_precision": float(ink["ink_precision"]),
        "component_delta": int(ink["component_delta"]),
        "seam_ratio": float(seam_ratio(source_rgb, rendered)),
        "halo_ratio": float(halo_ratio(rendered, list(fixture["palette"]))),
    }
    structure = dict((final.metrics.get("A_structure") or {}))
    structure["open_required_cycle"] = _has_open_required_cycle(svg_path)
    verdict, quality_reasons = _quality_verdict(output, final.verdict)
    reason_codes = list(dict.fromkeys(
        list(final.hard_fail_codes)
        + list(final.soft_warning_codes)
        + [f"unmeasured:{name}" for name in final.unmeasured_required]
        + quality_reasons
    ))
    if mode == "photo_poster" and "accepted_photo_product_limit" not in reason_codes:
        reason_codes.append("accepted_photo_product_limit")

    status = "failed" if verdict == "failed" else "completed"
    return {
        "repeat_index": repeat_index,
        "status": status,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "artifact_sha256": artifact_sha,
        "evaluator_sha256": final.sha256,
        "output_digest_match": final.sha256 == artifact_sha,
        "score_snapshot_match": _score_snapshot_match(best, svg_path),
        "structure": {
            "structural_safe": bool(structure.get("structural_safe")),
            "has_bitmap": bool(structure.get("has_raster")),
            "nonfinite": bool(structure.get("nonfinite")),
            "open_required_cycle": bool(structure["open_required_cycle"]),
            "path_count": int(structure.get("path_count") or 0),
            "byte_read_stable": bool(structure.get("byte_read_stable")),
        },
        "metrics": metrics,
        "candidate": str(best.get("name") or ""),
        "total_score": best.get("total_score"),
        "fidelity_score": best.get("fidelity_score"),
    }


def _worker(queue: Any, mode: str, repeat_index: int, source_path: str, job_dir: str, fixture: dict[str, Any]) -> None:
    try:
        sample = _run_sample_inner(mode, repeat_index, Path(source_path), Path(job_dir), fixture)
        queue.put({"ok": True, "sample": sample})
    except BaseException as exc:  # fail closed across process boundary
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_sample_isolated(
    mode: str,
    repeat_index: int,
    source_path: Path,
    job_dir: Path,
    fixture: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker,
        args=(queue, mode, repeat_index, str(source_path), str(job_dir), fixture),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["release_repeat_timeout"],
            "artifact_sha256": None,
        }
    if queue.empty():
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["release_repeat_no_result"],
            "artifact_sha256": None,
        }
    payload = queue.get()
    if not payload.get("ok") or process.exitcode != 0:
        return {
            "repeat_index": repeat_index,
            "status": "failed",
            "verdict": "failed",
            "reason_codes": ["release_repeat_exception"],
            "error": payload.get("error"),
            "artifact_sha256": None,
        }
    return dict(payload["sample"])


def _mode_record(mode: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {sample.get("status") for sample in samples}
    verdicts = {sample.get("verdict") for sample in samples}
    reasons = list(dict.fromkeys(
        code
        for sample in samples
        for code in (sample.get("reason_codes") or [])
    ))
    if statuses == {"unavailable"}:
        status = "unavailable"
    elif statuses == {"completed"}:
        status = "production_ready" if verdicts == {"production_ready"} else "needs_review"
    else:
        status = "failed"
    if mode == "photo_poster" and "accepted_photo_product_limit" not in reasons:
        reasons.append("accepted_photo_product_limit")
    return {
        "mode": mode,
        "status": status,
        "reason_codes": reasons,
        "samples": samples,
    }


def run_release_corpus(
    output_dir: Path,
    *,
    engine_version: str,
    repeat_count: int = REPEAT_COUNT,
    timeout_seconds: int = REPEAT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if repeat_count != REPEAT_COUNT:
        raise ValueError(f"CVE-4 requires exactly {REPEAT_COUNT} repeats")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")

    output_dir = Path(output_dir)
    fixture_dir = output_dir / "fixtures"
    job_root = output_dir / "jobs"
    report_path = output_dir / "core_release_report.json"
    modes: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": str(engine_version),
        "repeat_count": repeat_count,
        "required_workflows": list(REQUIRED_WORKFLOWS),
        "modes": modes,
        "validation": None,
    }

    for mode in PRODUCTION_MODES:
        source_path = fixture_dir / f"{mode}.png"
        fixture = _fixture(mode, source_path)
        samples: list[dict[str, Any]] = []
        for repeat_index in range(1, repeat_count + 1):
            sample = _run_sample_isolated(
                mode,
                repeat_index,
                source_path,
                job_root / mode / f"repeat-{repeat_index}",
                fixture,
                timeout_seconds,
            )
            samples.append(sample)
            provisional = modes + [_mode_record(mode, samples)]
            _write_json(report_path, {**payload, "modes": provisional})
        modes.append(_mode_record(mode, samples))
        _write_json(report_path, payload)

    validation = validate_release_report(payload)
    payload["validation"] = validation
    _write_json(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("core_release_artifacts"))
    parser.add_argument("--engine-version", default="unknown")
    parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
    parser.add_argument("--repeat-timeout-seconds", type=int, default=REPEAT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    report = run_release_corpus(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        timeout_seconds=args.repeat_timeout_seconds,
    )
    validation = report["validation"]
    print(json.dumps({
        "status": validation["status"],
        "mode_count": len(report["modes"]),
        "repeat_count": report["repeat_count"],
        "reason_codes": validation["reason_codes"],
    }, sort_keys=True))
    return 0 if validation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
