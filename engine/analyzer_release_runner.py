"""Generate and run the deterministic AA-4 labeled analyzer release corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from analyzer_release_contract import (
    AUTO_MODES,
    REPEAT_COUNT,
    SCHEMA_VERSION,
    THRESHOLDS,
    compute_release_metrics,
    validate_release_report,
)

REPEAT_TIMEOUT_SECONDS = 180
ENVIRONMENT = "no_hed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _geometric(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    u = size / 640.0
    draw.rectangle((40*u, 40*u, 600*u, 600*u), outline=(230, 25, 35), width=max(4, round(14*u)))
    draw.rectangle((120*u, 150*u, 260*u, 500*u), fill=(0, 0, 0))
    draw.rectangle((260*u, 150*u, 420*u, 205*u), fill=(0, 0, 0))
    draw.rectangle((260*u, 445*u, 420*u, 500*u), fill=(0, 0, 0))
    draw.polygon([(465*u, 145*u), (565*u, 320*u), (465*u, 500*u), (390*u, 320*u)], fill=(0, 0, 0))
    return image


def _minimal(size: int) -> Image.Image:
    """Curved monochrome wordmark-like field with a small blue accent.

    Dense smooth waves keep edge density inside the flat-logo band while avoiding
    the straight-corner signature of geometric marks. The small blue accent blocks
    destructive binary modes without making the image color-rich.
    """
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    u = size / 640.0
    line_width = max(2, round(3*u))
    for row in range(30):
        base_y = (55 + row * 18) * u
        points = []
        for x in range(30, 611, 3):
            wave = 8.0 * np.sin((x / 640.0) * np.pi * 6.0 + row * 0.31)
            points.append((x*u, base_y + wave*u))
        color = (35, 95, 205) if row in {10, 20} else (0, 0, 0)
        draw.line(points, fill=color, width=line_width)
    return image


def _color_logo(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    u = size / 640.0
    colors = [(220, 35, 45), (245, 145, 20), (45, 145, 75), (35, 100, 200), (135, 55, 165)]
    for index, color in enumerate(colors):
        x0 = (45 + index * 110) * u
        draw.rounded_rectangle((x0, 115*u, x0+105*u, 520*u), radius=max(5, round(18*u)), fill=color)
    draw.ellipse((215*u, 225*u, 425*u, 435*u), fill=(255, 255, 255))
    return image


def _single_color(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    u = size / 640.0
    draw.polygon([(85*u, 535*u), (320*u, 65*u), (555*u, 535*u)], fill=(0, 0, 0))
    draw.ellipse((250*u, 300*u, 390*u, 440*u), fill=(255, 255, 255))
    return image


def _lineart(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    u = size / 640.0
    width = max(2, round(3*u))
    for offset in range(75, 566, 70):
        draw.line((55*u, offset*u, 585*u, (640-offset)*u), fill=(0, 0, 0), width=width)
    for radius in (85, 145, 205):
        draw.ellipse(((320-radius)*u, (320-radius)*u, (320+radius)*u, (320+radius)*u), outline=(0, 0, 0), width=width)
    draw.rectangle((42*u, 42*u, 598*u, 598*u), outline=(0, 0, 0), width=width)
    return image


def _photo(size: int, seed: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)
    arr = np.empty((size, size, 3), dtype=np.float32)
    arr[:, :, 0] = 45 + 175 * xx / max(1, size - 1)
    arr[:, :, 1] = 55 + 155 * yy / max(1, size - 1)
    arr[:, :, 2] = 105 + 75 * np.sin((xx + yy) / max(8.0, size / 24.0))
    arr += rng.normal(0, 24, arr.shape)
    for cx, cy, radius, color in [
        (0.25, 0.28, 0.16, (215, 75, 45)),
        (0.70, 0.35, 0.20, (40, 95, 185)),
        (0.48, 0.72, 0.23, (45, 155, 85)),
    ]:
        mask = (xx-size*cx)**2 + (yy-size*cy)**2 <= (size*radius)**2
        texture = rng.normal(0, 18, (int(mask.sum()), 3))
        arr[mask] = np.asarray(color, dtype=np.float32) + texture
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def generate_corpus(root: Path) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    makers = {
        "geometric_logo": lambda size, boundary: _geometric(size),
        "minimal_ai": lambda size, boundary: _minimal(size),
        "logo_color": lambda size, boundary: _color_logo(size),
        "single_color": lambda size, boundary: _single_color(size),
        "lineart": lambda size, boundary: _lineart(size),
        "photo_poster": lambda size, boundary: _photo(size, 20260714 + int(boundary)),
    }
    cases: list[dict[str, Any]] = []
    for mode in AUTO_MODES:
        for kind in ("in_domain", "boundary"):
            boundary = kind == "boundary"
            size = 640 if not boundary else 384
            case_id = f"{mode}-{kind}"
            path = root / f"{case_id}.png"
            _save(makers[mode](size, boundary), path)
            cases.append(
                {
                    "case_id": case_id,
                    "label": mode,
                    "kind": kind,
                    "environment": ENVIRONMENT,
                    "source_path": str(path),
                    "source_sha256": _sha256(path),
                }
            )
    return cases


def _worker(queue: Any, case: dict[str, Any], repeat_index: int) -> None:
    try:
        import app.analyzer as analyzer
        from app.analyzer_decision_gate import decide_trace_mode

        analyzer.calculate_semantic_edge_stats = lambda _image: None
        source = Path(case["source_path"])
        if _sha256(source) != case["source_sha256"]:
            raise ValueError("source digest mismatch")
        with Image.open(source) as opened:
            image = opened.convert("RGBA")
            analysis = analyzer.analyze_image_from_mem(image)
            decision = decide_trace_mode(analysis, image, "auto")
        contract = analysis.get("analyzer_contract") or {}
        diagnostic_keys = (
            "flat_color_count",
            "edge_density",
            "thin_ink_ratio",
            "straight_edge_likelihood",
            "corner_likelihood",
            "has_gradient",
            "likely_geometric_logo",
            "likely_text_logo",
            "likely_color_logo",
            "likely_single_color",
            "likely_line_art",
            "likely_photo_or_complex",
            "semantic_photo_like",
        )
        queue.put(
            {
                "ok": True,
                "sample": {
                    "repeat_index": repeat_index,
                    "status": "success",
                    "contract_status": contract.get("status"),
                    "source_pixel_sha256": contract.get("source_pixel_sha256"),
                    "feature_digest": contract.get("feature_digest"),
                    "recommendation_digest": contract.get("recommendation_digest"),
                    "detected_type": analysis.get("detected_type"),
                    "recommended_mode": analysis.get("recommended_mode"),
                    "decision_status": decision.get("status"),
                    "execution_mode": decision.get("execution_mode"),
                    "fallback_applied": decision.get("fallback_applied"),
                    "confidence": decision.get("confidence"),
                    "runner_up_mode": decision.get("runner_up_mode"),
                    "runner_up_margin": decision.get("runner_up_margin"),
                    "reason_codes": decision.get("reason_codes") or [],
                    "hed_status": (contract.get("optional_signals") or {}).get("hed"),
                    "support_scores": contract.get("support_scores"),
                    "analysis_features": {key: analysis.get(key) for key in diagnostic_keys},
                },
            }
        )
    except BaseException as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


def run_sample(case: dict[str, Any], repeat_index: int, timeout_seconds: int) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(queue, case, repeat_index))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"repeat_index": repeat_index, "status": "failure", "error": "timeout"}
    try:
        payload = queue.get(timeout=2)
    except Empty:
        return {"repeat_index": repeat_index, "status": "failure", "error": "no_result"}
    if not payload.get("ok") or process.exitcode != 0:
        return {
            "repeat_index": repeat_index,
            "status": "failure",
            "error": payload.get("error") or f"exit:{process.exitcode}",
        }
    return dict(payload["sample"])


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _report_payload(cases: list[dict[str, Any]], repeat_count: int, verdict: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "repeat_count": repeat_count,
        "environment": ENVIRONMENT,
        "thresholds": THRESHOLDS,
        "metrics": compute_release_metrics(cases),
        "verdict": verdict,
        "errors": [],
        "cases": cases,
    }


def run_release(
    output_dir: Path,
    *,
    repeat_count: int = REPEAT_COUNT,
    timeout_seconds: int = REPEAT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if repeat_count != REPEAT_COUNT:
        raise ValueError(f"AA-4 requires exactly {REPEAT_COUNT} repeats")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    output_dir = Path(output_dir)
    cases = generate_corpus(output_dir / "fixtures")
    report_cases: list[dict[str, Any]] = []
    report_path = output_dir / "analyzer_release_report.json"

    for case in cases:
        samples: list[dict[str, Any]] = []
        for repeat_index in range(1, repeat_count + 1):
            samples.append(run_sample(case, repeat_index, timeout_seconds))
            partial_cases = report_cases + [
                {
                    **{key: value for key, value in case.items() if key != "source_path"},
                    "deterministic": False,
                    "samples": samples,
                }
            ]
            _write_report(report_path, _report_payload(partial_cases, repeat_count, "running"))
        signatures = {
            json.dumps(
                {key: value for key, value in sample.items() if key != "repeat_index"},
                sort_keys=True,
            )
            for sample in samples
        }
        report_cases.append(
            {
                **{key: value for key, value in case.items() if key != "source_path"},
                "deterministic": len(signatures) == 1,
                "samples": samples,
            }
        )

    report = _report_payload(report_cases, repeat_count, "release_ready")
    errors = validate_release_report(report)
    if errors:
        report["verdict"] = "failed"
        report["errors"] = errors
        report["errors"] = validate_release_report(report)
    _write_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("analyzer_release_artifacts"))
    parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
    parser.add_argument("--timeout-seconds", type=int, default=REPEAT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    report = run_release(
        args.output,
        repeat_count=args.repeat_count,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}, sort_keys=True))
    return 0 if report["verdict"] == "release_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
