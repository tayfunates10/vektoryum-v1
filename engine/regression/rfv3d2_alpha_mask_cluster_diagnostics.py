"""RFV-3D2 alpha/mask cluster diagnostics.

This module is diagnostic-only. It binds the five worst true-alpha cases from the
RFV-3B run after PR #105 and can reproduce the selected SVG once per case to
publish sanitized alpha-plane evidence. It never changes production selection,
thresholds, corpus, retry policy, or release state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, rgba_sha256

SCHEMA = "vektoryum-rfv3d2-alpha-mask-cluster-diagnostics-v1"
LIVE_SCHEMA = "vektoryum-rfv3d2-live-alpha-pair-v1"
SOURCE_MAIN_SHA = "c797f11f92a8d9d5ca879a798ff7c738590dad30"
SOURCE_HEAD_SHA = "5082e01d9777734e9d9da70a6f8d8d73e7676c30"
SOURCE_RUN_ID = 29683096355
AGGREGATE_ARTIFACT_ID = 8442804012
AGGREGATE_ARTIFACT_DIGEST = "sha256:43f6664b557ffc4e2cdb82a04a09ca65318721e01a7c0408968ed6ffe2a3aa22"
CORPUS_ARTIFACT_ID = 8441210832
CORPUS_ARTIFACT_DIGEST = "sha256:2b768850b11fabf37c2dd761c1c477e0798dd5b709d6a0643bcf402224b67744"
CASES = [
    "qualification-public-11",
    "qualification-public-17",
    "qualification-public-12",
    "qualification-public-18",
    "qualification-public-14",
]
CATEGORIES = {
    "qualification-public-11": "multicolor",
    "qualification-public-17": "complex_illustration",
    "qualification-public-12": "multicolor",
    "qualification-public-18": "complex_illustration",
    "qualification-public-14": "gradient_artwork",
}
_PATH_OR_SECRET = re.compile(
    r"(?:/home/|/tmp/|[A-Za-z]:\\|authorization\s*:|bearer\s+|token=|traceback|runner\\?[_ -]?temp)",
    re.IGNORECASE,
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def alpha_profile(alpha: np.ndarray) -> dict[str, Any]:
    plane = np.asarray(alpha, dtype=np.uint8)
    if plane.ndim != 2 or plane.size == 0:
        raise ValueError("non-empty 2D alpha plane required")
    return {
        "has_alpha": bool((plane < 255).any()),
        "alpha_sha256": hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest(),
        "soft_coverage": float(plane.astype(np.float32).mean() / 255.0),
        "binary_coverage": float((plane >= 1).mean()),
        "fully_transparent_ratio": float((plane == 0).mean()),
        "partial_alpha_ratio": float(((plane > 0) & (plane < 255)).mean()),
        "fully_opaque_ratio": float((plane == 255).mean()),
    }


def diagnose_alpha_pair(source_alpha: np.ndarray, render_alpha: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source_alpha, dtype=np.uint8)
    render = np.asarray(render_alpha, dtype=np.uint8)
    metrics = alpha_plane_metrics(source, render)
    source_profile = alpha_profile(source)
    render_profile = alpha_profile(render)

    if not source_profile["has_alpha"]:
        diagnosis = "not_applicable_opaque_source"
    elif metrics["alpha_iou"] >= 0.98 and metrics["alpha_mae"] <= 0.02:
        diagnosis = "alpha_preserved"
    elif (
        render_profile["soft_coverage"] >= 0.995
        and abs(metrics["alpha_iou"] - source_profile["soft_coverage"]) <= 1e-4
    ):
        diagnosis = "opaque_canvas_collapse"
    elif render_profile["soft_coverage"] - source_profile["soft_coverage"] >= 0.10:
        diagnosis = "alpha_overcoverage"
    else:
        diagnosis = "alpha_shape_regression"

    return {
        "schema": LIVE_SCHEMA,
        "diagnosis": diagnosis,
        "source": source_profile,
        "render": render_profile,
        "metrics": metrics,
    }


def diagnose_case(source_path: Path, svg_path: Path, *, case_id: str, category: str) -> dict[str, Any]:
    with Image.open(source_path) as opened:
        source_rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    height, width = source_rgba.shape[:2]
    render_rgba = render_svg_to_rgba(svg_path, width, height)
    if render_rgba is None:
        raise RuntimeError("selected SVG RGBA render failed")
    if render_rgba.shape[:2] != (height, width):
        raise RuntimeError("selected SVG RGBA render dimensions drifted")
    pair = diagnose_alpha_pair(source_rgba[:, :, 3], render_rgba[:, :, 3])
    pair.update(
        {
            "case_id": case_id,
            "category": category,
            "source_rgba_sha256": rgba_sha256(source_rgba),
            "render_rgba_sha256": rgba_sha256(render_rgba),
            "selected_svg_sha256": hashlib.sha256(Path(svg_path).read_bytes()).hexdigest(),
        }
    )
    return pair


def validate_evidence(payload: dict[str, Any]) -> None:
    expected_source = {
        "repository": "tayfunates10/vektoryum-v1",
        "main_sha": SOURCE_MAIN_SHA,
        "measurement_head_sha": SOURCE_HEAD_SHA,
        "workflow_run_id": SOURCE_RUN_ID,
        "aggregate_artifact_id": AGGREGATE_ARTIFACT_ID,
        "aggregate_artifact_digest": AGGREGATE_ARTIFACT_DIGEST,
        "corpus_artifact_id": CORPUS_ARTIFACT_ID,
        "corpus_artifact_digest": CORPUS_ARTIFACT_DIGEST,
    }
    if payload.get("schema") != SCHEMA:
        raise ValueError("diagnostic schema drift")
    if payload.get("source") != expected_source:
        raise ValueError("diagnostic source binding drift")
    scope = payload.get("scope")
    if scope != {
        "case_ids": CASES,
        "categories": CATEGORIES,
        "selection": "five_lowest_true_alpha_iou_cases",
    }:
        raise ValueError("diagnostic scope drift")

    observations = payload.get("observations")
    if not isinstance(observations, list) or len(observations) != len(CASES):
        raise ValueError("diagnostic observations incomplete")
    if [row.get("case_id") for row in observations] != CASES:
        raise ValueError("diagnostic case ordering drift")

    seen: set[str] = set()
    max_gap = 0.0
    for row in observations:
        case_id = row.get("case_id")
        if case_id in seen or case_id not in CATEGORIES:
            raise ValueError("diagnostic case identity drift")
        seen.add(case_id)
        if row.get("category") != CATEGORIES[case_id]:
            raise ValueError("diagnostic category drift")
        if row.get("source_has_alpha") is not True:
            raise ValueError("diagnostic true-alpha scope drift")
        if row.get("exact_evaluator_completed") is not True or row.get("metric_source") != "exact_final_artifact":
            raise ValueError("diagnostic exact evaluator binding drift")
        codes = row.get("hard_fail_codes")
        if not isinstance(codes, list) or not {"alpha_iou_below_min", "alpha_mae_above_max"}.issubset(codes):
            raise ValueError("diagnostic alpha failure signature drift")
        alpha_iou = row.get("alpha_iou")
        coverage = row.get("source_alpha_soft_coverage")
        gap = row.get("absolute_signature_gap")
        if not all(_finite(value) for value in (alpha_iou, coverage, gap)):
            raise ValueError("diagnostic non-finite metric")
        calculated = abs(float(alpha_iou) - float(coverage))
        if abs(calculated - float(gap)) > 1e-12:
            raise ValueError("diagnostic signature gap mismatch")
        if calculated > 1e-5 or row.get("opaque_canvas_signature") is not True:
            raise ValueError("diagnostic opaque-canvas signature drift")
        max_gap = max(max_gap, calculated)

    diagnosis = payload.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("diagnosis missing")
    if diagnosis.get("status") != "strong_signature_pending_live_render_confirmation":
        raise ValueError("diagnosis status drift")
    if diagnosis.get("signature") != "alpha_iou_equals_source_alpha_soft_coverage":
        raise ValueError("diagnosis signature drift")
    if diagnosis.get("hypothesis") != "selected_svg_render_alpha_is_full_canvas_opaque":
        raise ValueError("diagnosis hypothesis drift")
    if diagnosis.get("signature_case_count") != len(CASES):
        raise ValueError("diagnosis case count drift")
    if not _finite(diagnosis.get("max_absolute_gap")) or abs(float(diagnosis["max_absolute_gap"]) - max_gap) > 1e-12:
        raise ValueError("diagnosis max gap drift")
    if diagnosis.get("production_fix_authorized") is not False:
        raise ValueError("diagnostics cannot authorize production")
    if payload.get("release_decision") != "no_go" or payload.get("rfv4_allowed") is not False:
        raise ValueError("release/RFV-4 decision drift")
    if _PATH_OR_SECRET.search(json.dumps(payload, sort_keys=True, ensure_ascii=True)):
        raise ValueError("path, secret, runner location or traceback leaked")


def run_live(corpus_root: Path, case_id: str, work_root: Path, out_path: Path) -> dict[str, Any]:
    from app.pipeline_entry import run_pipeline  # noqa: PLC0415
    from engine.regression.rfv3_measurement_runner import load_qualification_cases  # noqa: PLC0415

    if case_id not in CASES:
        raise ValueError("case is outside RFV-3D2 diagnostic scope")
    cases = {case.case_id: case for case in load_qualification_cases(corpus_root)}
    case = cases[case_id]
    source_path = (corpus_root / case.source_path).resolve()
    job_dir = (work_root / case_id).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = opened.convert("RGBA")
        result = run_pipeline(image, source_path, "auto", job_dir)
    best = result.get("best") or {}
    svg_path = Path(best.get("svg_path") or "")
    if not svg_path.is_file():
        raise RuntimeError("pipeline selected SVG is missing")
    payload = diagnose_case(
        source_path,
        svg_path,
        case_id=case_id,
        category=CATEGORIES[case_id],
    )
    payload.update(
        {
            "engine_version": SOURCE_MAIN_SHA,
            "source_sha256": case.source_sha256,
            "mode_used": result.get("mode_used"),
            "production_fix_authorized": False,
            "release_decision": "no_go",
            "rfv4_allowed": False,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFV-3D2 alpha/mask diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    live = sub.add_parser("live")
    live.add_argument("--corpus-root", type=Path, required=True)
    live.add_argument("--case-id", required=True)
    live.add_argument("--work-root", type=Path, required=True)
    live.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "verify":
            validate_evidence(json.loads(args.evidence.read_text(encoding="utf-8")))
            result = {"status": "verified", "schema": SCHEMA, "case_count": len(CASES)}
        else:
            payload = run_live(args.corpus_root, args.case_id, args.work_root, args.out)
            result = {
                "status": "completed",
                "case_id": args.case_id,
                "diagnosis": payload["diagnosis"],
            }
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)[:200]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
