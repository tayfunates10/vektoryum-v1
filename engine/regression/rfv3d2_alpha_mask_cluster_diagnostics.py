"""RFV-3D2 alpha/mask cluster diagnostics (diagnostic-only)."""
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
LIVE_PR_HEAD_SHA = "fc923cb2af7e5d9dfa3e63666dd3b0757185c6cc"
LIVE_WORKFLOW_RUN_ID = 29689639516
CASES = [
    "qualification-public-11", "qualification-public-17", "qualification-public-12",
    "qualification-public-18", "qualification-public-14",
]
CATEGORIES = {
    "qualification-public-11": "multicolor",
    "qualification-public-17": "complex_illustration",
    "qualification-public-12": "multicolor",
    "qualification-public-18": "complex_illustration",
    "qualification-public-14": "gradient_artwork",
}
LIVE_ARTIFACTS = {
    "qualification-public-11": (8444221003, "sha256:a3ca5eac7452a2aa3872617e998404e2e811db770fc427476cdfbd1968215682"),
    "qualification-public-17": (8443263360, "sha256:1101433d860da4d3f6e5af1ded7b86255997ef31996adb908235900d086dafdd"),
    "qualification-public-12": (8443463050, "sha256:de1ae09b8ffd00a82af24a97e71c8e3b8dbb5030e3b75c94db172b8b5b48dc54"),
    "qualification-public-18": (8443238140, "sha256:3809702dd88229eefea184ab56c705ee56fc2e7924df067ca517ed4b093d5a69"),
    "qualification-public-14": (8443326981, "sha256:2e21a528ef0c7a2b386119f1cf1883f92b30243635d9b1c3c776ca82c8fcd27a"),
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PATH_OR_SECRET = re.compile(r"(?:/home/|/tmp/|[A-Za-z]:\\|authorization\s*:|bearer\s+|token=|traceback|runner\\?[_ -]?temp)", re.I)


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
    source_profile, render_profile = alpha_profile(source), alpha_profile(render)
    if not source_profile["has_alpha"]:
        diagnosis = "not_applicable_opaque_source"
    elif metrics["alpha_iou"] >= 0.98 and metrics["alpha_mae"] <= 0.02:
        diagnosis = "alpha_preserved"
    elif render_profile["soft_coverage"] >= 0.995 and abs(metrics["alpha_iou"] - source_profile["soft_coverage"]) <= 1e-4:
        diagnosis = "opaque_canvas_collapse"
    elif render_profile["soft_coverage"] - source_profile["soft_coverage"] >= 0.10:
        diagnosis = "alpha_overcoverage"
    else:
        diagnosis = "alpha_shape_regression"
    return {"schema": LIVE_SCHEMA, "diagnosis": diagnosis, "source": source_profile, "render": render_profile, "metrics": metrics}


def diagnose_case(source_path: Path, svg_path: Path, *, case_id: str, category: str) -> dict[str, Any]:
    with Image.open(source_path) as opened:
        source_rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    height, width = source_rgba.shape[:2]
    render_rgba = render_svg_to_rgba(svg_path, width, height)
    if render_rgba is None or render_rgba.shape[:2] != (height, width):
        raise RuntimeError("selected SVG RGBA render failed or dimensions drifted")
    pair = diagnose_alpha_pair(source_rgba[:, :, 3], render_rgba[:, :, 3])
    pair.update({
        "case_id": case_id, "category": category,
        "source_rgba_sha256": rgba_sha256(source_rgba),
        "render_rgba_sha256": rgba_sha256(render_rgba),
        "selected_svg_sha256": hashlib.sha256(Path(svg_path).read_bytes()).hexdigest(),
    })
    return pair


def _validate_live_summary(row: dict[str, Any], case_id: str) -> None:
    artifact_id, artifact_digest = LIVE_ARTIFACTS[case_id]
    expected = {"artifact_id": artifact_id, "artifact_digest": artifact_digest, "case_id": case_id, "category": CATEGORIES[case_id], "diagnosis": "opaque_canvas_collapse"}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"live artifact binding drift: {case_id}:{key}")
    for key in ("source_sha256", "source_rgba_sha256", "selected_svg_sha256", "render_rgba_sha256", "source_alpha_sha256", "render_alpha_sha256"):
        if not isinstance(row.get(key), str) or not _HEX64.fullmatch(row[key]):
            raise ValueError(f"live hash drift: {case_id}:{key}")
    if not all(_finite(row.get(k)) for k in ("source_soft_coverage", "render_soft_coverage", "alpha_iou")):
        raise ValueError("live scalar metric drift")
    if float(row["render_soft_coverage"]) < 0.995 or abs(float(row["alpha_iou"]) - float(row["source_soft_coverage"])) > 1e-4:
        raise ValueError("live opaque-canvas metric drift")


def validate_evidence(payload: dict[str, Any]) -> None:
    expected_source = {
        "repository": "tayfunates10/vektoryum-v1", "main_sha": SOURCE_MAIN_SHA,
        "measurement_head_sha": SOURCE_HEAD_SHA, "workflow_run_id": SOURCE_RUN_ID,
        "aggregate_artifact_id": AGGREGATE_ARTIFACT_ID, "aggregate_artifact_digest": AGGREGATE_ARTIFACT_DIGEST,
        "corpus_artifact_id": CORPUS_ARTIFACT_ID, "corpus_artifact_digest": CORPUS_ARTIFACT_DIGEST,
    }
    if payload.get("schema") != SCHEMA or payload.get("source") != expected_source:
        raise ValueError("diagnostic schema or source binding drift")
    if payload.get("scope") != {"case_ids": CASES, "categories": CATEGORIES, "selection": "five_lowest_true_alpha_iou_cases"}:
        raise ValueError("diagnostic scope drift")
    observations = payload.get("observations")
    if not isinstance(observations, list) or [r.get("case_id") for r in observations] != CASES:
        raise ValueError("diagnostic observations incomplete")
    max_gap = 0.0
    for row in observations:
        case_id = row["case_id"]
        if row.get("category") != CATEGORIES[case_id] or row.get("source_has_alpha") is not True:
            raise ValueError("diagnostic case/category drift")
        if row.get("exact_evaluator_completed") is not True or row.get("metric_source") != "exact_final_artifact":
            raise ValueError("diagnostic evaluator binding drift")
        codes = row.get("hard_fail_codes")
        if not isinstance(codes, list) or not {"alpha_iou_below_min", "alpha_mae_above_max"}.issubset(codes):
            raise ValueError("diagnostic alpha failure signature drift")
        values = row.get("alpha_iou"), row.get("source_alpha_soft_coverage"), row.get("absolute_signature_gap")
        if not all(_finite(v) for v in values):
            raise ValueError("diagnostic non-finite metric")
        calculated = abs(float(values[0]) - float(values[1]))
        if abs(calculated - float(values[2])) > 1e-12 or calculated > 1e-5 or row.get("opaque_canvas_signature") is not True:
            raise ValueError("diagnostic signature gap mismatch")
        max_gap = max(max_gap, calculated)
    diagnosis = payload.get("diagnosis", {})
    if diagnosis.get("status") != "proven_by_live_selected_svg_render" or diagnosis.get("signature") != "alpha_iou_equals_source_alpha_soft_coverage" or diagnosis.get("hypothesis") != "selected_svg_render_alpha_is_full_canvas_opaque":
        raise ValueError("diagnosis status/signature drift")
    if diagnosis.get("signature_case_count") != len(CASES) or not _finite(diagnosis.get("max_absolute_gap")) or abs(float(diagnosis["max_absolute_gap"]) - max_gap) > 1e-12:
        raise ValueError("diagnosis aggregate drift")
    if diagnosis.get("production_fix_authorized") is not False:
        raise ValueError("diagnostics cannot authorize production")
    live = payload.get("live_confirmation", {})
    if live.get("pr_head_sha") != LIVE_PR_HEAD_SHA or live.get("workflow_run_id") != LIVE_WORKFLOW_RUN_ID or live.get("schema") != LIVE_SCHEMA:
        raise ValueError("live confirmation binding drift")
    rows = live.get("cases")
    if not isinstance(rows, list) or [r.get("case_id") for r in rows] != CASES:
        raise ValueError("live confirmation cases incomplete")
    for row, case_id in zip(rows, CASES):
        _validate_live_summary(row, case_id)
    if payload.get("release_decision") != "no_go" or payload.get("rfv4_allowed") is not False:
        raise ValueError("release/RFV-4 decision drift")
    if _PATH_OR_SECRET.search(json.dumps(payload, sort_keys=True, ensure_ascii=True)):
        raise ValueError("path, secret, runner location or traceback leaked")


def run_live(corpus_root: Path, case_id: str, work_root: Path, out_path: Path) -> dict[str, Any]:
    from app.pipeline_entry import run_pipeline
    from engine.regression.rfv3_measurement_runner import load_qualification_cases
    if case_id not in CASES:
        raise ValueError("case is outside RFV-3D2 diagnostic scope")
    case = {c.case_id: c for c in load_qualification_cases(corpus_root)}[case_id]
    source_path = (corpus_root / case.source_path).resolve()
    job_dir = (work_root / case_id).resolve(); job_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        result = run_pipeline(opened.convert("RGBA"), source_path, "auto", job_dir)
    svg_path = Path((result.get("best") or {}).get("svg_path") or "")
    if not svg_path.is_file():
        raise RuntimeError("pipeline selected SVG is missing")
    payload = diagnose_case(source_path, svg_path, case_id=case_id, category=CATEGORIES[case_id])
    payload.update({"engine_version": SOURCE_MAIN_SHA, "source_sha256": case.source_sha256, "mode_used": result.get("mode_used"), "production_fix_authorized": False, "release_decision": "no_go", "rfv4_allowed": False})
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFV-3D2 alpha/mask diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify"); verify.add_argument("--evidence", type=Path, required=True)
    live = sub.add_parser("live"); live.add_argument("--corpus-root", type=Path, required=True); live.add_argument("--case-id", required=True); live.add_argument("--work-root", type=Path, required=True); live.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "verify":
            validate_evidence(json.loads(args.evidence.read_text(encoding="utf-8")))
            result = {"status": "verified", "schema": SCHEMA, "case_count": len(CASES)}
        else:
            payload = run_live(args.corpus_root, args.case_id, args.work_root, args.out)
            result = {"status": "completed", "case_id": args.case_id, "diagnosis": payload["diagnosis"]}
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)[:200]}, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
