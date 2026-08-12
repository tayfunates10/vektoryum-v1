"""Repeat-aware root-cause analysis for the Vektoryum output diagnostic.

The base diagnostic owns fixture generation, evaluator metrics and evidence files.
This layer wraps the unchanged production entry point so each repeat also records
why the pipeline chose its mode and candidate. The captured data is deliberately
restricted to JSON-safe decision fields; source assets and internal absolute paths
are never published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from app.pipeline_entry import run_pipeline as production_run_pipeline
from engine.regression import output_quality_suite as suite

SCHEMA = "vektoryum-output-quality-root-cause-v1"
ROOT_CAUSE_FILENAME = "output-quality-root-cause.json"
ROOT_CAUSE_SUMMARY = "output-quality-root-cause.md"

_ANALYSIS_FIELDS = (
    "recommended_mode",
    "estimated_color_count",
    "likely_geometric_logo",
    "likely_color_logo",
    "likely_lineart",
    "has_gradient",
    "has_transparency",
    "edge_density",
    "width",
    "height",
)
_PREPROCESS_FIELDS = (
    "auto_color_count",
    "actual_color_count",
    "color_count",
    "processing_scale",
    "upscaled",
    "downscaled",
    "background_removed",
    "method",
)
_SCORE_FIELDS = (
    "path_count",
    "node_count",
    "unique_colors",
    "edge_f1",
    "has_bitmap",
    "geometry_score",
    "editability_score",
)
_REJECTED_JOURNAL_STATUSES = {"rolled_back", "failed", "budget_exhausted"}


def _safe_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 8)
    return None


def _safe_mapping(value: object, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for field in fields:
        scalar = _safe_scalar(value.get(field))
        if scalar is not None:
            output[field] = scalar
    return output


def _file_sha256(value: object) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    try:
        path = Path(value)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _journal_rejections(output: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    """Return candidate SHAs explicitly rejected by the production journal."""
    report = output.get("transform_journal") or {}
    rejected: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for stage in report.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        status = str(stage.get("status") or "")
        candidate_sha = stage.get("candidate_sha256")
        accepted_sha = stage.get("accepted_sha256")
        if (
            status not in _REJECTED_JOURNAL_STATUSES
            or not isinstance(candidate_sha, str)
            or candidate_sha == accepted_sha
        ):
            continue
        rejected.add(candidate_sha)
        evidence.append(
            {
                "stage_id": _safe_scalar(stage.get("stage_id")),
                "status": status,
                "candidate_sha256": candidate_sha,
                "accepted_sha256": _safe_scalar(accepted_sha),
                "reason_codes": [
                    str(code) for code in (stage.get("reason_codes") or [])
                    if isinstance(code, (str, int, float, bool))
                ],
            }
        )
    return rejected, evidence


def candidate_snapshot(
    candidate: object,
    *,
    rejected_candidate_shas: set[str] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    snapshot: dict[str, Any] = {}
    for field in (
        "name",
        "engine",
        "rendered_ok",
        "fidelity_score",
        "total_score",
        "selection_safe",
        "selection_disqualified",
    ):
        scalar = _safe_scalar(candidate.get(field))
        if scalar is not None:
            snapshot[field] = scalar
    details = _safe_mapping(candidate.get("score_details"), _SCORE_FIELDS)
    if details:
        snapshot["score_details"] = details
    svg_sha = _file_sha256(candidate.get("svg_path"))
    if svg_sha is not None:
        snapshot["svg_sha256"] = svg_sha
        snapshot["final_eligible"] = svg_sha not in (rejected_candidate_shas or set())
    else:
        # Missing diagnostic SHA is not proof of a rollback. Preserve the legacy
        # candidate visibility and let downstream fail-closed checks use the
        # explicit journal evidence when available.
        snapshot["final_eligible"] = True
    return snapshot or None


def pipeline_snapshot(output: object) -> dict[str, Any]:
    if not isinstance(output, dict):
        return {"capture_status": "invalid_pipeline_output"}
    rejected_shas, journal_rejections = _journal_rejections(output)
    scored = [
        candidate_snapshot(item, rejected_candidate_shas=rejected_shas)
        for item in (output.get("scored") or [])
    ]
    candidates = [item for item in scored if item]
    candidates.sort(
        key=lambda item: (
            -float(item.get("fidelity_score") or -1.0),
            -float(item.get("total_score") or -1.0),
            str(item.get("name") or ""),
        )
    )
    return {
        "capture_status": "ok",
        "mode_used": _safe_scalar(output.get("mode_used")),
        "mode_warning": _safe_scalar(output.get("mode_warning")),
        "selection_reason": _safe_scalar(output.get("selection_reason")),
        "analysis": _safe_mapping(output.get("analysis"), _ANALYSIS_FIELDS),
        "preprocess": _safe_mapping(output.get("preprocess_report"), _PREPROCESS_FIELDS),
        "best": candidate_snapshot(output.get("best"), rejected_candidate_shas=rejected_shas),
        "raw_best": candidate_snapshot(output.get("raw_best"), rejected_candidate_shas=rejected_shas),
        "candidate_count": len(candidates),
        "candidates": candidates[:16],
        "journal_rejections": journal_rejections,
    }


def infer_root_causes(case_result: dict[str, Any]) -> list[dict[str, str]]:
    """Return evidence-bound hypotheses; never claim certainty beyond metrics."""
    metrics = case_result.get("metrics") or {}
    diagnostics = case_result.get("pipeline_diagnostics") or []
    primary = diagnostics[0] if diagnostics else {}
    mode = primary.get("mode_used")
    best = primary.get("best") or {}
    causes: list[dict[str, str]] = []

    def add(code: str, confidence: str, evidence: str) -> None:
        if code not in {item["code"] for item in causes}:
            causes.append({"code": code, "confidence": confidence, "evidence": evidence})

    component_delta = metrics.get("component_delta")
    seam_ratio = metrics.get("seam_ratio")
    ssim = metrics.get("ssim")
    edge_f1 = metrics.get("edge_f1_1px")
    de_p95 = metrics.get("de00_p95")
    min_iou = metrics.get("min_component_iou")
    alpha_iou = metrics.get("alpha_iou")

    if mode == "single_color" and (
        (isinstance(component_delta, (int, float)) and component_delta > 0)
        or (isinstance(de_p95, (int, float)) and de_p95 > 8.0)
    ):
        add(
            "auto_mode_color_collapse",
            "high",
            "Auto mode selected single_color while topology/color evidence shows material loss.",
        )
    if isinstance(seam_ratio, (int, float)) and seam_ratio > 0.05:
        if isinstance(edge_f1, (int, float)) and edge_f1 >= 0.98 and isinstance(de_p95, (int, float)) and de_p95 <= 8.0:
            add(
                "seam_metric_light_region_sensitivity",
                "medium",
                "Seam ratio is high while edge geometry remains intact; near-white source regions are likely counted as missing foreground.",
            )
        else:
            add(
                "missing_or_collapsed_region",
                "high",
                "Large seam ratio is accompanied by weak edge/color fidelity.",
            )
    if (
        isinstance(ssim, (int, float))
        and ssim < 0.90
        and isinstance(edge_f1, (int, float))
        and edge_f1 >= 0.99
        and (component_delta in (0, 0.0))
        and isinstance(min_iou, (int, float))
        and min_iou >= 0.99
    ):
        add(
            "ssim_tone_canonicalization_sensitivity",
            "high",
            "Geometry/topology agree but SSIM fails, indicating tone or antialias canonicalization rather than shape loss.",
        )
    if isinstance(min_iou, (int, float)) and min_iou < 0.90 and not (
        isinstance(component_delta, (int, float)) and component_delta > 0
    ):
        add(
            "small_component_shape_drift",
            "medium",
            "Topology is preserved but the smallest component falls below the component-IoU target.",
        )
    if (
        isinstance(alpha_iou, (int, float))
        and alpha_iou >= 0.995
        and isinstance(edge_f1, (int, float))
        and edge_f1 < 0.95
    ):
        add(
            "alpha_preserved_color_boundary_drift",
            "high",
            "Alpha plane is accurate while visible edge/color metrics regress; compositing or gradient selection is implicated.",
        )
    if best.get("engine") == "gradient" and isinstance(de_p95, (int, float)) and de_p95 > 8.0:
        add(
            "gradient_candidate_color_transition_error",
            "high",
            "The selected gradient candidate exceeds the p95 color-error target.",
        )
    if not causes and case_result.get("severity") == "pass":
        add("no_material_failure_detected", "high", "All configured hard and soft gates passed.")
    if not causes and case_result.get("severity") in {"medium", "high"}:
        add("quality_gate_failure_requires_visual_review", "low", "Metrics failed but the current evidence does not isolate one stage.")
    return causes


def run_root_cause_suite(
    output_root: Path,
    *,
    engine_version: str,
    repeat_count: int,
    fail_on: str,
    case_limit: int | None = None,
) -> dict[str, Any]:
    if repeat_count < 2:
        raise ValueError("root-cause suite requires at least two repeats")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = suite.generate_cases(output_root / "generated")
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive when provided")
        cases = cases[:case_limit]

    case_results: list[dict[str, Any]] = []
    for case in cases:
        captures: list[dict[str, Any]] = []

        def capturing_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = production_run_pipeline(*args, **kwargs)
            captures.append(pipeline_snapshot(result))
            return result

        original = suite.run_pipeline
        suite.run_pipeline = capturing_pipeline
        try:
            case_result = suite.run_diagnostic_case(
                case,
                output_root=output_root,
                engine_version=engine_version,
                repeat_count=repeat_count,
            )
        finally:
            suite.run_pipeline = original
        case_result["pipeline_diagnostics"] = captures
        case_result["root_cause_hypotheses"] = infer_root_causes(case_result)
        case_results.append(case_result)

    base_report = suite.build_report(
        case_results,
        engine_version=engine_version,
        repeat_count=repeat_count,
        fail_on=fail_on,
    )
    suite.write_report(output_root, base_report)
    root_report = {
        "schema": SCHEMA,
        "engine_version": engine_version,
        "case_count": len(case_results),
        "repeat_count": repeat_count,
        "all_deterministic": base_report["all_deterministic"],
        "structural_failure_codes": base_report["structural_failure_codes"],
        "hard_failure_codes": base_report["hard_failure_codes"],
        "severity_counts": base_report["severity_counts"],
        "cases": base_report["cases"],
    }
    (output_root / ROOT_CAUSE_FILENAME).write_text(
        json.dumps(root_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_root_cause_summary(output_root, root_report)
    return root_report


def write_root_cause_summary(output_root: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Vektoryum output-quality root-cause analysis",
        "",
        f"- Cases: **{report['case_count']}**",
        f"- Repeats: **{report['repeat_count']}**",
        f"- Byte deterministic: **{report['all_deterministic']}**",
        f"- Structural failures: `{', '.join(report['structural_failure_codes']) or 'none'}`",
        "",
    ]
    for case in report["cases"]:
        diagnostics = case.get("pipeline_diagnostics") or []
        primary = diagnostics[0] if diagnostics else {}
        best = primary.get("best") or {}
        lines.extend(
            [
                f"## {case['case_id']} — {case['severity']}",
                "",
                f"- Mode: `{primary.get('mode_used', 'unknown')}`",
                f"- Winner: `{best.get('name', 'unknown')}` / `{best.get('engine', 'unknown')}`",
                f"- Selection: `{primary.get('selection_reason', 'unknown')}`",
                f"- Artifact deterministic: **{case.get('deterministic')}**",
                f"- Hard codes: `{', '.join(case.get('hard_fail_codes') or []) or 'none'}`",
            ]
        )
        for hypothesis in case.get("root_cause_hypotheses") or []:
            lines.append(
                f"- `{hypothesis['code']}` ({hypothesis['confidence']}): {hypothesis['evidence']}"
            )
        lines.append("")
    (Path(output_root) / ROOT_CAUSE_SUMMARY).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--fail-on", choices=("none", "structural", "hard"), default="structural")
    args = parser.parse_args()
    report = run_root_cause_suite(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        fail_on=args.fail_on,
        case_limit=args.case_limit,
    )
    failed = suite.should_fail(report, args.fail_on)
    print(
        json.dumps(
            {
                "status": "failed" if failed else "ok",
                "schema": report["schema"],
                "case_count": report["case_count"],
                "repeat_count": report["repeat_count"],
                "all_deterministic": report["all_deterministic"],
                "severity_counts": report["severity_counts"],
                "structural_failure_codes": report["structural_failure_codes"],
                "hard_failure_codes": report["hard_failure_codes"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
