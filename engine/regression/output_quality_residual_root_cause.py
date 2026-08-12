"""Residual-aware root-cause wrapper for the production output diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine.regression import output_quality_residual_suite as residual_suite
from engine.regression import output_quality_root_cause as base_root_cause

SCHEMA = "vektoryum-output-quality-residual-root-cause-v1"
_BASE_INFER_ROOT_CAUSES = base_root_cause.infer_root_causes


def _add(
    causes: list[dict[str, str]],
    code: str,
    confidence: str,
    evidence: str,
) -> None:
    if code not in {item.get("code") for item in causes}:
        causes.append({"code": code, "confidence": confidence, "evidence": evidence})


def infer_root_causes(case_result: dict[str, Any]) -> list[dict[str, str]]:
    """Extend existing hypotheses with evidence from independent residual axes."""
    causes = list(_BASE_INFER_ROOT_CAUSES(case_result))
    residual = case_result.get("residual") or {}
    multi_scale = case_result.get("multi_scale") or {}
    metrics = case_result.get("metrics") or {}
    diagnostics = case_result.get("pipeline_diagnostics") or []
    primary = diagnostics[0] if diagnostics else {}
    pipeline_error = str(case_result.get("pipeline_error") or "")

    source_recall = residual.get("source_component_recall")
    small_recall = residual.get("small_component_recall")
    min_component_iou = residual.get("min_component_iou")
    render_precision = residual.get("render_component_precision")
    boundary_p95 = residual.get("boundary_p95_px")
    edge_error = residual.get("edge_error_ratio")
    interior_error = residual.get("interior_error_ratio")
    de_p95 = residual.get("de00_p95")
    visible_ratio = residual.get("visible_error_ratio")

    if isinstance(source_recall, (int, float)) and source_recall < 1.0:
        _add(
            causes,
            "connected_component_loss",
            "high",
            f"Residual source component recall is {float(source_recall):.6f}<1.0.",
        )
    if isinstance(small_recall, (int, float)) and small_recall < 1.0:
        _add(
            causes,
            "micro_detail_loss",
            "high",
            f"Residual small-component recall is {float(small_recall):.6f}<1.0.",
        )
    aggregate_iou = metrics.get("min_component_iou")
    if (
        isinstance(min_component_iou, (int, float))
        and min_component_iou < 0.95
        and isinstance(aggregate_iou, (int, float))
        and aggregate_iou >= 0.95
    ):
        _add(
            causes,
            "aggregate_component_metric_blind_spot",
            "high",
            "The legacy component signal passes while true connected-component minimum IoU fails.",
        )
    if isinstance(render_precision, (int, float)) and render_precision < 1.0:
        _add(
            causes,
            "spurious_render_component",
            "high",
            f"Render component precision is {float(render_precision):.6f}<1.0.",
        )
    if isinstance(boundary_p95, (int, float)) and boundary_p95 > 0.75:
        _add(
            causes,
            "color_boundary_drift",
            "high",
            f"Symmetric color-boundary p95 is {float(boundary_p95):.6f}px>0.75px.",
        )
    if (
        isinstance(interior_error, (int, float))
        and isinstance(edge_error, (int, float))
        and interior_error > edge_error
        and isinstance(visible_ratio, (int, float))
        and visible_ratio > 0.01
    ):
        _add(
            causes,
            "interior_tone_residual",
            "medium",
            "Visible residual is dominated by interior pixels rather than the source edge band.",
        )

    min_multiscale_recall = multi_scale.get("min_source_component_recall")
    min_multiscale_small = multi_scale.get("min_small_component_recall")
    boundary_excess = multi_scale.get("max_normalized_boundary_excess_px")
    if (
        (isinstance(min_multiscale_recall, (int, float)) and min_multiscale_recall < 1.0)
        or (isinstance(min_multiscale_small, (int, float)) and min_multiscale_small < 1.0)
    ):
        _add(
            causes,
            "multi_scale_detail_instability",
            "high",
            "One or more rasterization scales lose source or small connected components.",
        )
    if isinstance(boundary_excess, (int, float)) and boundary_excess > 0.25:
        _add(
            causes,
            "multi_scale_boundary_instability",
            "high",
            f"Normalized multi-scale boundary excess is {float(boundary_excess):.6f}px>0.25px.",
        )

    alpha = residual.get("alpha") or {}
    alpha_iou = alpha.get("alpha_iou")
    alpha_mae = alpha.get("alpha_mae")
    if (
        residual.get("source_has_alpha") is True
        and isinstance(alpha_iou, (int, float))
        and isinstance(alpha_mae, (int, float))
        and alpha_iou >= 0.995
        and alpha_mae <= 0.005
        and (
            (isinstance(de_p95, (int, float)) and de_p95 > 2.3)
            or (isinstance(visible_ratio, (int, float)) and visible_ratio > 0.01)
        )
    ):
        _add(
            causes,
            "alpha_mask_pass_visible_composite_fail",
            "high",
            "Alpha geometry passes while visible composite color residual remains outside tolerance.",
        )

    if (
        "source_alpha_candidate_knockout_byte_budget_rejected:" in pipeline_error
        or "source_alpha_mask_rectangle_budget_exceeded:" in pipeline_error
    ):
        _add(
            causes,
            "alpha_fallback_exhausted",
            "high",
            "All safe alpha candidates exhausted before a budget-admissible downloadable artifact was selected.",
        )

    best = primary.get("best") or {}
    best_details = best.get("score_details") or {}
    if best:
        best_fidelity = best.get("fidelity_score")
        best_total = best.get("total_score")
        best_paths = best_details.get("path_count")
        best_nodes = best_details.get("node_count")
        for candidate in primary.get("candidates") or []:
            if candidate.get("name") == best.get("name"):
                continue
            if candidate.get("final_eligible") is False:
                continue
            details = candidate.get("score_details") or {}
            values = (
                candidate.get("fidelity_score"),
                candidate.get("total_score"),
                details.get("path_count"),
                details.get("node_count"),
                best_fidelity,
                best_total,
                best_paths,
                best_nodes,
            )
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            if (
                float(candidate["fidelity_score"]) > float(best_fidelity)
                and float(candidate["total_score"]) > float(best_total)
                and float(details["path_count"]) <= float(best_paths)
                and float(details["node_count"]) <= float(best_nodes)
            ):
                _add(
                    causes,
                    "pareto_dominated_winner",
                    "high",
                    "Another final-eligible candidate improves fidelity and total score without increasing path or node complexity.",
                )
                break

    return causes


def run_root_cause_suite(
    output_root: Path,
    *,
    engine_version: str,
    repeat_count: int,
    fail_on: str,
    case_limit: int | None = None,
) -> dict[str, Any]:
    original_suite = base_root_cause.suite
    original_infer = base_root_cause.infer_root_causes
    base_root_cause.suite = residual_suite
    base_root_cause.infer_root_causes = infer_root_causes
    try:
        report = base_root_cause.run_root_cause_suite(
            output_root,
            engine_version=engine_version,
            repeat_count=repeat_count,
            fail_on=fail_on,
            case_limit=case_limit,
        )
    finally:
        base_root_cause.infer_root_causes = original_infer
        base_root_cause.suite = original_suite

    cases = report.get("cases") or []
    pass_count = sum(1 for item in cases if (item.get("near_zero") or {}).get("ready") is True)
    measured_count = sum(1 for item in cases if item.get("near_zero_measured") is True)
    contracts = [item.get("near_zero") for item in cases]
    report.update(
        {
            "schema": SCHEMA,
            "near_zero_pass_count": pass_count,
            "near_zero_measured_count": measured_count,
            "near_zero_blocker_histogram": residual_suite.blocker_histogram(contracts),
            "all_near_zero": bool(cases) and pass_count == len(cases),
        }
    )
    output_root = Path(output_root)
    (output_root / base_root_cause.ROOT_CAUSE_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_root / base_root_cause.ROOT_CAUSE_SUMMARY).open("a", encoding="utf-8") as handle:
        handle.write("\n## Near-zero residual diagnosis\n\n")
        handle.write(f"- Near-zero: **{pass_count}/{len(cases)}**\n")
        handle.write(f"- Residual measured: **{measured_count}/{len(cases)}**\n")
        handle.write(
            f"- Blockers: `{json.dumps(report['near_zero_blocker_histogram'], sort_keys=True)}`\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument(
        "--fail-on",
        choices=("none", "structural", "hard", "near-zero"),
        default="structural",
    )
    args = parser.parse_args()
    report = run_root_cause_suite(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        fail_on=args.fail_on,
        case_limit=args.case_limit,
    )
    failed = residual_suite.should_fail(report, args.fail_on)
    print(
        json.dumps(
            {
                "status": "failed" if failed else "ok",
                "schema": report["schema"],
                "case_count": report["case_count"],
                "repeat_count": report["repeat_count"],
                "all_deterministic": report["all_deterministic"],
                "near_zero_pass_count": report["near_zero_pass_count"],
                "near_zero_measured_count": report["near_zero_measured_count"],
                "near_zero_blocker_histogram": report["near_zero_blocker_histogram"],
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
