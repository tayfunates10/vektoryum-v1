"""AI-4 verified residual/multi-scale/topology diagnostic entry point."""
from __future__ import annotations

from typing import Any

from engine.regression import output_quality_residual_root_cause as root_cause
from engine.regression import output_quality_residual_suite as suite
from engine.regression import residual_measurement_contract as contract
from engine.regression import residual_multiscale_source as multiscale


def _install_contract() -> None:
    suite.measure_residual_error = contract.measure_residual_error
    suite.build_near_zero_contract = contract.build_near_zero_contract
    suite.measure_multiscale_svg = multiscale.measure_multiscale_svg_from_base
    multiscale.measure_residual_error = contract.measure_residual_error

    original_build_report = suite.build_report
    def build_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return contract.decorate_report(original_build_report(*args, **kwargs))
    suite.build_report = build_report

    original_infer = root_cause.infer_root_causes
    def infer_root_causes(case_result: dict[str, Any]) -> list[dict[str, str]]:
        causes = list(original_infer(case_result))
        codes = {item.get("code") for item in causes}
        near_zero = case_result.get("near_zero") or {}
        blockers = set(near_zero.get("blocker_codes") or [])
        if blockers & {
            "topology_metric_missing", "topology_hole_loss", "topology_ring_break", "topology_adjacency_loss",
            "shared_boundary_metric_missing",
            "shared_boundary_gap", "shared_boundary_double_line", "shared_boundary_overlap",
            "shared_boundary_drift", "shared_boundary_transition_recall",
        } and "topology_shared_boundary_instability" not in codes:
            causes.append({
                "code": "topology_shared_boundary_instability",
                "confidence": "high",
                "evidence": "Topology/shared-boundary hard-stop blocker is present in the AI-4 residual contract.",
            })
        if "multi_scale_source_contract_verified" in blockers and "multi_scale_source_contract_unverified" not in codes:
            causes.append({
                "code": "multi_scale_source_contract_unverified",
                "confidence": "high",
                "evidence": "Analytic normalized source contract did not verify at all requested scales.",
            })
        return causes
    root_cause.infer_root_causes = infer_root_causes


if __name__ == "__main__":
    _install_contract()
    root_cause.main()
