"""AI-4 measurement contract composition: native residual + topology + verified scale source."""
from __future__ import annotations

from typing import Any, Iterable

from engine.regression import residual_error_metrics as base
from engine.regression.residual_topology_metrics import extend_near_zero_contract, measure_topology_residual

MEASUREMENT_CONTRACT_VERSION = "vektoryum-ai4-measurement-contract-v1"
HISTORICAL_FIXTURES = (
    "qa-gray-border-counter", "qa-shared-boundary", "qa-ring-holes", "qa-monoline",
    "qa-small-details", "qa-transparent-overlap", "qa-lowres-badge",
)
RECONSTRUCTED_FIXTURES = (
    "qa-micro-component-ladder", "qa-thin-negative-space", "qa-hard-stop-gradient",
    "qa-soft-alpha-shadow", "qa-neutral-tone-steps",
)


def measure_residual_error(source_rgba, render_rgba, *, palette_size: int = 8) -> dict[str, Any]:
    result = base.measure_residual_error(source_rgba, render_rgba, palette_size=palette_size)
    result["topology"] = measure_topology_residual(source_rgba, render_rgba, palette_size=palette_size)
    result["measurement_contract_version"] = MEASUREMENT_CONTRACT_VERSION
    return result


def build_near_zero_contract(
    residual: dict[str, Any] | None,
    *,
    deterministic: bool | None,
    structural_failures: Iterable[str],
    multi_scale: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = base.build_near_zero_contract(
        residual,
        deterministic=deterministic,
        structural_failures=structural_failures,
        multi_scale=multi_scale,
    )
    result = extend_near_zero_contract(contract, residual, multi_scale)
    result["measurement_contract_version"] = MEASUREMENT_CONTRACT_VERSION
    return result


def decorate_report(report: dict[str, Any]) -> dict[str, Any]:
    report["measurement_contract_version"] = MEASUREMENT_CONTRACT_VERSION
    report["fixture_provenance"] = {
        "historical": list(HISTORICAL_FIXTURES),
        "reconstructed": list(RECONSTRUCTED_FIXTURES),
    }
    # Keep the provenance statement machine-readable and impossible to mistake.
    report["fixture_provenance"]["statement"] = (
        "The five reconstructed fixtures are vulnerability tests and are not claimed "
        "byte-identical to historical local fixtures."
    )
    cases = report.get("cases") or []
    for case in cases:
        cid = case.get("case_id")
        case["fixture_provenance"] = "historical" if cid in HISTORICAL_FIXTURES else "reconstructed"
    multiscale = [case.get("multi_scale") or {} for case in cases]
    report["multi_scale_source_contract_verified"] = bool(cases) and all(
        item.get("source_contract_verified") is True for item in multiscale
    )
    report["multi_scale_release_evidence"] = (
        "VERIFIED" if report["multi_scale_source_contract_verified"] else "UNVERIFIED"
    )
    return report
