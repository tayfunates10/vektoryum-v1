"""AI-4 measurement contract: residual colour + observable semantic geometry/topology."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from engine.regression import residual_error_metrics as base
from engine.regression.residual_semantic_geometry import (
    SEMANTIC_GEOMETRY_POLICY_VERSION,
    measure_alpha_support_geometry,
    measure_continuous_hard_shape_components,
)
from engine.regression.residual_topology_metrics import extend_near_zero_contract, measure_topology_residual

MEASUREMENT_CONTRACT_VERSION = "vektoryum-ai4-measurement-contract-v2"
HISTORICAL_FIXTURES = (
    "qa-gray-border-counter", "qa-shared-boundary", "qa-ring-holes", "qa-monoline",
    "qa-small-details", "qa-transparent-overlap", "qa-lowres-badge",
)
RECONSTRUCTED_FIXTURES = (
    "qa-micro-component-ladder", "qa-thin-negative-space", "qa-hard-stop-gradient",
    "qa-soft-alpha-shadow", "qa-neutral-tone-steps",
)

_COMPONENT_KEYS = (
    "background_class_id",
    "min_component_area_px",
    "small_component_area_max_px",
    "source_component_count",
    "render_component_count",
    "matched_source_count",
    "matched_render_count",
    "unmatched_source_count",
    "unmatched_render_count",
    "unmatched_source_area_ratio",
    "unmatched_render_area_ratio",
    "source_component_recall",
    "render_component_precision",
    "small_component_count",
    "small_component_recall",
    "min_component_iou",
    "mean_component_iou",
    "area_weighted_component_iou",
    "worst_component",
)
_BOUNDARY_KEYS = ("boundary_mean_px", "boundary_p95_px", "boundary_max_px")


def _raw_sha256(rgba: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rgba, dtype=np.uint8).tobytes()).hexdigest()


@lru_cache(maxsize=8)
def _analytic_scene_registry(size: int) -> dict[str, tuple[str, np.ndarray, str, str]]:
    """Build one exact-hash registry per verified scale, then reuse it.

    This is identification only. No raster resize or fuzzy source matching is
    allowed: a source is eligible for semantic overrides only when its RGBA bytes
    exactly equal one verified analytic fixture at the same requested size.
    """
    from engine.regression import residual_multiscale_source as multiscale

    registry: dict[str, tuple[str, np.ndarray, str, str]] = {}
    for builder_name, builder in multiscale._ANALYTIC_BUILDERS.items():
        scene = builder(int(size))
        rgba = np.asarray(scene.image.convert("RGBA"), dtype=np.uint8)
        registry[_raw_sha256(rgba)] = (
            str(builder_name),
            np.asarray(scene.labels, dtype=np.uint8).copy(),
            str(scene.palette_mode),
            str(scene.geometry_sha256),
        )
    return registry


def _match_analytic_source(source_rgba: np.ndarray) -> tuple[str, np.ndarray, str, str] | None:
    source = np.asarray(source_rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4 or source.shape[0] != source.shape[1]:
        return None
    size = int(source.shape[0])
    if size not in (64, 128, 256, 512, 1024):
        return None
    return _analytic_scene_registry(size).get(_raw_sha256(source))


def _legacy_geometry_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in (*_COMPONENT_KEYS, *_BOUNDARY_KEYS)}


def measure_residual_error(source_rgba, render_rgba, *, palette_size: int = 8) -> dict[str, Any]:
    source = np.asarray(source_rgba, dtype=np.uint8)
    rendered = np.asarray(render_rgba, dtype=np.uint8)
    result = base.measure_residual_error(source, rendered, palette_size=palette_size)

    # Preserve the former palette-derived geometry/topology as explicit evidence.
    # A blocker removed by an observability correction therefore stays auditable.
    result["palette_geometry_measurement"] = _legacy_geometry_snapshot(result)
    palette_topology = measure_topology_residual(source, rendered, palette_size=palette_size)
    result["palette_topology_measurement"] = palette_topology
    result["topology"] = palette_topology
    result["semantic_geometry"] = {
        "policy_version": SEMANTIC_GEOMETRY_POLICY_VERSION,
        "applicable": False,
        "source_match": "unmatched",
        "overrides": [],
    }

    match = _match_analytic_source(source)
    if match is not None:
        builder_name, source_labels, palette_mode, geometry_sha256 = match
        semantic_meta = {
            "policy_version": SEMANTIC_GEOMETRY_POLICY_VERSION,
            "applicable": True,
            "source_match": "exact_verified_analytic_rgba",
            "builder_name": builder_name,
            "palette_mode": palette_mode,
            "source_geometry_sha256": geometry_sha256,
            "overrides": [],
        }

        if builder_name == "_draw_soft_alpha_shadow":
            # The soft shadow is a continuous alpha field, not a discrete colour
            # region. Alpha support/plane fidelity owns its topology; hard blue
            # and red shapes remain measurable as observable semantic components.
            support = measure_alpha_support_geometry(source, rendered)
            hard = measure_continuous_hard_shape_components(
                rendered,
                source_labels,
                excluded_source_labels=(1,),
            )
            result.update(hard["component"])
            result.update(support["boundary"])
            result["topology"] = support["topology"]
            semantic_meta.update(
                {
                    "overrides": ["component", "boundary", "topology"],
                    "reason": "continuous alpha support is measured by alpha fidelity, not palette topology",
                    "excluded_source_labels": [1],
                    "mapping": hard["mapping"],
                    "topology_observability": support["topology_observability"],
                }
            )

        elif builder_name == "_draw_transparent_overlap":
            # A flattened semi-transparent overlap does not uniquely expose the
            # hidden layer-to-layer colour boundary. Binary alpha support is
            # observable and is therefore the topology/boundary hard-stop here;
            # visible composite and alpha-plane gates remain unchanged.
            support = measure_alpha_support_geometry(source, rendered)
            result.update(support["boundary"])
            result["topology"] = support["topology"]
            semantic_meta.update(
                {
                    "overrides": ["boundary", "topology"],
                    "reason": "shared colour-layer boundary is not identifiable from flattened alpha composite",
                    "mapping": support["mapping"],
                    "topology_observability": support["topology_observability"],
                }
            )

        result["semantic_geometry"] = semantic_meta

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
    report["semantic_geometry_policy_version"] = SEMANTIC_GEOMETRY_POLICY_VERSION
    report["fixture_provenance"] = {
        "historical": list(HISTORICAL_FIXTURES),
        "reconstructed": list(RECONSTRUCTED_FIXTURES),
    }
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
