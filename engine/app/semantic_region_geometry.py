"""Production-renderer calibration wrapper for fail-closed semantic geometry."""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_primitive_geometry import calibrate_primitives

SemanticRegionFitError = _impl.SemanticRegionFitError
_CACHE_MAX = 12
_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    key = _impl._key(labels, colors)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return copy.deepcopy(cached)

    report = _impl._core.build_semantic_region_elements(labels, colors)
    base = list(report.get("elements") or [])
    regions = list(report.get("regions") or [])
    calibrated, primitive_report, repair_models = calibrate_primitives(
        labels, colors, base, regions
    )
    repaired, repair_report = _impl._repair(
        labels, colors, calibrated, regions, repair_models
    )

    primitive_delta = len(calibrated) - len(base)
    repair_delta = len(repaired) - len(calibrated)
    total_delta = len(repaired) - len(base)
    if total_delta:
        strategies = dict(report.get("strategy_counts") or {})
        if primitive_delta:
            strategies["five_scale_semantic_primitive_calibration"] = primitive_delta
        if repair_delta:
            strategies["render_feedback_topology_anchor"] = repair_delta
        report["strategy_counts"] = dict(sorted(strategies.items()))
        report["path_count"] = int(report.get("path_count") or 0) + total_delta
        report["node_count"] = int(report.get("node_count") or 0) + 2 * total_delta

    report["elements"] = repaired
    report["five_scale_semantic_primitive_calibration"] = primitive_report
    report["five_scale_semantic_primitive_changed"] = sum(
        1 for item in primitive_report if bool(item.get("changed"))
    )
    report["scale_topology_repair"] = repair_report
    report["scale_topology_anchor_count"] = repair_delta

    _CACHE[key] = copy.deepcopy(report)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
