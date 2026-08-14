"""Production-renderer calibration wrapper for fail-closed semantic geometry."""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_discrete_optimizer import refine_scale_stable_geometry
from app.scale_stable_parent_geometry import calibrate_parent_regions
from app.scale_stable_primitive_geometry import calibrate_primitives
from app.scale_stable_seam_geometry import stabilize_nested_seams

SemanticRegionFitError = _impl.SemanticRegionFitError
_ELLIPSE_INSET_OFFSETS = (0.0, -0.10, -0.20, -0.30, 0.10, 0.20)
_CACHE_MAX = 12
_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _ellipse_element(mask: np.ndarray, fill: str, inset: float) -> str | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    rx = ((x1 - x0) / 2.0) - float(inset)
    ry = ((y1 - y0) / 2.0) - float(inset)
    if rx < 1.0 or ry < 1.0:
        return None
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    d = (
        f"M{_impl._fmt(cx-rx)} {_impl._fmt(cy)}"
        f"A{_impl._fmt(rx)} {_impl._fmt(ry)} 0 1 0 {_impl._fmt(cx+rx)} {_impl._fmt(cy)}"
        f"A{_impl._fmt(rx)} {_impl._fmt(ry)} 0 1 0 {_impl._fmt(cx-rx)} {_impl._fmt(cy)}Z"
    )
    return f'<path fill="{fill}" d="{d}"/>'


def _ellipse_metrics(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    node_map: np.ndarray,
    node_id: int,
    color_index: int,
    source_counts: list[int],
) -> dict[str, Any] | None:
    height, width = labels.shape
    native = _impl._render(width, height, elements, colors, width, height)
    if native is None:
        return None
    source_mask = node_map == int(node_id)
    native_iou = _impl._best_iou(native, source_mask, int(color_index))
    native_mismatch = float(np.mean(native != labels))
    native_counts = _impl._counts(native, len(colors))
    native_lineage_error = int(
        sum(
            abs(int(actual) - int(expected))
            for actual, expected in zip(native_counts, source_counts, strict=True)
        )
    )

    quarter_width = max(16, int(round(width * 0.25)))
    quarter_height = max(16, int(round(height * 0.25)))
    quarter = _impl._render(width, height, elements, colors, quarter_width, quarter_height)
    quarter_lineage_error: int | None = None
    quarter_present = False
    if quarter is not None:
        quarter_counts = _impl._counts(quarter, len(colors))
        quarter_lineage_error = int(
            sum(
                abs(int(actual) - int(expected))
                for actual, expected in zip(quarter_counts, source_counts, strict=True)
            )
        )
        projected = _impl._project(source_mask, quarter_width, quarter_height)
        quarter_present = bool(np.any((quarter == int(color_index)) & projected))

    return {
        "native_iou": float(native_iou),
        "native_mismatch": native_mismatch,
        "native_lineage_error": native_lineage_error,
        "quarter_lineage_error": quarter_lineage_error,
        "quarter_present": quarter_present,
    }


def _ellipse_score(
    metrics: dict[str, Any], inset: float, base_inset: float
) -> tuple[float, int, int, float, float, float]:
    iou = float(metrics["native_iou"])
    deficit = max(0.0, float(_impl._NATIVE_MIN_IOU) - iou)
    quarter_error = metrics.get("quarter_lineage_error")
    quarter_error_value = int(quarter_error) if quarter_error is not None else 10**6
    return (
        round(deficit, 9),
        quarter_error_value,
        0 if bool(metrics.get("quarter_present")) else 1,
        -iou,
        float(metrics["native_mismatch"]),
        abs(float(inset) - float(base_inset)),
    )


def _calibrate_ellipses(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    node_map, nodes, _root, _depth, _parent = _impl._core._connected_region_graph(
        labels, len(colors)
    )
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]
    out = list(elements)
    reports: list[dict[str, Any]] = []
    cursor = 0
    base_inset = float(getattr(_impl._core, "_ELLIPSE_INSET", 0.10))

    for region in regions:
        count = int(region.get("path_count") or 0)
        start = cursor
        cursor += count
        if count != 1 or str(region.get("strategy") or "") != "pixel_center_ellipse":
            continue
        node_id = int(region["node_id"])
        color_index = int(region["color_index"])
        source_mask = node_map == node_id
        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        current = _ellipse_metrics(
            labels, colors, out, node_map, node_id, color_index, source_counts
        )
        if current is None:
            continue
        best = current
        best_inset = base_inset
        best_element = out[start]
        best_score = _ellipse_score(best, best_inset, base_inset)

        for offset in _ELLIPSE_INSET_OFFSETS:
            inset = base_inset + float(offset)
            candidate = _ellipse_element(source_mask, fill, inset)
            if candidate is None:
                continue
            trial = list(out)
            trial[start] = candidate
            metrics = _ellipse_metrics(
                labels, colors, trial, node_map, node_id, color_index, source_counts
            )
            if metrics is None:
                continue
            if float(metrics["native_iou"]) + 1e-12 < float(current["native_iou"]):
                continue
            if int(metrics["native_lineage_error"]) > int(current["native_lineage_error"]):
                continue
            score = _ellipse_score(metrics, inset, base_inset)
            if score < best_score:
                best_score = score
                best = metrics
                best_inset = inset
                best_element = candidate

        out[start] = best_element
        reports.append(
            {
                "node_id": node_id,
                "color_index": color_index,
                "source_area": int(np.count_nonzero(source_mask)),
                "base_inset": round(base_inset, 4),
                "selected_inset": round(float(best_inset), 4),
                "changed": best_element != elements[start],
                "before": current,
                "after": best,
            }
        )
    return out, reports


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
    calibrated, ellipse_report = _calibrate_ellipses(labels, colors, base, regions)
    parented, parent_report = calibrate_parent_regions(
        labels, colors, calibrated, regions
    )
    regularized, primitive_report, models = calibrate_primitives(
        labels, colors, parented, regions
    )
    refined, refinement_report = refine_scale_stable_geometry(
        labels, colors, regularized, regions, primitive_report
    )
    seamed, seam_report = stabilize_nested_seams(labels, colors, refined)
    repaired, repair_report = _impl._repair(
        labels, colors, seamed, regions, models
    )

    primitive_delta = len(regularized) - len(base)
    refinement_delta = len(refined) - len(regularized)
    seam_delta = len(seamed) - len(refined)
    anchor_delta = len(repaired) - len(seamed)
    total_delta = len(repaired) - len(base)
    if total_delta:
        strategies = dict(report.get("strategy_counts") or {})
        if primitive_delta:
            strategies["five_scale_source_primitive_calibration"] = primitive_delta
        if refinement_delta:
            strategies["five_scale_contract_refinement"] = refinement_delta
        if seam_delta:
            strategies["five_scale_nested_seam_stabilization"] = seam_delta
        if anchor_delta:
            strategies["render_feedback_topology_anchor"] = anchor_delta
        report["strategy_counts"] = dict(sorted(strategies.items()))
        report["path_count"] = int(report.get("path_count") or 0) + total_delta
        report["node_count"] = int(report.get("node_count") or 0) + 2 * total_delta

    report["elements"] = repaired
    report["ellipse_renderer_calibration"] = ellipse_report
    report["ellipse_renderer_calibration_changed"] = sum(
        1 for item in ellipse_report if bool(item.get("changed"))
    )
    report["five_scale_parent_calibration"] = parent_report
    report["five_scale_parent_calibration_changed"] = sum(
        1 for item in parent_report if bool(item.get("changed"))
    )
    report["five_scale_source_primitive_calibration"] = primitive_report
    report["five_scale_source_primitive_changed"] = sum(
        1 for item in primitive_report if bool(item.get("changed"))
    )
    report["five_scale_contract_refinement"] = refinement_report
    report["five_scale_contract_refinement_changed"] = sum(
        1 for item in refinement_report if bool(item.get("changed"))
    )
    report["five_scale_nested_seam_stabilization"] = seam_report
    report["five_scale_nested_seam_count"] = seam_delta
    report["scale_topology_repair"] = repair_report
    report["scale_topology_anchor_count"] = anchor_delta

    _CACHE[key] = copy.deepcopy(report)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
