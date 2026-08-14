"""Final bounded five-scale refinement for source-derived semantic geometry.

This pass reuses only generic primitive models and candidates already inferred
from connected source masks by the production geometry pipeline. It exists to
resolve renderer phase cases where an earlier conservative pass rejected a
five-scale-better candidate solely because native rasterization moved by a few
pixels. External acceptance thresholds, policies, corpus, palette caps, shared
budgets and vector-only requirements are unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app import scale_stable_discrete_optimizer as _disc
from app import scale_stable_parent_geometry as _parent

_NATIVE_MISMATCH_SLACK = 0.00025


def _flatten(parts: list[list[str]]) -> list[str]:
    return [element for part in parts for element in part]


def _scene_safe(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if float(after["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
        return False
    if float(after["native_mismatch"]) > float(before["native_mismatch"]) + _NATIVE_MISMATCH_SLACK:
        return False
    return not any(
        int(current) > int(previous)
        for current, previous in zip(
            after["lineage_errors"], before["lineage_errors"], strict=True
        )
    )


def _eligible_leaf_model(model: dict[str, Any]) -> bool:
    kind = str(model.get("kind") or "")
    if kind == "ellipse":
        return True
    if kind != "rect":
        return False
    width = float(model.get("width") or 0.0)
    height = float(model.get("height") or 0.0)
    if min(width, height) <= 0.0 or max(width, height) > 8.0:
        return False
    return max(width, height) / max(1e-9, min(width, height)) <= 1.25


def _candidate_parts(model: dict[str, Any], family: str, fill: str) -> list[list[str]]:
    kind = str(model.get("kind") or "")
    if family == "nested_parent" and kind == "rounded_rect":
        return _disc._rounded_parent_candidates(model, fill)
    if family == "nested_parent" and kind == "ellipse":
        return _disc._parent_ellipse_candidates(model, fill)
    if kind == "ellipse":
        return _disc._ellipse_candidates(model, fill)
    if kind == "rect":
        return _disc._compact_rect_candidates(model, fill)
    return []


def refine_final_geometry(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
    primitive_report: list[dict[str, Any]],
    *prior_reports: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    node_map, nodes, _root, _depth, parent = _impl._core._connected_region_graph(
        labels, len(colors)
    )
    child_count = [0 for _ in nodes]
    for parent_id in parent:
        if parent_id is not None:
            child_count[int(parent_id)] += 1
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]

    primitive_by_node = {
        int(item["node_id"]): item
        for item in primitive_report
        if isinstance(item, dict) and "node_id" in item
    }
    delta_by_node: dict[int, int] = {}
    for collection in (primitive_report, *prior_reports):
        for item in collection:
            if not isinstance(item, dict) or "node_id" not in item:
                continue
            node_id = int(item["node_id"])
            delta_by_node[node_id] = delta_by_node.get(node_id, 0) + int(
                item.get("element_delta") or 0
            )

    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        node_id = int(region.get("node_id", -1))
        count = int(region.get("path_count") or 0) + delta_by_node.get(node_id, 0)
        parts.append(list(elements[cursor:cursor + count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), []

    sh, sw = labels.shape
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        node_id = int(region.get("node_id", -1))
        if node_id < 0 or node_id >= len(nodes) or not parts[index]:
            continue

        model: dict[str, Any] | None = None
        family: str | None = None
        prior = primitive_by_node.get(node_id)
        if prior is not None and child_count[node_id] == 0:
            proposed = dict(prior.get("model") or {})
            if _eligible_leaf_model(proposed):
                model = proposed
                family = "leaf_compact"

        if model is None and child_count[node_id] > 0 and len(parts[index]) == 1:
            source_mask = node_map == node_id
            proposed = _parent._rounded_model(source_mask)
            if proposed is None:
                proposed = _parent._ellipse_model(source_mask)
            if proposed is not None:
                model = dict(proposed)
                family = "nested_parent"

        if model is None or family is None:
            continue

        color_index = int(region["color_index"])
        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        base_part = list(parts[index])
        isolated_before = _disc._isolated_metrics(model, base_part, fill, sw, sh)
        scene_before = _disc._scene_metrics(
            labels, colors, _flatten(parts), node_map, node_id,
            color_index, source_counts,
        )
        if isolated_before is None or scene_before is None:
            continue
        if (
            int(isolated_before["component_error"]) == 0
            and float(isolated_before["min_iou"]) >= 0.995
        ):
            continue

        best_part = base_part
        best_isolated = isolated_before
        best_scene = scene_before
        best_key = _disc._isolated_score(isolated_before, len(base_part)) + (
            float(scene_before["native_mismatch"]),
        )
        for candidate in _candidate_parts(model, family, fill):
            isolated = _disc._isolated_metrics(model, candidate, fill, sw, sh)
            if isolated is None:
                continue
            if float(isolated["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            trial_parts = list(parts)
            trial_parts[index] = list(candidate)
            scene = _disc._scene_metrics(
                labels, colors, _flatten(trial_parts), node_map, node_id,
                color_index, source_counts,
            )
            if scene is None or not _scene_safe(scene_before, scene):
                continue
            key = _disc._isolated_score(isolated, len(candidate)) + (
                float(scene["native_mismatch"]),
            )
            if key < best_key:
                best_key = key
                best_part = list(candidate)
                best_isolated = isolated
                best_scene = scene

        changed = best_part != base_part
        if changed:
            parts[index] = best_part
        reports.append({
            "node_id": node_id,
            "color_index": color_index,
            "family": family,
            "model": model,
            "changed": changed,
            "element_delta": len(best_part) - len(base_part),
            "before": {"isolated": isolated_before, "scene": scene_before},
            "after": {"isolated": best_isolated, "scene": best_scene},
        })

    return _flatten(parts), reports


__all__ = ["refine_final_geometry"]
