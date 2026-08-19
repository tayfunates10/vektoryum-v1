"""Final bounded five-scale refinement for source-derived semantic geometry.

This pass evaluates only candidate families not already exhausted by the prior
source-phase pass: non-scaling cardinal anchors for leaf ellipses and compact
nested ellipse phase corrections.  Ellipse models are reconstructed directly
from connected source masks so scoring follows Pillow's source semantics rather
than a prior raster-cell approximation.  External acceptance thresholds,
policies, corpus, palette caps, shared budgets and vector-only requirements are
unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app import scale_stable_discrete_optimizer as _disc
from app import scale_stable_source_phase as _phase

_NATIVE_MISMATCH_SLACK = 0.00025
_SCENE_CANDIDATE_LIMIT = 3


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


def _ellipse_model_from_source(mask: np.ndarray, *, parent: bool) -> dict[str, Any] | None:
    raw = _phase._endpoint_parent_ellipse(mask) if parent else _phase._exact_leaf_ellipse(mask)
    if raw is None:
        return None
    if str(raw.get("kind") or "") == "center_radius_ellipse":
        return {
            "kind": "ellipse",
            "cx": float(raw["cx"]),
            "cy": float(raw["cy"]),
            "rx": float(raw["rx"]),
            "ry": float(raw["ry"]),
        }
    if str(raw.get("kind") or "") == "endpoint_ellipse":
        x0, y0, x1, y1 = (float(raw[key]) for key in ("x0", "y0", "x1", "y1"))
        return {
            "kind": "ellipse",
            "cx": (x0 + x1) / 2.0,
            "cy": (y0 + y1) / 2.0,
            "rx": (x1 - x0) / 2.0,
            "ry": (y1 - y0) / 2.0,
        }
    return None


def _candidate_parts(model: dict[str, Any], family: str, fill: str) -> list[list[str]]:
    if family == "leaf_ellipse":
        # Single-element ellipse phase candidates are already covered upstream.
        return [part for part in _disc._ellipse_candidates(model, fill) if len(part) > 1]
    if family == "nested_ellipse":
        return _disc._parent_ellipse_candidates(model, fill)
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

        source_mask = node_map == node_id
        model: dict[str, Any] | None = None
        family: str | None = None
        if child_count[node_id] == 0:
            model = _ellipse_model_from_source(source_mask, parent=False)
            if model is not None:
                family = "leaf_ellipse"
        elif len(parts[index]) == 1:
            model = _ellipse_model_from_source(source_mask, parent=True)
            if model is not None:
                family = "nested_ellipse"

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

        base_key = _disc._isolated_score(isolated_before, len(base_part))
        ranked: list[tuple[tuple[float, float, float, float, int], list[str], dict[str, Any]]] = []
        for candidate in _candidate_parts(model, family, fill):
            isolated = _disc._isolated_metrics(model, candidate, fill, sw, sh)
            if isolated is None:
                continue
            if float(isolated["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            key = _disc._isolated_score(isolated, len(candidate))
            if key < base_key:
                ranked.append((key, list(candidate), isolated))
        ranked.sort(key=lambda item: item[0])

        best_part = base_part
        best_isolated = isolated_before
        best_scene = scene_before
        for _key, candidate, isolated in ranked[:_SCENE_CANDIDATE_LIMIT]:
            trial_parts = list(parts)
            trial_parts[index] = candidate
            scene = _disc._scene_metrics(
                labels, colors, _flatten(trial_parts), node_map, node_id,
                color_index, source_counts,
            )
            if scene is not None and _scene_safe(scene_before, scene):
                best_part = candidate
                best_isolated = isolated
                best_scene = scene
                break

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
