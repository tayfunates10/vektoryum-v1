"""Five-scale renderer selection for leaf line and micro-rectangle geometry."""
from __future__ import annotations

from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_primitive_geometry import candidate_elements


def _leaf_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    kind = str(model.get("kind") or "")
    if kind == "line":
        return candidate_elements(model, fill)
    if kind == "rect":
        converted = {
            "kind": "small_rect",
            "x": model["x"],
            "y": model["y"],
            "width": model["width"],
            "height": model["height"],
        }
        return candidate_elements(converted, fill)
    return _impl._candidates(model, fill)


def _score(metrics: dict[str, Any], count: int) -> tuple[float, float, float, float, int]:
    lineage = [int(value) for value in metrics["lineage_errors"]]
    return (
        float(sum(lineage)),
        -float(metrics["min_scale_iou"]),
        -float(metrics["mean_scale_iou"]),
        float(metrics["native_mismatch"]),
        int(count),
    )


def regularize(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    node_map, nodes, _root, _depth, _parent = _impl._core._connected_region_graph(
        labels, len(colors)
    )
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]
    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        count = int(region.get("path_count") or 0)
        parts.append(list(elements[cursor : cursor + count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), [], {}

    reports: list[dict[str, Any]] = []
    models: dict[int, dict[str, Any]] = {}
    for index, region in enumerate(regions):
        if len(parts[index]) != 1:
            continue
        node_id = int(region["node_id"])
        color_index = int(region["color_index"])
        strategy = str(region.get("strategy") or "")
        model = _impl._infer(node_map == node_id, strategy)
        if model is None:
            continue
        models[node_id] = model
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        base = _impl._metrics(
            labels,
            colors,
            _impl._flatten(parts),
            node_map,
            node_id,
            color_index,
            source_counts,
            model,
        )
        if base is None:
            continue
        best = base
        best_part = list(parts[index])
        best_score = _score(best, len(best_part))
        before_lineage = [int(value) for value in base["lineage_errors"]]

        for candidate in _leaf_candidates(model, fill):
            trial = list(parts)
            trial[index] = candidate
            metrics = _impl._metrics(
                labels,
                colors,
                _impl._flatten(trial),
                node_map,
                node_id,
                color_index,
                source_counts,
                model,
            )
            if metrics is None or float(metrics["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            after_lineage = [int(value) for value in metrics["lineage_errors"]]
            if after_lineage[2] != 0:
                continue
            if any(after > before for after, before in zip(after_lineage, before_lineage, strict=True)):
                continue
            if float(metrics["native_mismatch"]) > float(base["native_mismatch"]) + 0.00025:
                continue
            score = _score(metrics, len(candidate))
            if score < best_score:
                best = metrics
                best_part = list(candidate)
                best_score = score

        changed = best_part != parts[index]
        if changed:
            parts[index] = best_part
        reports.append(
            {
                "node_id": node_id,
                "color_index": color_index,
                "strategy": strategy,
                "model": model,
                "changed": changed,
                "before": base,
                "after": best,
                "element_delta": len(best_part) - 1,
            }
        )
    return _impl._flatten(parts), reports, models


__all__ = ["regularize"]
