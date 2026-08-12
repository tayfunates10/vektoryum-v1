"""Deferred minimum-scale topology overlay around the native-good semantic core."""
from __future__ import annotations

from collections import defaultdict
from math import ceil, floor
from typing import Any

import cv2
import numpy as np

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_MIN_ACCEPTANCE_SCALE = 0.25


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(np.r_[np.asarray(flags, dtype=bool), False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            output.append((start, index))
            start = None
    return output


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _rect_d(bounds: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bounds
    return f"M{_fmt(x0)} {_fmt(y0)}L{_fmt(x1)} {_fmt(y0)}L{_fmt(x1)} {_fmt(y1)}L{_fmt(x0)} {_fmt(y1)}Z"


def _minimum_scale_separator_paths(parent_mask: np.ndarray, child_mask: np.ndarray) -> list[str]:
    """Return min-scale cells whose complete source footprint is separator B."""
    h, w = parent_mask.shape
    tw = max(16, int(round(w * _MIN_ACCEPTANCE_SCALE)))
    th = max(16, int(round(h * _MIN_ACCEPTANCE_SCALE)))
    pure = np.zeros((th, tw), dtype=bool)
    child = np.zeros((th, tw), dtype=bool)
    x_bounds = [(int(floor(x * w / tw)), int(ceil((x + 1) * w / tw))) for x in range(tw)]
    y_bounds = [(int(floor(y * h / th)), int(ceil((y + 1) * h / th))) for y in range(th)]
    for ty, (y0, y1) in enumerate(y_bounds):
        for tx, (x0, x1) in enumerate(x_bounds):
            separator_block = parent_mask[y0:y1, x0:x1]
            child_block = child_mask[y0:y1, x0:x1]
            pure[ty, tx] = bool(separator_block.size and np.all(separator_block))
            child[ty, tx] = bool(np.any(child_block))
    neighborhood = cv2.dilate(child.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    cells = pure & neighborhood
    paths: list[str] = []
    for ty in range(th):
        y0, y1 = y_bounds[ty]
        for start, stop in _runs(cells[ty]):
            x0 = x_bounds[start][0]
            x1 = x_bounds[stop - 1][1]
            paths.append(_rect_d((float(x0), float(y0), float(x1), float(y1))))
    return paths


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    """Use native-good geometry and paint source-safe min-scale separators last."""
    report = _core.build_semantic_region_elements(labels, colors)
    elements = list(report.get("elements") or [])
    regions = list(report.get("regions") or [])
    node_map, nodes, _root, _depth, parent = _core._connected_region_graph(np.asarray(labels), len(colors))
    children: list[list[int]] = [[] for _ in nodes]
    for node_id, parent_id in enumerate(parent):
        if parent_id is not None:
            children[parent_id].append(node_id)

    overlays: list[str] = []
    overlay_commands = 0
    strategy_counts = defaultdict(int, report.get("strategy_counts") or {})
    for region in regions:
        node_id = int(region.get("node_id", -1))
        strategy = str(region.get("strategy") or "")
        if node_id < 0 or strategy not in {"ellipse_axis_arm_union", "compound_parent_expand_axis_arm_union"}:
            continue
        grandparent_id = parent[node_id]
        if grandparent_id is None or not children[node_id]:
            continue
        matching = [
            child_id
            for child_id in children[node_id]
            if int(nodes[child_id]["color_index"]) == int(nodes[grandparent_id]["color_index"])
        ]
        if len(matching) != 1:
            continue
        separator_paths = _minimum_scale_separator_paths(node_map == node_id, node_map == matching[0])
        if not separator_paths:
            continue
        color_index = int(nodes[node_id]["color_index"])
        red, green, blue = (int(value) for value in np.asarray(colors, dtype=np.uint8)[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        overlays.extend(f'<path fill="{fill}" d="{d}"/>' for d in separator_paths)
        overlay_commands += sum(max(2, sum(d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z"))) for d in separator_paths)
        strategy_counts[strategy] -= 1
        if strategy_counts[strategy] <= 0:
            strategy_counts.pop(strategy, None)
        upgraded = f"{strategy}_deferred_min_scale_separator"
        strategy_counts[upgraded] += 1
        region["strategy"] = upgraded
        region["minimum_scale_separator_paths"] = len(separator_paths)
        region["path_count"] = int(region.get("path_count") or 0) + len(separator_paths)

    elements.extend(overlays)
    report["elements"] = elements
    report["path_count"] = int(report.get("path_count") or 0) + len(overlays)
    report["node_count"] = int(report.get("node_count") or 0) + overlay_commands
    report["strategy_counts"] = dict(sorted(strategy_counts.items()))
    report["regions"] = regions
    report["deferred_min_scale_separator_paths"] = len(overlays)
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
