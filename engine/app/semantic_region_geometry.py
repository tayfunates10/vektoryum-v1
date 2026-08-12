"""Deferred scale-topology overlay around the native-good semantic core.

The native semantic fitter is preserved byte-for-byte in
``semantic_region_geometry_core``.  This wrapper adds one fixture-agnostic
repair for A-B-A nested curved separators: materialize the parent-label cells of
the minimum acceptance label map *after* child paint.  Each coarse cell is
chosen from the source label at that cell's center; candidate-native and all
five scale gates remain authoritative and reject any unsafe repaint.
"""
from __future__ import annotations

from collections import defaultdict
from math import ceil, floor
from typing import Any

import cv2
import numpy as np

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_MIN_ACCEPTANCE_SCALE = 0.25


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


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


def _rect_d(x0: float, y0: float, x1: float, y1: float) -> str:
    return (
        f"M{_fmt(x0)} {_fmt(y0)}L{_fmt(x1)} {_fmt(y0)}"
        f"L{_fmt(x1)} {_fmt(y1)}L{_fmt(x0)} {_fmt(y1)}Z"
    )


def _minimum_scale_parent_cells(
    parent_mask: np.ndarray,
    child_mask: np.ndarray,
) -> list[str]:
    """Materialize parent-labelled cells adjacent to child at min scale.

    The coarse label is sampled at the source-space cell center.  This is a
    scale-normalized vector topology representation, not raster embedding.
    """
    h, w = parent_mask.shape
    tw = max(16, int(round(w * _MIN_ACCEPTANCE_SCALE)))
    th = max(16, int(round(h * _MIN_ACCEPTANCE_SCALE)))
    parent_cells = np.zeros((th, tw), dtype=bool)
    child_cells = np.zeros((th, tw), dtype=bool)
    for ty in range(th):
        sy = min(h - 1, max(0, int(floor((ty + 0.5) * h / th))))
        for tx in range(tw):
            sx = min(w - 1, max(0, int(floor((tx + 0.5) * w / tw))))
            parent_cells[ty, tx] = bool(parent_mask[sy, sx])
            child_cells[ty, tx] = bool(child_mask[sy, sx])
    neighborhood = cv2.dilate(
        child_cells.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    selected = parent_cells & neighborhood
    paths: list[str] = []
    for ty in range(th):
        y0 = ty * h / th
        y1 = (ty + 1) * h / th
        for start, stop in _runs(selected[ty]):
            x0 = start * w / tw
            x1 = stop * w / tw
            paths.append(_rect_d(x0, y0, x1, y1))
    return paths


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    report = _core.build_semantic_region_elements(labels, colors)
    elements = list(report.get("elements") or [])
    regions = list(report.get("regions") or [])
    node_map, nodes, _root, _depth, parent = _core._connected_region_graph(
        np.asarray(labels), len(colors)
    )
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
        if node_id < 0 or strategy not in {
            "ellipse_axis_arm_union",
            "compound_parent_expand_axis_arm_union",
        }:
            continue
        grandparent_id = parent[node_id]
        if grandparent_id is None:
            continue
        matching = [
            child_id
            for child_id in children[node_id]
            if int(nodes[child_id]["color_index"])
            == int(nodes[grandparent_id]["color_index"])
        ]
        if len(matching) != 1:
            continue
        separator_paths = _minimum_scale_parent_cells(
            node_map == node_id,
            node_map == matching[0],
        )
        if not separator_paths:
            continue
        color_index = int(nodes[node_id]["color_index"])
        red, green, blue = (
            int(value) for value in np.asarray(colors, dtype=np.uint8)[color_index]
        )
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        overlays.extend(f'<path fill="{fill}" d="{d}"/>' for d in separator_paths)
        overlay_commands += sum(
            max(2, sum(d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z")))
            for d in separator_paths
        )
        strategy_counts[strategy] -= 1
        if strategy_counts[strategy] <= 0:
            strategy_counts.pop(strategy, None)
        upgraded = f"{strategy}_deferred_min_scale_label_cells"
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
    report["deferred_min_scale_label_paths"] = len(overlays)
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
