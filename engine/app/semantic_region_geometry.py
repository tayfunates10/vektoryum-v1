"""Deferred scale-topology overlay around the native-good semantic core.

The native semantic fitter is preserved byte-for-byte in
``semantic_region_geometry_core``.  This wrapper adds only one generic repair:
for an A-B-A nested curved separator whose B parent is a compound ellipse/arm
region, measure the B annulus from the source label map and paint its midline
*after* every child region.  The ring is genuine source-space SVG geometry,
contains no fixture routing, and the caller still rejects it unless native and
all five render scales pass the existing transaction.
"""
from __future__ import annotations

from collections import defaultdict
from math import cos, pi, sin
from typing import Any

import numpy as np

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_RAY_COUNT = 32
_RAY_STEP = 0.10
_SAFETY_MARGIN = 1.0


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _largest_cycle(mask: np.ndarray) -> list[tuple[int, int]]:
    cycles = _core._boundary_cycles(mask)
    if not cycles:
        raise SemanticRegionFitError("region_without_boundary")
    import cv2  # local: wrapper stays thin
    return max(
        cycles,
        key=lambda points: abs(
            float(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)))
        ),
    )


def _ellipse_like(mask: np.ndarray) -> bool:
    cycle = _largest_cycle(mask)
    return _core._ellipse_from_cycle(cycle) is not None


def _separator_thickness(parent_mask: np.ndarray, child_mask: np.ndarray) -> float | None:
    ys, xs = np.nonzero(child_mask)
    if not len(xs):
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = parent_mask.shape
    max_step = float(max(h, w))
    values: list[float] = []
    for index in range(_RAY_COUNT):
        angle = (2.0 * pi * index) / _RAY_COUNT
        dx, dy = cos(angle), sin(angle)
        entered: float | None = None
        left_child = False
        last_xy: tuple[int, int] | None = None
        for step in np.arange(0.0, max_step, _RAY_STEP):
            x = int(round(cx + dx * step))
            y = int(round(cy + dy * step))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if last_xy == (x, y):
                continue
            last_xy = (x, y)
            if child_mask[y, x]:
                continue
            left_child = True
            if parent_mask[y, x]:
                if entered is None:
                    entered = step
            elif entered is not None:
                values.append(step - entered)
                break
            elif left_child:
                break
    return min(values) if len(values) >= _RAY_COUNT // 2 else None


def _deferred_separator_ring(parent_mask: np.ndarray, child_mask: np.ndarray, fill: str) -> str | None:
    if not _ellipse_like(child_mask):
        return None
    thickness = _separator_thickness(parent_mask, child_mask)
    if thickness is None or thickness <= _SAFETY_MARGIN + 1.0:
        return None
    ys, xs = np.nonzero(child_mask)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    mid = thickness / 2.0
    stroke = max(1.0, thickness - _SAFETY_MARGIN)
    d = _core._ellipse_d(cx - rx - mid, cy - ry - mid, cx + rx + mid, cy + ry + mid)
    if d is None:
        return None
    return f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="{_fmt(stroke)}"/>'


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
        color_index = int(nodes[node_id]["color_index"])
        red, green, blue = (
            int(value) for value in np.asarray(colors, dtype=np.uint8)[color_index]
        )
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        overlay = _deferred_separator_ring(
            node_map == node_id,
            node_map == matching[0],
            fill,
        )
        if overlay is None:
            continue
        overlays.append(overlay)
        strategy_counts[strategy] -= 1
        if strategy_counts[strategy] <= 0:
            strategy_counts.pop(strategy, None)
        upgraded = f"{strategy}_deferred_source_annulus"
        strategy_counts[upgraded] += 1
        region["strategy"] = upgraded
        region["deferred_separator_ring"] = True
        region["path_count"] = int(region.get("path_count") or 0) + 1

    elements.extend(overlays)
    report["elements"] = elements
    report["path_count"] = int(report.get("path_count") or 0) + len(overlays)
    report["node_count"] = int(report.get("node_count") or 0) + 4 * len(overlays)
    report["strategy_counts"] = dict(sorted(strategy_counts.items()))
    report["regions"] = regions
    report["deferred_separator_rings"] = len(overlays)
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
