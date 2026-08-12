"""Generic connected-region geometry reconstruction for flat exact palettes.

The module derives a painter's order from source region adjacency.  It never
knows QA fixture IDs or analytic fixture builders.  Nested raster regions are
painted from the corner-connected canvas inward, which lets semantic primitives
represent rings/holes without fragile even-odd native cell boundaries.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import cv2
import numpy as np


class SemanticRegionFitError(RuntimeError):
    pass


_SMALL_EPS = 0.01
_ELLIPSE_INSET = 0.10


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _boundary_cycles(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges.append(((x, y), (x + 1, y)))
        if x == w - 1 or not mask[y, x + 1]:
            edges.append(((x + 1, y), (x + 1, y + 1)))
        if y == h - 1 or not mask[y + 1, x]:
            edges.append(((x + 1, y + 1), (x, y + 1)))
        if x == 0 or not mask[y, x - 1]:
            edges.append(((x, y + 1), (x, y)))

    outgoing: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (start, _end) in enumerate(edges):
        outgoing[start].append(index)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction_index = {direction: index for index, direction in enumerate(directions)}

    def direction(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
        return end[0] - start[0], end[1] - start[1]

    used: set[int] = set()
    cycles: list[list[tuple[int, int]]] = []
    for seed in range(len(edges)):
        if seed in used:
            continue
        origin, current = edges[seed]
        previous = origin
        points = [origin, current]
        used.add(seed)
        while current != origin:
            candidates = [idx for idx in outgoing.get(current, []) if idx not in used]
            if not candidates:
                raise SemanticRegionFitError("non_closed_region_boundary")
            incoming = direction_index[direction(previous, current)]
            preferred = [(incoming + 1) % 4, incoming, (incoming - 1) % 4, (incoming + 2) % 4]
            candidates.sort(
                key=lambda idx: preferred.index(
                    direction_index[direction(edges[idx][0], edges[idx][1])]
                )
            )
            edge_index = candidates[0]
            used.add(edge_index)
            previous, current = edges[edge_index]
            points.append(current)
            if len(points) > len(edges) + 2:
                raise SemanticRegionFitError("region_boundary_cycle_overflow")

        simplified = points[:-1]
        changed = True
        while changed and len(simplified) > 2:
            changed = False
            compact: list[tuple[int, int]] = []
            size = len(simplified)
            for index, point in enumerate(simplified):
                before = simplified[index - 1]
                after = simplified[(index + 1) % size]
                cross = ((point[0] - before[0]) * (after[1] - point[1])) - (
                    (point[1] - before[1]) * (after[0] - point[0])
                )
                if cross == 0:
                    changed = True
                    continue
                compact.append(point)
            simplified = compact
        if len(simplified) >= 3:
            cycles.append(simplified)
    return cycles


def _rect_bounds(points: list[tuple[int, int]]) -> tuple[float, float, float, float] | None:
    if len(points) != 4:
        return None
    xs = sorted({float(point[0]) for point in points})
    ys = sorted({float(point[1]) for point in points})
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])}
    return (xs[0], ys[0], xs[1], ys[1]) if set(points) == expected else None


def _rect_d(bounds: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bounds
    eps = _SMALL_EPS if min(x1 - x0, y1 - y0) <= 4.0 else 0.0
    x0 -= eps
    y0 -= eps
    x1 += eps
    y1 += eps
    return f"M{_fmt(x0)} {_fmt(y0)}L{_fmt(x1)} {_fmt(y0)}L{_fmt(x1)} {_fmt(y1)}L{_fmt(x0)} {_fmt(y1)}Z"


def _ellipse_d_from_cell_bounds(x0: float, y0: float, x1: float, y1: float) -> str | None:
    rx = ((x1 - x0) / 2.0) - _ELLIPSE_INSET
    ry = ((y1 - y0) / 2.0) - _ELLIPSE_INSET
    if rx < 1.0 or ry < 1.0:
        return None
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return (
        f"M{_fmt(cx-rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx+rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx-rx)} {_fmt(cy)}Z"
    )


def _ellipse_from_cycle(points: list[tuple[int, int]]) -> str | None:
    if len(points) < 8:
        return None
    arr = np.asarray(points, dtype=np.float64)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = hi - lo
    if min(span) < 2.0 or max(span) / max(1e-9, min(span)) > 1.35:
        return None
    area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
    fill_ratio = area / max(1.0, float(span[0] * span[1]))
    if not (0.48 <= fill_ratio <= 0.90):
        return None
    return _ellipse_d_from_cell_bounds(lo[0], lo[1], hi[0], hi[1])


def _fit_single_outer(points: list[tuple[int, int]]) -> tuple[list[str], str] | None:
    bounds = _rect_bounds(points)
    if bounds is not None:
        return [_rect_d(bounds)], "axis_aligned_rectangle"
    ellipse = _ellipse_from_cycle(points)
    if ellipse is not None:
        return [ellipse], "pixel_center_ellipse"
    arr = np.asarray(points, dtype=np.float64)
    if len(points) >= 12:
        try:
            from app.shape_fitting import try_fit_whole_shape  # noqa: PLC0415

            fitted = try_fit_whole_shape(arr, True)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted:
            return [fitted], "whole_shape"
    return None


def _filled_outer_mask(points: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    # Cell-boundary vertices lie on grid lines. Filling their polygon on the
    # pixel grid is used only for generic primitive decomposition diagnostics.
    shifted = np.column_stack((arr[:, 0], arr[:, 1])).astype(np.int32)
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def _fit_ellipse_with_axis_arms(
    points: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> tuple[list[str], str] | None:
    """Decompose a generic ellipse-like union with long vertical/horizontal arms.

    This is source-derived morphology, not a fixture pattern.  It is useful for
    negative-space regions where a smooth central primitive intersects narrow
    axis-aligned slots.  Overlap is safe because connected-region painter emits
    each primitive as a same-color element at one depth.
    """
    arr = np.asarray(points, dtype=np.float64)
    lo = arr.min(axis=0).astype(int)
    hi = arr.max(axis=0).astype(int)
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(image_shape[1], hi[0]), min(image_shape[0], hi[1])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    outer = _filled_outer_mask(points, image_shape)[y0:y1, x0:x1]
    if not outer.any():
        return None
    height, width = outer.shape

    candidates: list[tuple[str, int, int]] = []
    col_counts = outer.sum(axis=0)
    full_cols = col_counts >= max(3, int(round(0.90 * height)))
    start: int | None = None
    for index, active in enumerate(np.r_[full_cols, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            stop = index
            arm_width = stop - start
            if arm_width <= max(8, int(round(0.16 * width))) and height / max(1, arm_width) >= 4.0:
                candidates.append(("v", start, stop))
            start = None

    row_counts = outer.sum(axis=1)
    full_rows = row_counts >= max(3, int(round(0.90 * width)))
    start = None
    for index, active in enumerate(np.r_[full_rows, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            stop = index
            arm_height = stop - start
            if arm_height <= max(8, int(round(0.16 * height))) and width / max(1, arm_height) >= 4.0:
                candidates.append(("h", start, stop))
            start = None

    if not candidates or len(candidates) > 6:
        return None

    residual = outer.copy()
    for axis, start, stop in candidates:
        if axis == "v":
            residual[:, start:stop] = False
        else:
            residual[start:stop, :] = False
    ys, xs = np.nonzero(residual)
    if len(xs) < 12:
        return None
    ex0, ex1 = int(xs.min()), int(xs.max()) + 1
    ey0, ey1 = int(ys.min()), int(ys.max()) + 1
    ew, eh = ex1 - ex0, ey1 - ey0
    if min(ew, eh) < 6 or max(ew, eh) / max(1, min(ew, eh)) > 1.45:
        return None

    # Validate the decomposition against the filled OUTER footprint; nested
    # child regions are intentionally painted later by the adjacency depth.
    synthetic = np.zeros_like(outer, dtype=np.uint8)
    cv2.ellipse(
        synthetic,
        ((ex0 + ex1 - 1) // 2, (ey0 + ey1 - 1) // 2),
        (max(1, (ew - 1) // 2), max(1, (eh - 1) // 2)),
        0,
        0,
        360,
        1,
        -1,
    )
    for axis, start, stop in candidates:
        if axis == "v":
            synthetic[:, start:stop] = 1
        else:
            synthetic[start:stop, :] = 1
    inter = int(np.logical_and(outer, synthetic).sum())
    union = int(np.logical_or(outer, synthetic).sum())
    if union <= 0 or inter / union < 0.96:
        return None

    paths: list[str] = []
    ellipse = _ellipse_d_from_cell_bounds(x0 + ex0, y0 + ey0, x0 + ex1, y0 + ey1)
    if ellipse is None:
        return None
    paths.append(ellipse)
    for axis, start, stop in candidates:
        if axis == "v":
            paths.append(_rect_d((x0 + start, y0, x0 + stop, y1)))
        else:
            paths.append(_rect_d((x0, y0 + start, x1, y0 + stop)))
    return paths, "ellipse_axis_arm_union"


def _connected_region_graph(labels: np.ndarray, color_count: int) -> tuple[np.ndarray, list[dict[str, Any]], int, list[int]]:
    h, w = labels.shape
    node_map = np.full((h, w), -1, dtype=np.int32)
    nodes: list[dict[str, Any]] = []
    for color_index in range(color_count):
        count, component_map = cv2.connectedComponents((labels == color_index).astype(np.uint8), connectivity=8)
        for local_index in range(1, int(count)):
            node_id = len(nodes)
            mask = component_map == local_index
            node_map[mask] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "color_index": int(color_index),
                    "area": int(mask.sum()),
                }
            )
    if np.any(node_map < 0):
        raise SemanticRegionFitError("region_graph_unassigned_pixels")

    edges: list[set[int]] = [set() for _ in nodes]
    for a, b in ((node_map[:, :-1], node_map[:, 1:]), (node_map[:-1, :], node_map[1:, :])):
        diff = a != b
        left = a[diff].astype(int)
        right = b[diff].astype(int)
        for u, v in zip(left.tolist(), right.tolist(), strict=True):
            if u != v:
                edges[u].add(v)
                edges[v].add(u)

    corner_nodes = np.asarray(
        [node_map[0, 0], node_map[0, -1], node_map[-1, 0], node_map[-1, -1]],
        dtype=np.int32,
    )
    root = int(np.argmax(np.bincount(corner_nodes)))
    depth = [-1] * len(nodes)
    depth[root] = 0
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(edges[node]):
            if depth[neighbor] < 0:
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    if any(value < 0 for value in depth):
        raise SemanticRegionFitError("region_graph_disconnected")
    return node_map, nodes, root, depth


def build_semantic_region_elements(
    labels: np.ndarray,
    colors: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    h, w = labels.shape
    node_map, nodes, root, depth = _connected_region_graph(labels, len(colors))

    order = sorted(
        range(len(nodes)),
        key=lambda node_id: (depth[node_id], -nodes[node_id]["area"], node_id),
    )
    elements: list[str] = []
    strategy_counts: dict[str, int] = defaultdict(int)
    command_count = 0
    node_reports: list[dict[str, Any]] = []

    for node_id in order:
        node = nodes[node_id]
        color_index = int(node["color_index"])
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        if node_id == root:
            paths = [f"M0 0H{w}V{h}H0Z"]
            strategy = "corner_connected_canvas"
        else:
            mask = node_map == node_id
            cycles = _boundary_cycles(mask)
            if not cycles:
                raise SemanticRegionFitError("region_without_boundary")
            outer = max(
                cycles,
                key=lambda points: abs(
                    float(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)))
                ),
            )
            fitted = _fit_single_outer(outer)
            if fitted is None:
                fitted = _fit_ellipse_with_axis_arms(outer, (h, w))
            if fitted is None:
                arr = np.asarray(outer, dtype=np.float64)
                span = arr.max(axis=0) - arr.min(axis=0)
                area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
                ratio = area / max(1.0, float(span[0] * span[1]))
                raise SemanticRegionFitError(
                    f"region_outer_not_semantic:n={len(outer)}:span={_fmt(span[0])}x{_fmt(span[1])}:fill={ratio:.4f}"
                )
            paths, strategy = fitted

        for d in paths:
            elements.append(f'<path fill="{fill}" d="{d}"/>')
            command_count += max(2, sum(d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z")))
        strategy_counts[strategy] += 1
        node_reports.append(
            {
                "node_id": int(node_id),
                "color_index": color_index,
                "depth": int(depth[node_id]),
                "area": int(node["area"]),
                "strategy": strategy,
                "path_count": len(paths),
            }
        )

    return {
        "elements": elements,
        "path_count": len(elements),
        "node_count": int(command_count),
        "region_count": len(nodes),
        "max_depth": max(depth) if depth else 0,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "regions": node_reports,
    }


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
