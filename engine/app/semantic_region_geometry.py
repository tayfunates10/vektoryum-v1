"""Generic connected-region semantic geometry for flat exact palettes.

The source label map is decomposed into connected regions, rooted at the corner
canvas, and painted by region depth. Primitive proposals are source-derived and
fixture-agnostic. Nested A-B-A separators receive source-mask-derived device-space
anchors only where the source pixel is already B; this prevents low-scale snap
merges without changing palette policy or raster embedding. The caller remains
authoritative: every proposal must pass native and five-scale validation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from math import cos, pi, sin
from typing import Any

import cv2
import numpy as np


class SemanticRegionFitError(RuntimeError):
    pass


_MIN_SEMANTIC_SOURCE_SIDE = 64
_SMALL_RECT_EPS = 0.01
_ELLIPSE_INSET = 0.10
_NESTED_INSET = 0.20
_RING_EXPAND = 0.25
_COMPOUND_PARENT_EXPAND = 0.25
_COMPOUND_CHILD_INSET = 0.01
_SEPARATOR_ANCHOR_DIRECTIONS = 16
_SEPARATOR_ANCHOR_DEVICE_PX = 1.0


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
            candidates.sort(key=lambda idx: preferred.index(direction_index[direction(edges[idx][0], edges[idx][1])]))
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
                cross = ((point[0] - before[0]) * (after[1] - point[1])) - ((point[1] - before[1]) * (after[0] - point[0]))
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
    epsilon = _SMALL_RECT_EPS if min(x1 - x0, y1 - y0) <= 4.0 else 0.0
    return f"M{_fmt(x0-epsilon)} {_fmt(y0-epsilon)}L{_fmt(x1+epsilon)} {_fmt(y0-epsilon)}L{_fmt(x1+epsilon)} {_fmt(y1+epsilon)}L{_fmt(x0-epsilon)} {_fmt(y1+epsilon)}Z"


def _ellipse_d(x0: float, y0: float, x1: float, y1: float) -> str | None:
    rx = ((x1 - x0) / 2.0) - _ELLIPSE_INSET
    ry = ((y1 - y0) / 2.0) - _ELLIPSE_INSET
    if rx < 1.0 or ry < 1.0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return f"M{_fmt(cx-rx)} {_fmt(cy)}A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx+rx)} {_fmt(cy)}A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx-rx)} {_fmt(cy)}Z"


def _ellipse_from_cycle(points: list[tuple[int, int]]) -> str | None:
    if len(points) < 8:
        return None
    arr = np.asarray(points, dtype=np.float64)
    low, high = arr.min(axis=0), arr.max(axis=0)
    span = high - low
    if min(span) < 2.0 or max(span) / max(1e-9, min(span)) > 1.35:
        return None
    area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
    occupancy = area / max(1.0, float(span[0] * span[1]))
    if not (0.48 <= occupancy <= 0.90):
        return None
    return _ellipse_d(low[0], low[1], high[0], high[1])


def _fit_single_outer(points: list[tuple[int, int]]) -> tuple[list[str], str] | None:
    rectangle = _rect_bounds(points)
    if rectangle is not None:
        return [_rect_d(rectangle)], "axis_aligned_rectangle"
    ellipse = _ellipse_from_cycle(points)
    if ellipse is not None:
        return [ellipse], "pixel_center_ellipse"
    if len(points) >= 12:
        arr = np.asarray(points, dtype=np.float64)
        try:
            from app.shape_fitting import try_fit_whole_shape  # noqa: PLC0415
            fitted = try_fit_whole_shape(arr, True)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted:
            return [fitted], "whole_shape"
    return None


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


def _filled_external_region(region_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(region_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled.astype(bool)


def _fit_ellipse_axis_arms(region_mask: np.ndarray) -> tuple[list[str], str] | None:
    external = _filled_external_region(region_mask)
    ys, xs = np.nonzero(external)
    if len(xs) < 16:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = external[y0:y1, x0:x1]
    height, width = crop.shape
    if min(height, width) < 8:
        return None

    arms: list[tuple[str, int, int]] = []
    for start, stop in _runs(crop.sum(axis=0) >= max(3, int(round(0.90 * height)))):
        arm_width = stop - start
        if arm_width <= max(8, int(round(0.16 * width))) and height / max(1, arm_width) >= 4.0:
            arms.append(("v", start, stop))
    for start, stop in _runs(crop.sum(axis=1) >= max(3, int(round(0.90 * width)))):
        arm_height = stop - start
        if arm_height <= max(8, int(round(0.16 * height))) and width / max(1, arm_height) >= 4.0:
            arms.append(("h", start, stop))
    if not arms or len(arms) > 6:
        return None

    residual = crop.copy()
    for axis, start, stop in arms:
        if axis == "v":
            residual[:, start:stop] = False
        else:
            residual[start:stop, :] = False
    ry, rx = np.nonzero(residual)
    if len(rx) < 12:
        return None
    ex0, ex1 = int(rx.min()), int(rx.max()) + 1
    ey0, ey1 = int(ry.min()), int(ry.max()) + 1
    ew, eh = ex1 - ex0, ey1 - ey0
    if min(ew, eh) < 6 or max(ew, eh) / max(1, min(ew, eh)) > 1.45:
        return None
    occupancy = int(residual.sum()) / max(1, ew * eh)
    if not (0.45 <= occupancy <= 0.90):
        return None
    ellipse = _ellipse_d(x0 + ex0, y0 + ey0, x0 + ex1, y0 + ey1)
    if ellipse is None:
        return None
    paths = [ellipse]
    for axis, start, stop in arms:
        if axis == "v":
            paths.append(_rect_d((x0 + start, y0, x0 + stop, y1)))
        else:
            paths.append(_rect_d((x0, y0 + start, x1, y0 + stop)))
    return paths, "ellipse_axis_arm_union"


def _connected_region_graph(labels: np.ndarray, color_count: int) -> tuple[np.ndarray, list[dict[str, Any]], int, list[int], list[int | None]]:
    h, w = labels.shape
    node_map = np.full((h, w), -1, dtype=np.int32)
    nodes: list[dict[str, Any]] = []
    for color_index in range(color_count):
        count, component_map = cv2.connectedComponents((labels == color_index).astype(np.uint8), connectivity=8)
        for local_index in range(1, int(count)):
            node_id = len(nodes)
            region = component_map == local_index
            node_map[region] = node_id
            nodes.append({"id": node_id, "color_index": int(color_index), "area": int(region.sum())})
    if np.any(node_map < 0):
        raise SemanticRegionFitError("region_graph_unassigned_pixels")

    edges: list[set[int]] = [set() for _ in nodes]
    for first_map, second_map in ((node_map[:, :-1], node_map[:, 1:]), (node_map[:-1, :], node_map[1:, :])):
        different = first_map != second_map
        for left, right in zip(first_map[different].astype(int).tolist(), second_map[different].astype(int).tolist(), strict=True):
            if left != right:
                edges[left].add(right)
                edges[right].add(left)
    corner_nodes = np.asarray([node_map[0, 0], node_map[0, -1], node_map[-1, 0], node_map[-1, -1]], dtype=np.int32)
    root = int(np.argmax(np.bincount(corner_nodes)))
    depth = [-1] * len(nodes)
    parent: list[int | None] = [None] * len(nodes)
    depth[root] = 0
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(edges[node]):
            if depth[neighbor] < 0:
                depth[neighbor] = depth[node] + 1
                parent[neighbor] = node
                queue.append(neighbor)
    if any(value < 0 for value in depth):
        raise SemanticRegionFitError("region_graph_disconnected")
    return node_map, nodes, root, depth, parent


def _bbox_transform(mask: np.ndarray, inset: float) -> str | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    width, height = x1 - x0, y1 - y0
    target_width, target_height = width - 2.0 * inset, height - 2.0 * inset
    if min(target_width, target_height) <= 0.0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return f"translate({_fmt(cx)} {_fmt(cy)}) scale({_fmt(target_width/width)} {_fmt(target_height/height)}) translate({_fmt(-cx)} {_fmt(-cy)})"


def _bbox_occupancy(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    box = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return int(mask.sum()) / max(1, box)


def _separator_anchor_points(parent_mask: np.ndarray, child_mask: np.ndarray) -> list[tuple[int, int]]:
    """Pick source B pixels along radial exits from a nested A child."""
    ys, xs = np.nonzero(child_mask)
    if not len(xs):
        return []
    cx = float(xs.mean())
    cy = float(ys.mean())
    h, w = parent_mask.shape
    max_step = int(max(h, w))
    points: set[tuple[int, int]] = set()
    for index in range(_SEPARATOR_ANCHOR_DIRECTIONS):
        angle = (2.0 * pi * index) / _SEPARATOR_ANCHOR_DIRECTIONS
        dx, dy = cos(angle), sin(angle)
        seen_child = False
        for step in np.arange(0.0, float(max_step), 0.25):
            x = int(round(cx + dx * step))
            y = int(round(cy + dy * step))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if child_mask[y, x]:
                seen_child = True
                continue
            if seen_child and parent_mask[y, x]:
                points.add((x, y))
                break
            if seen_child and not parent_mask[y, x]:
                break
    return sorted(points, key=lambda point: (point[1], point[0]))


def _separator_anchor_element(x: int, y: int, fill: str) -> str:
    return (
        f'<path d="M{_fmt(x+0.5)} {_fmt(y+0.5)}h0.001" fill="none" '
        f'stroke="{fill}" stroke-width="{_fmt(_SEPARATOR_ANCHOR_DEVICE_PX)}" '
        'vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    h, w = labels.shape
    if min(h, w) < _MIN_SEMANTIC_SOURCE_SIDE:
        raise SemanticRegionFitError(f"source_resolution_below_semantic_floor:{min(h, w)}<{_MIN_SEMANTIC_SOURCE_SIDE}")

    node_map, nodes, root, depth, parent = _connected_region_graph(labels, len(colors))
    children: list[list[int]] = [[] for _ in nodes]
    for node_id, parent_id in enumerate(parent):
        if parent_id is not None:
            children[parent_id].append(node_id)
    order = sorted(range(len(nodes)), key=lambda node_id: (depth[node_id], -nodes[node_id]["area"], node_id))
    elements: list[str] = []
    strategy_counts: dict[str, int] = defaultdict(int)
    assigned_strategy: dict[int, str] = {}
    command_count = 0
    reports: list[dict[str, Any]] = []

    for node_id in order:
        node = nodes[node_id]
        color_index = int(node["color_index"])
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        region_mask = node_map == node_id
        transform: str | None = None
        anchor_points: list[tuple[int, int]] = []

        if node_id == root:
            paths = [f"M0 0H{w}V{h}H0Z"]
            strategy = "corner_connected_canvas"
        else:
            cycles = _boundary_cycles(region_mask)
            if not cycles:
                raise SemanticRegionFitError("region_without_boundary")
            outer = max(cycles, key=lambda points: abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)))))
            fitted = _fit_single_outer(outer) or _fit_ellipse_axis_arms(region_mask)
            if fitted is None:
                arr = np.asarray(outer, dtype=np.float64)
                span = arr.max(axis=0) - arr.min(axis=0)
                area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
                ratio = area / max(1.0, float(span[0] * span[1]))
                raise SemanticRegionFitError(f"region_outer_not_semantic:n={len(outer)}:span={_fmt(span[0])}x{_fmt(span[1])}:fill={ratio:.4f}")
            paths, strategy = fitted

            if strategy == "whole_shape":
                occupancy = _bbox_occupancy(region_mask)
                if children[node_id] and occupancy <= 0.20:
                    transform = _bbox_transform(region_mask, -_RING_EXPAND)
                    if transform:
                        strategy = "ring_expand_whole_shape"
                elif depth[node_id] >= 2:
                    transform = _bbox_transform(region_mask, _NESTED_INSET)
                    if transform:
                        strategy = "nested_inset_whole_shape"
            elif strategy == "ellipse_axis_arm_union" and children[node_id]:
                transform = _bbox_transform(region_mask, -_COMPOUND_PARENT_EXPAND)
                if transform:
                    strategy = "compound_parent_expand_axis_arm_union"
            elif strategy == "pixel_center_ellipse":
                parent_id = parent[node_id]
                if parent_id is not None and assigned_strategy.get(parent_id) in {"ellipse_axis_arm_union", "compound_parent_expand_axis_arm_union"}:
                    transform = _bbox_transform(region_mask, _COMPOUND_CHILD_INSET)
                    if transform:
                        strategy = "compound_child_tiebreak_ellipse"

            grandparent_id = parent[node_id]
            if children[node_id] and grandparent_id is not None:
                matching_children = [child for child in children[node_id] if int(nodes[child]["color_index"]) == int(nodes[grandparent_id]["color_index"])]
                if len(matching_children) == 1 and strategy in {"ellipse_axis_arm_union", "compound_parent_expand_axis_arm_union"}:
                    anchor_points = _separator_anchor_points(region_mask, node_map == matching_children[0])
                    if anchor_points:
                        strategy = "compound_axis_arm_with_source_separator_anchors"

        for d in paths:
            transform_attr = f' transform="{transform}"' if transform else ""
            elements.append(f'<path fill="{fill}" d="{d}"{transform_attr}/>')
            command_count += max(2, sum(d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z")))
        for x, y in anchor_points:
            elements.append(_separator_anchor_element(x, y, fill))
            command_count += 2
        assigned_strategy[node_id] = strategy
        strategy_counts[strategy] += 1
        reports.append({"node_id": int(node_id), "color_index": color_index, "depth": int(depth[node_id]), "parent_node_id": parent[node_id], "child_count": len(children[node_id]), "area": int(node["area"]), "bbox_occupancy": round(_bbox_occupancy(region_mask), 6), "strategy": strategy, "path_count": len(paths) + len(anchor_points), "separator_anchor_count": len(anchor_points)})

    return {"elements": elements, "path_count": len(elements), "node_count": int(command_count), "region_count": len(nodes), "max_depth": max(depth) if depth else 0, "strategy_counts": dict(sorted(strategy_counts.items())), "regions": reports}


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
