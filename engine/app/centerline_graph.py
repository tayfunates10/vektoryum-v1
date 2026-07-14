"""Deterministic skeleton-to-centerline graph core."""
from __future__ import annotations

from math import hypot, isfinite
from typing import Any

import cv2
import numpy as np

Pixel = tuple[int, int]
Point = tuple[float, float]
_OFFSETS = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if dx or dy
)


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    if not isinstance(binary, np.ndarray) or binary.ndim != 2:
        raise ValueError("binary image must be a 2-D numpy array")
    work = np.where(binary > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(work)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(1, work.shape[0] + work.shape[1] + 4)):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, kernel)
        if cv2.countNonZero(work) == 0:
            return skeleton
    raise RuntimeError("skeletonization iteration budget exceeded")


def _raw_graph(mask: np.ndarray) -> dict[Pixel, set[Pixel]]:
    ys, xs = np.nonzero(mask > 0)
    pixels = {(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())}
    graph = {pixel: set() for pixel in pixels}
    for x, y in sorted(pixels, key=lambda point: (point[1], point[0])):
        for dx, dy in _OFFSETS:
            other = (x + dx, y + dy)
            if other not in pixels:
                continue
            if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                continue
            graph[(x, y)].add(other)
    return graph


def _collapse_junctions(
    raw: dict[Pixel, set[Pixel]],
) -> tuple[dict[int, set[int]], dict[int, Point]]:
    junctions = {pixel for pixel, neighbors in raw.items() if len(neighbors) >= 3}
    remaining = set(junctions)
    groups: list[list[Pixel]] = []
    while remaining:
        seed = min(remaining, key=lambda point: (point[1], point[0]))
        remaining.remove(seed)
        stack = [seed]
        group: list[Pixel] = []
        while stack:
            pixel = stack.pop()
            group.append(pixel)
            for other in raw[pixel]:
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        groups.append(sorted(group, key=lambda point: (point[1], point[0])))
    groups.sort(key=lambda group: (group[0][1], group[0][0]))

    mapping: dict[Pixel, int] = {}
    points: dict[int, Point] = {}
    next_id = 0
    for group in groups:
        for pixel in group:
            mapping[pixel] = next_id
        points[next_id] = (
            sum(pixel[0] for pixel in group) / len(group),
            sum(pixel[1] for pixel in group) / len(group),
        )
        next_id += 1
    for pixel in sorted(set(raw) - junctions, key=lambda point: (point[1], point[0])):
        mapping[pixel] = next_id
        points[next_id] = (float(pixel[0]), float(pixel[1]))
        next_id += 1

    graph = {node: set() for node in points}
    for pixel, neighbors in raw.items():
        left = mapping[pixel]
        for other in neighbors:
            right = mapping[other]
            if left != right:
                graph[left].add(right)
                graph[right].add(left)
    return graph, points


def _remove(node: int, graph: dict[int, set[int]], points: dict[int, Point]) -> None:
    for other in tuple(graph.get(node, ())):
        graph[other].discard(node)
    graph.pop(node, None)
    points.pop(node, None)


def _prune(
    graph: dict[int, set[int]],
    points: dict[int, Point],
    minimum: float,
) -> tuple[int, int]:
    branches = nodes = 0
    if minimum <= 0:
        return branches, nodes
    while True:
        changed = False
        endpoints = sorted(
            (node for node, neighbors in graph.items() if len(neighbors) == 1),
            key=lambda node: (points[node][1], points[node][0], node),
        )
        for endpoint in endpoints:
            if endpoint not in graph or len(graph[endpoint]) != 1:
                continue
            chain = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while True:
                choices = graph[current] - ({previous} if previous is not None else set())
                if len(choices) != 1:
                    break
                nxt = next(iter(choices))
                length += hypot(
                    points[nxt][0] - points[current][0],
                    points[nxt][1] - points[current][1],
                )
                chain.append(nxt)
                previous, current = current, nxt
                if len(graph[current]) != 2:
                    break
            if len(graph.get(current, ())) >= 3 and length < minimum:
                for node in chain[:-1]:
                    if node in graph:
                        _remove(node, graph, points)
                        nodes += 1
                branches += 1
                changed = True
                break
        if not changed:
            break
    for node in tuple(graph):
        if not graph[node]:
            _remove(node, graph, points)
    return branches, nodes


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _trace(
    graph: dict[int, set[int]],
    points: dict[int, Point],
) -> tuple[list[dict[str, Any]], int]:
    visited: set[tuple[int, int]] = set()
    paths: list[dict[str, Any]] = []
    order = lambda node: (points[node][1], points[node][0], node)
    anchors = sorted((node for node in graph if len(graph[node]) != 2), key=order)
    for anchor in anchors:
        for neighbor in sorted(graph[anchor], key=order):
            if _edge(anchor, neighbor) in visited:
                continue
            nodes = [anchor, neighbor]
            visited.add(_edge(anchor, neighbor))
            previous, current = anchor, neighbor
            while len(graph[current]) == 2:
                nxt = next(iter(graph[current] - {previous}))
                if _edge(current, nxt) in visited:
                    break
                visited.add(_edge(current, nxt))
                nodes.append(nxt)
                previous, current = current, nxt
            paths.append({"points": tuple(points[node] for node in nodes), "closed": False})

    edges = {_edge(node, other) for node in graph for other in graph[node] if node < other}
    while edges - visited:
        first = min(edges - visited)
        start = min(first, key=order)
        choices = sorted((n for n in graph[start] if _edge(start, n) not in visited), key=order)
        if not choices:
            break
        previous, current = start, choices[0]
        nodes = [start, current]
        visited.add(_edge(start, current))
        for _ in range(len(edges) + 1):
            if current == start:
                break
            choices = sorted((n for n in graph[current] if n != previous), key=order)
            fresh = [n for n in choices if _edge(current, n) not in visited]
            if not choices:
                break
            nxt = fresh[0] if fresh else choices[0]
            if _edge(current, nxt) in visited and nxt != start:
                break
            visited.add(_edge(current, nxt))
            nodes.append(nxt)
            previous, current = current, nxt
        paths.append({
            "points": tuple(points[node] for node in nodes),
            "closed": len(nodes) >= 4 and nodes[-1] == start,
        })
    return paths, len(visited)


def _simplify(points: tuple[Point, ...], closed: bool) -> tuple[Point, ...]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    for index in range(1, len(points) - 1):
        ax, ay = result[-1]
        bx, by = points[index]
        cx, cy = points[index + 1]
        if abs((bx - ax) * (cy - by) - (by - ay) * (cx - bx)) > 1e-9:
            result.append((bx, by))
    result.append(points[-1])
    if closed and result[0] != result[-1]:
        result.append(result[0])
    return tuple(result)


def trace_skeleton_graph(mask: np.ndarray, min_branch: float = 6.0) -> dict[str, Any]:
    if isinstance(min_branch, bool) or not isinstance(min_branch, (int, float)) or min_branch < 0:
        raise ValueError("min_branch must be a non-negative number")
    raw = _raw_graph(mask)
    if not raw:
        return {
            "paths": (),
            "report": {
                "schema_version": "centerline-graph-v1",
                "backend": "opencv_skeleton_graph",
                "measurement_available": True,
                "valid": False,
                "confidence": 0.0,
                "errors": ["empty_skeleton"],
                "topology": {},
            },
        }
    graph, points = _collapse_junctions(raw)
    pruned_branches, pruned_nodes = _prune(graph, points, float(min_branch))
    paths, serialized_edges = _trace(graph, points)
    paths = [
        {"points": _simplify(path["points"], path["closed"]), "closed": path["closed"]}
        for path in paths
    ]
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    isolated = sum(1 for neighbors in graph.values() if not neighbors)
    finite = all(isfinite(value) for point in points.values() for value in point)
    valid = bool(paths) and edge_count > 0 and serialized_edges == edge_count and not isolated and finite
    confidence = round(
        max(0.0, min(1.0, (serialized_edges / max(edge_count, 1)) * (1 - 0.5 * pruned_nodes / max(len(raw), 1)))),
        6,
    )
    topology = {
        "node_count": len(graph),
        "edge_count": edge_count,
        "path_count": len(paths),
        "endpoint_count": sum(1 for neighbors in graph.values() if len(neighbors) == 1),
        "junction_count": sum(1 for neighbors in graph.values() if len(neighbors) >= 3),
        "loop_count": sum(1 for path in paths if path["closed"]),
        "isolated_node_count": isolated,
        "pruned_spur_count": pruned_branches,
        "pruned_node_count": pruned_nodes,
        "source_skeleton_pixels": len(raw),
        "serialized_edge_count": serialized_edges,
        "min_branch": float(min_branch),
    }
    errors = []
    if not paths:
        errors.append("no_paths")
    if serialized_edges != edge_count:
        errors.append("edge_coverage_mismatch")
    if isolated:
        errors.append("isolated_nodes")
    if not finite:
        errors.append("non_finite_coordinates")
    return {
        "paths": tuple(paths),
        "report": {
            "schema_version": "centerline-graph-v1",
            "backend": "opencv_skeleton_graph",
            "measurement_available": True,
            "valid": valid,
            "confidence": confidence,
            "errors": errors,
            "topology": topology,
        },
    }


def validate_centerline_report(report: Any) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return False, ("centerline_report_missing",)
    if report.get("schema_version") != "centerline-graph-v1":
        errors.append("centerline_schema_mismatch")
    if report.get("backend") != "opencv_skeleton_graph":
        errors.append("centerline_backend_mismatch")
    if report.get("measurement_available") is not True:
        errors.append("centerline_measurement_unavailable")
    confidence = report.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        errors.append("centerline_confidence_invalid")
    topology = report.get("topology")
    if not isinstance(topology, dict):
        errors.append("centerline_topology_missing")
    else:
        for key in ("edge_count", "path_count", "isolated_node_count", "serialized_edge_count"):
            value = topology.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"centerline_{key}_invalid")
        if topology.get("edge_count", 0) <= 0 or topology.get("path_count", 0) <= 0:
            errors.append("centerline_graph_empty")
        if topology.get("isolated_node_count") != 0:
            errors.append("centerline_isolated_nodes")
        if topology.get("edge_count") != topology.get("serialized_edge_count"):
            errors.append("centerline_edge_coverage_mismatch")
    if report.get("valid") is not True:
        errors.append("centerline_graph_invalid")
    return not errors, tuple(dict.fromkeys(errors))
