"""Render-feedback scale-topology repair around the native-good semantic core.

The native semantic fitter is preserved byte-for-byte in
``semantic_region_geometry_core``. This fixture-agnostic wrapper renders the
source-derived SVG at the five acceptance scales. Native connected regions are
projected independently so downsampling can never erase their identity. Repair
is deliberately limited to a source semantic separator that the native fitter
classified as an ellipse/axis-arm compound; ordinary root-canvas merges and
independent small details are not modified. If two source regions of one palette
class collapse through such a separator, the wrapper finds the actual rendered
bridge and repaints the bridge point nearest the intervening source region using
that region's palette class. Every repair is re-rendered immediately; unchanged
native and five-scale transactions remain authoritative. No raster embedding,
fixture routing, threshold relaxation, or shared budget increase is used.
"""
from __future__ import annotations

import os
import tempfile
from collections import deque
from typing import Any

import cv2
import numpy as np

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_REPAIR_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
_MAX_REPAIR_ANCHORS_PER_SCALE = 12
_NEIGHBORS = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))
_ELIGIBLE_SEPARATOR_STRATEGIES = {
    "ellipse_axis_arm_union",
    "compound_parent_expand_axis_arm_union",
}


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" shape-rendering="crispEdges">'
        + "".join(elements)
        + "</svg>"
    )


def _nearest_palette_labels(rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int32)
    palette = np.asarray(colors, dtype=np.int32)
    diff = arr[:, :, None, :] - palette[None, None, :, :]
    return np.argmin(np.sum(diff * diff, axis=3, dtype=np.int64), axis=2).astype(np.int16)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    return max(0, int(count) - 1)


def _component_counts(labels: np.ndarray, color_count: int) -> list[int]:
    return [_component_count(labels == index) for index in range(color_count)]


def _render_labels(width: int, height: int, elements: list[str], colors: np.ndarray, target_width: int, target_height: int) -> np.ndarray | None:
    try:
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception:
        return None
    handle = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    path = handle.name
    try:
        handle.write(_svg_document(width, height, elements).encode("utf-8")); handle.close()
        rendered = render_svg_to_rgb(path, int(target_width), int(target_height))
        return None if rendered is None else _nearest_palette_labels(rendered, colors)
    finally:
        try: handle.close()
        except Exception: pass
        try: os.unlink(path)
        except OSError: pass


def _project_node_mask(node_map: np.ndarray, node_id: int, width: int, height: int) -> np.ndarray:
    native = (node_map == int(node_id)).astype(np.uint8)
    projected = cv2.resize(native, (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(bool)
    if projected.any():
        return projected
    ys, xs = np.nonzero(native)
    if len(xs):
        px = min(width - 1, max(0, int(round((float(xs.mean()) + 0.5) * width / node_map.shape[1] - 0.5))))
        py = min(height - 1, max(0, int(round((float(ys.mean()) + 0.5) * height / node_map.shape[0] - 0.5))))
        projected[py, px] = True
    return projected


def _tree_path(first: int, second: int, parent: list[int | None]) -> list[int]:
    first_chain: list[int] = []
    cursor: int | None = int(first)
    while cursor is not None:
        first_chain.append(cursor); cursor = parent[cursor]
    first_pos = {node_id: index for index, node_id in enumerate(first_chain)}
    second_chain: list[int] = []
    cursor = int(second)
    lca: int | None = None
    while cursor is not None:
        if cursor in first_pos:
            lca = cursor; break
        second_chain.append(cursor); cursor = parent[cursor]
    if lca is None:
        return []
    return first_chain[: first_pos[lca] + 1] + list(reversed(second_chain))


def _shortest_render_path(component: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[tuple[int,int]]:
    h, w = component.shape
    starts = np.argwhere(component & start)
    if not len(starts) or not (component & goal).any():
        return []
    queue: deque[tuple[int,int]] = deque()
    previous: dict[tuple[int,int], tuple[int,int] | None] = {}
    for y, x in starts.tolist():
        key = (int(y), int(x)); queue.append(key); previous[key] = None
    endpoint: tuple[int,int] | None = None
    while queue and endpoint is None:
        y, x = queue.popleft()
        if goal[y, x]: endpoint = (y, x); break
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy; key = (ny, nx)
            if nx < 0 or nx >= w or ny < 0 or ny >= h or key in previous or not component[ny, nx]:
                continue
            previous[key] = (y, x); queue.append(key)
    if endpoint is None:
        return []
    path: list[tuple[int,int]] = []
    cursor: tuple[int,int] | None = endpoint
    while cursor is not None:
        path.append(cursor); cursor = previous[cursor]
    path.reverse()
    return path


def _bridge_path_repair(
    render_labels: np.ndarray,
    class_index: int,
    node_map: np.ndarray,
    nodes: list[dict[str, Any]],
    parent: list[int | None],
    strategy_by_node: dict[int, str],
) -> tuple[int,int,int,dict[str,Any]] | None:
    """Find a rendered bridge only across an eligible semantic separator."""
    class_index = int(class_index)
    render_mask = render_labels == class_index
    ren_count, ren_cc = cv2.connectedComponents(render_mask.astype(np.uint8), connectivity=8)
    target_h, target_w = render_mask.shape
    class_nodes = [index for index, node in enumerate(nodes) if int(node["color_index"]) == class_index]
    projected = {node_id: _project_node_mask(node_map, node_id, target_w, target_h) for node_id in class_nodes}

    for render_id in range(1, ren_count):
        component = ren_cc == render_id
        overlapping = [node_id for node_id in class_nodes if (component & projected[node_id]).any()]
        if len(overlapping) < 2:
            continue
        for pair_index, first in enumerate(overlapping[:-1]):
            for second in overlapping[pair_index + 1:]:
                source_path = _tree_path(first, second, parent)
                separators = [
                    node_id
                    for node_id in source_path[1:-1]
                    if int(nodes[node_id]["color_index"]) != class_index
                    and strategy_by_node.get(int(node_id)) in _ELIGIBLE_SEPARATOR_STRATEGIES
                ]
                if not separators:
                    continue
                path = _shortest_render_path(component, projected[first], projected[second])
                if not path:
                    continue
                best: tuple[float,int,int,int,int] | None = None
                for separator_node in separators:
                    separator_mask = _project_node_mask(node_map, separator_node, target_w, target_h)
                    distance = cv2.distanceTransform((~separator_mask).astype(np.uint8), cv2.DIST_L2, 3)
                    replacement = int(nodes[separator_node]["color_index"])
                    for y, x in path:
                        candidate = (float(distance[y, x]), int(y), int(x), replacement, int(separator_node))
                        if best is None or candidate < best:
                            best = candidate
                if best is None:
                    continue
                score, y, x, replacement, separator_node = best
                return int(x), int(y), int(replacement), {
                    "source_node_pair": [int(first), int(second)],
                    "source_tree_path": [int(v) for v in source_path],
                    "separator_node": int(separator_node),
                    "separator_strategy": strategy_by_node.get(int(separator_node)),
                    "bridge_path_length": len(path),
                    "distance_to_separator": round(float(score), 4),
                }
    return None


def _device_anchor(x: int, y: int, target_width: int, target_height: int, source_width: int, source_height: int, fill: str) -> str:
    cx = (float(x) + 0.5) * float(source_width) / float(target_width)
    cy = (float(y) + 0.5) * float(source_height) / float(target_height)
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def _repair_component_merges(labels: np.ndarray, colors: np.ndarray, elements: list[str], strategy_by_node: dict[int, str]) -> tuple[list[str], list[dict[str, Any]]]:
    source_height, source_width = labels.shape
    node_map, nodes, _root, _depth, parent = _core._connected_region_graph(labels, len(colors))
    source_counts = [sum(1 for node in nodes if int(node["color_index"]) == index) for index in range(len(colors))]
    repaired = list(elements); reports: list[dict[str, Any]] = []
    for factor in _REPAIR_FACTORS:
        target_width = max(16, int(round(source_width * factor))); target_height = max(16, int(round(source_height * factor)))
        anchors: list[dict[str, Any]] = []
        for _attempt in range(_MAX_REPAIR_ANCHORS_PER_SCALE):
            rendered = _render_labels(source_width, source_height, repaired, colors, target_width, target_height)
            if rendered is None: break
            counts = _component_counts(rendered, len(colors))
            merged_classes = [index for index,(expected,actual) in enumerate(zip(source_counts,counts,strict=True)) if actual < expected]
            if not merged_classes: break
            applied = False
            for class_index in merged_classes:
                repair = _bridge_path_repair(rendered, class_index, node_map, nodes, parent, strategy_by_node)
                if repair is None: continue
                x,y,replacement,detail = repair
                red,green,blue = (int(value) for value in np.asarray(colors,dtype=np.uint8)[replacement]); fill=f"#{red:02x}{green:02x}{blue:02x}"
                repaired.append(_device_anchor(x,y,target_width,target_height,source_width,source_height,fill))
                anchors.append({"class_index":int(class_index),"replacement_class_index":int(replacement),"target_pixel":[int(x),int(y)],**detail})
                applied=True; break
            if not applied: break
        final_labels = _render_labels(source_width,source_height,repaired,colors,target_width,target_height)
        reports.append({"scale":f"{factor:g}x","size":[target_width,target_height],"anchors":anchors,"final_component_counts":_component_counts(final_labels,len(colors)) if final_labels is not None else None})
    return repaired,reports


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    report = _core.build_semantic_region_elements(labels, colors)
    base_elements = list(report.get("elements") or [])
    strategy_by_node = {
        int(region["node_id"]): str(region.get("strategy") or "")
        for region in (report.get("regions") or [])
        if "node_id" in region
    }
    repaired, repair_report = _repair_component_merges(
        np.asarray(labels), np.asarray(colors,dtype=np.uint8), base_elements, strategy_by_node
    )
    added = len(repaired)-len(base_elements)
    if added:
        strategies = dict(report.get("strategy_counts") or {}); strategies["render_feedback_bridge_cut_anchor"] = added
        report["strategy_counts"] = dict(sorted(strategies.items())); report["path_count"] = int(report.get("path_count") or 0)+added; report["node_count"] = int(report.get("node_count") or 0)+2*added
    report["elements"] = repaired; report["scale_topology_repair"] = repair_report; report["scale_topology_anchor_count"] = added
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
