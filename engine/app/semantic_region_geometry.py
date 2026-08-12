"""Render-feedback scale-topology repair around the native-good semantic core.

The native semantic fitter is preserved byte-for-byte in
``semantic_region_geometry_core``. This fixture-agnostic wrapper renders the
source-derived SVG at the five acceptance scales. When source components of one
palette class collapse into one rendered component, it finds an actual bridge
path between the projected source components and repaints only a bridge pixel
that belongs to another source palette class. Each repair is re-rendered; the
unchanged native and five-scale transactions remain authoritative. No raster
embedding, fixture routing, threshold relaxation, or shared budget increase is
used.
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


def _source_target_labels(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(np.asarray(labels, dtype=np.int16), (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(np.int16)


def _bridge_path_repair(render_labels: np.ndarray, reference_labels: np.ndarray, class_index: int) -> tuple[int,int,int] | None:
    """Return a source-mismatching pixel on a real path merging reference CCs."""
    class_index = int(class_index)
    render_mask = render_labels == class_index
    ref_mask = reference_labels == class_index
    ref_count, ref_cc = cv2.connectedComponents(ref_mask.astype(np.uint8), connectivity=8)
    ren_count, ren_cc = cv2.connectedComponents(render_mask.astype(np.uint8), connectivity=8)
    if ref_count <= ren_count:
        return None
    h, w = render_mask.shape
    for render_id in range(1, ren_count):
        component = ren_cc == render_id
        overlapping = sorted(int(v) for v in np.unique(ref_cc[component & (ref_cc > 0)]))
        if len(overlapping) < 2:
            continue
        start_id, goal_id = overlapping[0], overlapping[1]
        start_points = np.argwhere(component & (ref_cc == start_id))
        goal = component & (ref_cc == goal_id)
        if not len(start_points) or not goal.any():
            continue
        queue: deque[tuple[int,int]] = deque()
        previous: dict[tuple[int,int], tuple[int,int] | None] = {}
        for y, x in start_points.tolist():
            key = (int(y), int(x)); queue.append(key); previous[key] = None
        endpoint: tuple[int,int] | None = None
        while queue and endpoint is None:
            y, x = queue.popleft()
            if goal[y, x]:
                endpoint = (y, x); break
            for dx, dy in _NEIGHBORS:
                nx, ny = x + dx, y + dy
                key = (ny, nx)
                if nx < 0 or nx >= w or ny < 0 or ny >= h or key in previous or not component[ny, nx]:
                    continue
                previous[key] = (y, x); queue.append(key)
        if endpoint is None:
            continue
        path: list[tuple[int,int]] = []
        cursor: tuple[int,int] | None = endpoint
        while cursor is not None:
            path.append(cursor); cursor = previous[cursor]
        bridge = [(y,x) for y,x in reversed(path) if int(reference_labels[y,x]) != class_index]
        if not bridge:
            continue
        y, x = bridge[len(bridge)//2]
        return int(x), int(y), int(reference_labels[y,x])
    return None


def _device_anchor(x: int, y: int, target_width: int, target_height: int, source_width: int, source_height: int, fill: str) -> str:
    cx = (float(x) + 0.5) * float(source_width) / float(target_width)
    cy = (float(y) + 0.5) * float(source_height) / float(target_height)
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def _repair_component_merges(labels: np.ndarray, colors: np.ndarray, elements: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    source_height, source_width = labels.shape
    source_counts = _component_counts(labels, len(colors))
    repaired = list(elements); reports: list[dict[str, Any]] = []
    for factor in _REPAIR_FACTORS:
        target_width = max(16, int(round(source_width * factor))); target_height = max(16, int(round(source_height * factor)))
        reference = _source_target_labels(labels, target_width, target_height); anchors: list[dict[str, Any]] = []
        for _attempt in range(_MAX_REPAIR_ANCHORS_PER_SCALE):
            rendered = _render_labels(source_width, source_height, repaired, colors, target_width, target_height)
            if rendered is None: break
            counts = _component_counts(rendered, len(colors))
            merged_classes = [index for index,(expected,actual) in enumerate(zip(source_counts,counts,strict=True)) if actual < expected]
            if not merged_classes: break
            applied = False
            for class_index in merged_classes:
                repair = _bridge_path_repair(rendered, reference, class_index)
                if repair is None: continue
                x,y,replacement = repair
                red,green,blue = (int(value) for value in np.asarray(colors,dtype=np.uint8)[replacement]); fill=f"#{red:02x}{green:02x}{blue:02x}"
                repaired.append(_device_anchor(x,y,target_width,target_height,source_width,source_height,fill))
                anchors.append({"class_index":int(class_index),"replacement_class_index":int(replacement),"target_pixel":[int(x),int(y)]})
                applied=True; break
            if not applied: break
        final_labels = _render_labels(source_width,source_height,repaired,colors,target_width,target_height)
        reports.append({"scale":f"{factor:g}x","size":[target_width,target_height],"anchors":anchors,"final_component_counts":_component_counts(final_labels,len(colors)) if final_labels is not None else None})
    return repaired,reports


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    report = _core.build_semantic_region_elements(labels, colors)
    base_elements = list(report.get("elements") or [])
    repaired, repair_report = _repair_component_merges(np.asarray(labels), np.asarray(colors,dtype=np.uint8), base_elements)
    added = len(repaired)-len(base_elements)
    if added:
        strategies = dict(report.get("strategy_counts") or {}); strategies["render_feedback_bridge_cut_anchor"] = added
        report["strategy_counts"] = dict(sorted(strategies.items())); report["path_count"] = int(report.get("path_count") or 0)+added; report["node_count"] = int(report.get("node_count") or 0)+2*added
    report["elements"] = repaired; report["scale_topology_repair"] = repair_report; report["scale_topology_anchor_count"] = added
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
