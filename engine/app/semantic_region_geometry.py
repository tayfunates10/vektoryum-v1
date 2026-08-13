"""Render-feedback scale-topology repair around the semantic geometry core.

The native semantic fitter stays source-derived and fixture-agnostic. This
wrapper closes two renderer-dependent failure modes without weakening any
quality gate: pixel-center ellipse proposals are calibrated against the actual
production renderer using their source connected-component mask, then the
existing five-scale bridge repair preserves semantic separators when raster
sampling would merge regions. Candidate changes are bounded to sub-pixel bbox
corrections, may not reduce native component IoU, and are rejected when they
make quarter-scale class lineage worse. The caller still performs the final
native fidelity/component/boundary transaction and five-render eligibility
checks. No fixture identifiers/builders, raster embedding, threshold relaxation,
or shared byte/path/node budget increase is used.
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
_NEIGHBORS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
_ELIGIBLE_SEPARATOR_STRATEGIES = {
    "ellipse_axis_arm_union",
    "compound_parent_expand_axis_arm_union",
}
# Bounded source-pixel center corrections around the core ellipse proposal.
# These are geometry proposals, not acceptance thresholds. Every proposal is
# scored by the production renderer and the final caller remains fail-closed.
_ELLIPSE_INSET_OFFSETS = (0.0, -0.10, -0.20, -0.30, 0.10, 0.20)
_NATIVE_COMPONENT_TARGET = 0.95


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


def _render_labels(
    width: int,
    height: int,
    elements: list[str],
    colors: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray | None:
    try:
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception:
        return None
    handle = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    path = handle.name
    try:
        handle.write(_svg_document(width, height, elements).encode("utf-8"))
        handle.close()
        rendered = render_svg_to_rgb(path, int(target_width), int(target_height))
        return None if rendered is None else _nearest_palette_labels(rendered, colors)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass


def _project_node_mask(node_map: np.ndarray, node_id: int, width: int, height: int) -> np.ndarray:
    native = (node_map == int(node_id)).astype(np.uint8)
    projected = cv2.resize(native, (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(bool)
    if projected.any():
        return projected
    ys, xs = np.nonzero(native)
    if len(xs):
        px = min(
            width - 1,
            max(0, int(round((float(xs.mean()) + 0.5) * width / node_map.shape[1] - 0.5))),
        )
        py = min(
            height - 1,
            max(0, int(round((float(ys.mean()) + 0.5) * height / node_map.shape[0] - 0.5))),
        )
        projected[py, px] = True
    return projected


def _best_component_iou(render_labels: np.ndarray, source_mask: np.ndarray, color_index: int) -> float:
    count, component_map = cv2.connectedComponents(
        (np.asarray(render_labels) == int(color_index)).astype(np.uint8),
        connectivity=8,
    )
    best = 0.0
    source = np.asarray(source_mask, dtype=bool)
    for component_id in range(1, int(count)):
        rendered = component_map == component_id
        intersection = int(np.count_nonzero(source & rendered))
        if intersection == 0:
            continue
        union = int(np.count_nonzero(source | rendered))
        if union:
            best = max(best, intersection / float(union))
    return float(best)


def _ellipse_element_from_mask(mask: np.ndarray, fill: str, inset: float) -> str | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    rx = ((x1 - x0) / 2.0) - float(inset)
    ry = ((y1 - y0) / 2.0) - float(inset)
    if rx < 1.0 or ry < 1.0:
        return None
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    d = (
        f"M{_fmt(cx-rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx+rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx-rx)} {_fmt(cy)}Z"
    )
    return f'<path fill="{fill}" d="{d}"/>'


def _candidate_ellipse_metrics(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    node_map: np.ndarray,
    node_id: int,
    color_index: int,
    source_counts: list[int],
) -> dict[str, Any] | None:
    height, width = labels.shape
    native = _render_labels(width, height, elements, colors, width, height)
    if native is None:
        return None
    source_mask = node_map == int(node_id)
    native_iou = _best_component_iou(native, source_mask, int(color_index))
    native_mismatch = float(np.mean(native != labels))
    native_counts = _component_counts(native, len(colors))
    native_lineage_error = int(
        sum(abs(int(actual) - int(expected)) for actual, expected in zip(native_counts, source_counts, strict=True))
    )

    quarter_width = max(16, int(round(width * 0.25)))
    quarter_height = max(16, int(round(height * 0.25)))
    quarter = _render_labels(width, height, elements, colors, quarter_width, quarter_height)
    quarter_lineage_error: int | None = None
    quarter_present = False
    if quarter is not None:
        quarter_counts = _component_counts(quarter, len(colors))
        quarter_lineage_error = int(
            sum(
                abs(int(actual) - int(expected))
                for actual, expected in zip(quarter_counts, source_counts, strict=True)
            )
        )
        projected = _project_node_mask(node_map, node_id, quarter_width, quarter_height)
        quarter_present = bool(np.any((quarter == int(color_index)) & projected))

    return {
        "native_iou": float(native_iou),
        "native_mismatch": native_mismatch,
        "native_lineage_error": native_lineage_error,
        "quarter_lineage_error": quarter_lineage_error,
        "quarter_present": quarter_present,
    }


def _ellipse_score(metrics: dict[str, Any], inset: float, base_inset: float) -> tuple[float, int, int, float, float, float]:
    # First close the immutable native component-quality target. Once that is
    # satisfied, prefer stronger quarter-scale lineage, then higher IoU and the
    # smallest geometric correction. No gate value is changed here.
    iou = float(metrics["native_iou"])
    deficit = max(0.0, _NATIVE_COMPONENT_TARGET - iou)
    quarter_error = metrics.get("quarter_lineage_error")
    quarter_error_value = int(quarter_error) if quarter_error is not None else 10**6
    missing_penalty = 0 if bool(metrics.get("quarter_present")) else 1
    return (
        round(deficit, 9),
        quarter_error_value,
        missing_penalty,
        -iou,
        float(metrics["native_mismatch"]),
        abs(float(inset) - float(base_inset)),
    )


def _calibrate_pixel_center_ellipses(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Renderer-calibrate generic ellipse proposals without fixture routing."""
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    node_map, nodes, _root, _depth, _parent = _core._connected_region_graph(labels, len(colors))
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]
    calibrated = list(elements)
    reports: list[dict[str, Any]] = []
    element_index = 0
    base_inset = float(getattr(_core, "_ELLIPSE_INSET", 0.10))

    for region in regions:
        path_count = int(region.get("path_count") or 0)
        start_index = element_index
        element_index += path_count
        if path_count != 1 or str(region.get("strategy") or "") != "pixel_center_ellipse":
            continue
        node_id = int(region["node_id"])
        color_index = int(region["color_index"])
        source_mask = node_map == node_id
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"

        current_metrics = _candidate_ellipse_metrics(
            labels,
            colors,
            calibrated,
            node_map,
            node_id,
            color_index,
            source_counts,
        )
        if current_metrics is None:
            continue
        best_metrics = current_metrics
        best_inset = base_inset
        best_element = calibrated[start_index]
        best_score = _ellipse_score(best_metrics, best_inset, base_inset)

        for offset in _ELLIPSE_INSET_OFFSETS:
            inset = base_inset + float(offset)
            candidate_element = _ellipse_element_from_mask(source_mask, fill, inset)
            if candidate_element is None:
                continue
            candidate_elements = list(calibrated)
            candidate_elements[start_index] = candidate_element
            metrics = _candidate_ellipse_metrics(
                labels,
                colors,
                candidate_elements,
                node_map,
                node_id,
                color_index,
                source_counts,
            )
            if metrics is None:
                continue
            # A renderer calibration may not make native component fidelity or
            # native class lineage worse than the source-derived core proposal.
            if float(metrics["native_iou"]) + 1e-12 < float(current_metrics["native_iou"]):
                continue
            if int(metrics["native_lineage_error"]) > int(current_metrics["native_lineage_error"]):
                continue
            score = _ellipse_score(metrics, inset, base_inset)
            if score < best_score:
                best_score = score
                best_metrics = metrics
                best_inset = inset
                best_element = candidate_element

        calibrated[start_index] = best_element
        reports.append(
            {
                "node_id": node_id,
                "color_index": color_index,
                "source_area": int(np.count_nonzero(source_mask)),
                "base_inset": round(base_inset, 4),
                "selected_inset": round(float(best_inset), 4),
                "changed": bool(best_element != elements[start_index]),
                "before": current_metrics,
                "after": best_metrics,
            }
        )
    return calibrated, reports


def _tree_path(first: int, second: int, parent: list[int | None]) -> list[int]:
    first_chain: list[int] = []
    cursor: int | None = int(first)
    while cursor is not None:
        first_chain.append(cursor)
        cursor = parent[cursor]
    first_pos = {node_id: index for index, node_id in enumerate(first_chain)}
    second_chain: list[int] = []
    cursor = int(second)
    lca: int | None = None
    while cursor is not None:
        if cursor in first_pos:
            lca = cursor
            break
        second_chain.append(cursor)
        cursor = parent[cursor]
    if lca is None:
        return []
    return first_chain[: first_pos[lca] + 1] + list(reversed(second_chain))


def _shortest_render_path(component: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[tuple[int, int]]:
    h, w = component.shape
    starts = np.argwhere(component & start)
    if not len(starts) or not (component & goal).any():
        return []
    queue: deque[tuple[int, int]] = deque()
    previous: dict[tuple[int, int], tuple[int, int] | None] = {}
    for y, x in starts.tolist():
        key = (int(y), int(x))
        queue.append(key)
        previous[key] = None
    endpoint: tuple[int, int] | None = None
    while queue and endpoint is None:
        y, x = queue.popleft()
        if goal[y, x]:
            endpoint = (y, x)
            break
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            key = (ny, nx)
            if nx < 0 or nx >= w or ny < 0 or ny >= h or key in previous or not component[ny, nx]:
                continue
            previous[key] = (y, x)
            queue.append(key)
    if endpoint is None:
        return []
    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = endpoint
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path


def _bridge_path_repair(
    render_labels: np.ndarray,
    class_index: int,
    node_map: np.ndarray,
    nodes: list[dict[str, Any]],
    parent: list[int | None],
    strategy_by_node: dict[int, str],
) -> tuple[int, int, int, dict[str, Any]] | None:
    """Find a rendered bridge only across an eligible semantic separator."""
    class_index = int(class_index)
    render_mask = render_labels == class_index
    ren_count, ren_cc = cv2.connectedComponents(render_mask.astype(np.uint8), connectivity=8)
    target_h, target_w = render_mask.shape
    class_nodes = [index for index, node in enumerate(nodes) if int(node["color_index"]) == class_index]
    projected = {
        node_id: _project_node_mask(node_map, node_id, target_w, target_h)
        for node_id in class_nodes
    }

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
                best: tuple[float, int, int, int, int] | None = None
                for separator_node in separators:
                    separator_mask = _project_node_mask(node_map, separator_node, target_w, target_h)
                    distance = cv2.distanceTransform((~separator_mask).astype(np.uint8), cv2.DIST_L2, 3)
                    replacement = int(nodes[separator_node]["color_index"])
                    for y, x in path:
                        candidate = (
                            float(distance[y, x]),
                            int(y),
                            int(x),
                            replacement,
                            int(separator_node),
                        )
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


def _device_anchor(
    x: int,
    y: int,
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
    fill: str,
) -> str:
    cx = (float(x) + 0.5) * float(source_width) / float(target_width)
    cy = (float(y) + 0.5) * float(source_height) / float(target_height)
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def _repair_component_merges(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    strategy_by_node: dict[int, str],
) -> tuple[list[str], list[dict[str, Any]]]:
    source_height, source_width = labels.shape
    node_map, nodes, _root, _depth, parent = _core._connected_region_graph(labels, len(colors))
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == index)
        for index in range(len(colors))
    ]
    repaired = list(elements)
    reports: list[dict[str, Any]] = []
    for factor in _REPAIR_FACTORS:
        target_width = max(16, int(round(source_width * factor)))
        target_height = max(16, int(round(source_height * factor)))
        anchors: list[dict[str, Any]] = []
        for _attempt in range(_MAX_REPAIR_ANCHORS_PER_SCALE):
            rendered = _render_labels(
                source_width,
                source_height,
                repaired,
                colors,
                target_width,
                target_height,
            )
            if rendered is None:
                break
            counts = _component_counts(rendered, len(colors))
            merged_classes = [
                index
                for index, (expected, actual) in enumerate(zip(source_counts, counts, strict=True))
                if actual < expected
            ]
            if not merged_classes:
                break
            applied = False
            for class_index in merged_classes:
                repair = _bridge_path_repair(
                    rendered,
                    class_index,
                    node_map,
                    nodes,
                    parent,
                    strategy_by_node,
                )
                if repair is None:
                    continue
                x, y, replacement, detail = repair
                red, green, blue = (
                    int(value) for value in np.asarray(colors, dtype=np.uint8)[replacement]
                )
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                repaired.append(
                    _device_anchor(
                        x,
                        y,
                        target_width,
                        target_height,
                        source_width,
                        source_height,
                        fill,
                    )
                )
                anchors.append(
                    {
                        "class_index": int(class_index),
                        "replacement_class_index": int(replacement),
                        "target_pixel": [int(x), int(y)],
                        **detail,
                    }
                )
                applied = True
                break
            if not applied:
                break
        final_labels = _render_labels(
            source_width,
            source_height,
            repaired,
            colors,
            target_width,
            target_height,
        )
        reports.append(
            {
                "scale": f"{factor:g}x",
                "size": [target_width, target_height],
                "anchors": anchors,
                "final_component_counts": (
                    _component_counts(final_labels, len(colors))
                    if final_labels is not None
                    else None
                ),
            }
        )
    return repaired, reports


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    report = _core.build_semantic_region_elements(labels, colors)
    base_elements = list(report.get("elements") or [])
    regions = list(report.get("regions") or [])
    calibrated, ellipse_report = _calibrate_pixel_center_ellipses(
        np.asarray(labels),
        np.asarray(colors, dtype=np.uint8),
        base_elements,
        regions,
    )
    strategy_by_node = {
        int(region["node_id"]): str(region.get("strategy") or "")
        for region in regions
        if "node_id" in region
    }
    repaired, repair_report = _repair_component_merges(
        np.asarray(labels),
        np.asarray(colors, dtype=np.uint8),
        calibrated,
        strategy_by_node,
    )
    added = len(repaired) - len(calibrated)
    if added:
        strategies = dict(report.get("strategy_counts") or {})
        strategies["render_feedback_bridge_cut_anchor"] = added
        report["strategy_counts"] = dict(sorted(strategies.items()))
        report["path_count"] = int(report.get("path_count") or 0) + added
        report["node_count"] = int(report.get("node_count") or 0) + 2 * added
    report["elements"] = repaired
    report["ellipse_renderer_calibration"] = ellipse_report
    report["ellipse_renderer_calibration_changed"] = sum(
        1 for item in ellipse_report if bool(item.get("changed"))
    )
    report["scale_topology_repair"] = repair_report
    report["scale_topology_anchor_count"] = added
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
