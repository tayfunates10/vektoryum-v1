"""Renderer-verified scale-stable semantic primitive reconstruction.

The core fitter remains fixture agnostic. This wrapper inverse-fits simple
raster primitives (ellipse, thin line, micro rectangle) from connected source
masks and evaluates vector proposals with the production renderer at the five
contract scales. A proposal is eligible only when it preserves native
component fidelity/lineage and improves source-derived primitive
self-consistency. The final caller still owns the full fidelity, boundary,
component and budget transactions.
"""
from __future__ import annotations

import os
import tempfile
from collections import deque
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
_NATIVE_COMPONENT_TARGET = 0.95
_MAX_BRIDGE_ANCHORS_PER_SCALE = 12
_NEIGHBORS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
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


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    return max(0, int(count) - 1)


def _component_counts(labels: np.ndarray, color_count: int) -> list[int]:
    return [_component_count(labels == index) for index in range(color_count)]


def _best_component_iou(render_labels: np.ndarray, source_mask: np.ndarray, color_index: int) -> float:
    count, component_map = cv2.connectedComponents(
        (np.asarray(render_labels) == int(color_index)).astype(np.uint8), connectivity=8
    )
    source = np.asarray(source_mask, dtype=bool)
    best = 0.0
    for component_id in range(1, int(count)):
        rendered = component_map == component_id
        intersection = int(np.count_nonzero(source & rendered))
        if not intersection:
            continue
        union = int(np.count_nonzero(source | rendered))
        if union:
            best = max(best, intersection / float(union))
    return float(best)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if not union else float(np.count_nonzero(a & b) / union)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pil_mask(width: int, height: int, draw_fn: Any) -> np.ndarray:
    image = Image.new("L", (int(width), int(height)), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.asarray(image, dtype=np.uint8) > 0


def _scaled(value: float, target: int, source: int) -> int:
    return int(round(float(value) * float(target) / float(source)))


def _model_mask(
    model: dict[str, Any],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    kind = str(model["kind"])
    fx = float(target_width) / float(source_width)
    fy = float(target_height) / float(source_height)
    if kind == "ellipse":
        cx = _scaled(model["cx"], target_width, source_width)
        cy = _scaled(model["cy"], target_height, source_height)
        rx = max(1, int(round(float(model["rx"]) * fx)))
        ry = max(1, int(round(float(model["ry"]) * fy)))
        return _pil_mask(
            target_width,
            target_height,
            lambda draw: draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255),
        )
    if kind == "line":
        x0 = _scaled(model["x0"], target_width, source_width)
        x1 = _scaled(model["x1"], target_width, source_width)
        y0 = _scaled(model["y0"], target_height, source_height)
        y1 = _scaled(model["y1"], target_height, source_height)
        width_factor = fy if model["axis"] == "h" else fx
        stroke = max(1, int(round(float(model["stroke_width"]) * width_factor)))
        return _pil_mask(
            target_width,
            target_height,
            lambda draw: draw.line((x0, y0, x1, y1), fill=255, width=stroke),
        )
    left = _scaled(model["x"], target_width, source_width)
    top = _scaled(model["y"], target_height, source_height)
    width = max(1, int(round(float(model["width"]) * fx)))
    height = max(1, int(round(float(model["height"]) * fy)))
    return _pil_mask(
        target_width,
        target_height,
        lambda draw: draw.rectangle((left, top, left + width - 1, top + height - 1), fill=255),
    )


def _infer_line_model(mask: np.ndarray) -> dict[str, Any] | None:
    bounds = _bbox(mask)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0 + 1, y1 - y0 + 1
    short, long = min(width, height), max(width, height)
    if short > 4 or long < max(5, short * 3):
        return None
    source = np.asarray(mask, dtype=bool)
    canvas_h, canvas_w = source.shape
    best: tuple[float, dict[str, Any]] | None = None
    if width >= height:
        for anchor in range(y0 - 2, y1 + 3):
            model = {
                "kind": "line",
                "axis": "h",
                "x0": x0,
                "y0": anchor,
                "x1": x1,
                "y1": anchor,
                "stroke_width": height,
            }
            predicted = _model_mask(model, canvas_w, canvas_h, canvas_w, canvas_h)
            score = _mask_iou(source, predicted)
            if best is None or score > best[0]:
                best = (score, model)
    else:
        for anchor in range(x0 - 2, x1 + 3):
            model = {
                "kind": "line",
                "axis": "v",
                "x0": anchor,
                "y0": y0,
                "x1": anchor,
                "y1": y1,
                "stroke_width": width,
            }
            predicted = _model_mask(model, canvas_w, canvas_h, canvas_w, canvas_h)
            score = _mask_iou(source, predicted)
            if best is None or score > best[0]:
                best = (score, model)
    return best[1] if best is not None and best[0] >= 0.98 else None


def _infer_primitive(mask: np.ndarray, strategy: str) -> dict[str, Any] | None:
    bounds = _bbox(mask)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0 + 1, y1 - y0 + 1
    occupancy = float(np.mean(np.asarray(mask, dtype=bool)[y0 : y1 + 1, x0 : x1 + 1]))
    if "ellipse" in str(strategy):
        rx = (width - 1) / 2.0
        ry = (height - 1) / 2.0
        if min(rx, ry) >= 1.0:
            model = {
                "kind": "ellipse",
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
                "rx": rx,
                "ry": ry,
            }
            predicted = _model_mask(model, mask.shape[1], mask.shape[0], mask.shape[1], mask.shape[0])
            if _mask_iou(mask, predicted) >= 0.94:
                return model
    if occupancy >= 0.999:
        line = _infer_line_model(mask)
        if line is not None:
            return line
        if max(width, height) <= 8:
            return {"kind": "rect", "x": x0, "y": y0, "width": width, "height": height}
    return None


def _candidate_elements(model: dict[str, Any], fill: str) -> list[list[str]]:
    kind = str(model["kind"])
    output: list[list[str]] = []
    if kind == "ellipse":
        cx, cy = float(model["cx"]), float(model["cy"])
        rx, ry = float(model["rx"]), float(model["ry"])
        for offset in (0.0, 0.125, 0.25):
            output.append([
                f'<ellipse cx="{_fmt(cx + offset)}" cy="{_fmt(cy + offset)}" '
                f'rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" '
                f'stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'
            ])
            output.append([
                f'<ellipse cx="{_fmt(cx + offset)}" cy="{_fmt(cy + offset)}" '
                f'rx="{_fmt(rx + 0.25)}" ry="{_fmt(ry + 0.25)}" fill="{fill}"/>'
            ])
        return output
    if kind == "line":
        x0, y0 = float(model["x0"]), float(model["y0"])
        x1, y1 = float(model["x1"]), float(model["y1"])
        stroke = float(model["stroke_width"])
        for offset in (0.0, 0.25, 0.5):
            for extension in (0.0, 0.25):
                if model["axis"] == "h":
                    d = f"M{_fmt(x0)} {_fmt(y0 + offset)}H{_fmt(x1 + extension)}"
                else:
                    d = f"M{_fmt(x0 + offset)} {_fmt(y0)}V{_fmt(y1 + extension)}"
                normal = (
                    f'<path d="{d}" fill="none" stroke="{fill}" '
                    f'stroke-width="{_fmt(stroke)}" stroke-linecap="butt"/>'
                )
                minimum = (
                    f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="1" '
                    'vector-effect="non-scaling-stroke" stroke-linecap="butt"/>'
                )
                output.append([normal, minimum])
        return output
    x, y = float(model["x"]), float(model["y"])
    width, height = float(model["width"]), float(model["height"])
    output.append([
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" fill="{fill}"/>'
    ])
    cx, cy = x + width / 2.0, y + height / 2.0
    point = f"M{_fmt(cx)} {_fmt(cy)}h0.001"
    if abs(width - height) <= 1e-9:
        output.append([
            f'<path d="{point}" fill="none" stroke="{fill}" stroke-width="{_fmt(width)}" '
            'stroke-linecap="square"/>',
            f'<path d="{point}" fill="none" stroke="{fill}" stroke-width="1" '
            'vector-effect="non-scaling-stroke" stroke-linecap="square"/>',
        ])
    output.append([
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" fill="{fill}"/>',
        f'<path d="M{_fmt(x + 0.5)} {_fmt(y + 0.5)}h0.001" fill="none" stroke="{fill}" '
        'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>',
    ])
    return output


def _flatten(parts: list[list[str]]) -> list[str]:
    return [element for part in parts for element in part]


def _primitive_score(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    node_map: np.ndarray,
    node_id: int,
    color_index: int,
    source_counts: list[int],
    model: dict[str, Any],
) -> dict[str, Any] | None:
    source_h, source_w = labels.shape
    source_mask = node_map == int(node_id)
    scale_ious: list[float] = []
    lineage_errors: list[int] = []
    native_mismatch: float | None = None
    native_iou: float | None = None
    for factor in _FACTORS:
        target_w = max(16, int(round(source_w * factor)))
        target_h = max(16, int(round(source_h * factor)))
        rendered = _render_labels(source_w, source_h, elements, colors, target_w, target_h)
        if rendered is None:
            return None
        expected = _model_mask(model, source_w, source_h, target_w, target_h)
        scale_ious.append(_best_component_iou(rendered, expected, color_index))
        counts = _component_counts(rendered, len(colors))
        lineage_errors.append(
            int(sum(abs(int(a) - int(b)) for a, b in zip(counts, source_counts, strict=True)))
        )
        if factor == 1.0:
            native_mismatch = float(np.mean(rendered != labels))
            native_iou = _best_component_iou(rendered, source_mask, color_index)
    if native_mismatch is None or native_iou is None:
        return None
    return {
        "native_iou": float(native_iou),
        "native_mismatch": float(native_mismatch),
        "lineage_errors": lineage_errors,
        "scale_ious": scale_ious,
        "min_scale_iou": float(min(scale_ious)),
        "mean_scale_iou": float(np.mean(scale_ious)),
    }


def _score_tuple(metrics: dict[str, Any], element_count: int) -> tuple[float, float, float, float, int]:
    return (
        float(sum(int(value) for value in metrics["lineage_errors"])),
        -float(metrics["min_scale_iou"]),
        -float(metrics["mean_scale_iou"]),
        float(metrics["native_mismatch"]),
        int(element_count),
    )


def _regularize_primitives(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    node_map, nodes, _root, _depth, _parent = _core._connected_region_graph(labels, len(colors))
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
        return list(elements), []
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        if len(parts[index]) != 1:
            continue
        node_id = int(region["node_id"])
        color_index = int(region["color_index"])
        model = _infer_primitive(node_map == node_id, str(region.get("strategy") or ""))
        if model is None:
            continue
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        baseline = _primitive_score(
            labels, colors, _flatten(parts), node_map, node_id, color_index, source_counts, model
        )
        if baseline is None:
            continue
        best_metrics = baseline
        best_part = list(parts[index])
        best_score = _score_tuple(baseline, len(best_part))
        for candidate_part in _candidate_elements(model, fill):
            trial_parts = list(parts)
            trial_parts[index] = candidate_part
            metrics = _primitive_score(
                labels, colors, _flatten(trial_parts), node_map, node_id, color_index, source_counts, model
            )
            if metrics is None:
                continue
            if float(metrics["native_iou"]) + 1e-12 < _NATIVE_COMPONENT_TARGET:
                continue
            if any(int(value) != 0 for value in metrics["lineage_errors"]):
                continue
            if float(metrics["native_mismatch"]) > float(baseline["native_mismatch"]) + 0.00025:
                continue
            score = _score_tuple(metrics, len(candidate_part))
            if score < best_score:
                best_score = score
                best_metrics = metrics
                best_part = candidate_part
        changed = best_part != parts[index]
        if changed:
            parts[index] = best_part
        reports.append({
            "node_id": node_id,
            "color_index": color_index,
            "strategy": str(region.get("strategy") or ""),
            "model": model,
            "changed": changed,
            "before": baseline,
            "after": best_metrics,
            "element_delta": len(best_part) - 1,
        })
    return _flatten(parts), reports


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
    chain: list[int] = []
    cursor: int | None = int(first)
    while cursor is not None:
        chain.append(cursor)
        cursor = parent[cursor]
    positions = {node_id: index for index, node_id in enumerate(chain)}
    other: list[int] = []
    cursor = int(second)
    while cursor is not None and cursor not in positions:
        other.append(cursor)
        cursor = parent[cursor]
    if cursor is None:
        return []
    return chain[: positions[cursor] + 1] + list(reversed(other))


def _shortest_render_path(component: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[tuple[int, int]]:
    h, w = component.shape
    starts = np.argwhere(component & start)
    if not len(starts) or not np.any(component & goal):
        return []
    queue: deque[tuple[int, int]] = deque()
    previous: dict[tuple[int, int], tuple[int, int] | None] = {}
    for y, x in starts.tolist():
        point = (int(y), int(x))
        queue.append(point)
        previous[point] = None
    endpoint: tuple[int, int] | None = None
    while queue and endpoint is None:
        y, x = queue.popleft()
        if goal[y, x]:
            endpoint = (y, x)
            break
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            point = (ny, nx)
            if nx < 0 or nx >= w or ny < 0 or ny >= h or point in previous or not component[ny, nx]:
                continue
            previous[point] = (y, x)
            queue.append(point)
    if endpoint is None:
        return []
    output: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = endpoint
    while cursor is not None:
        output.append(cursor)
        cursor = previous[cursor]
    output.reverse()
    return output


def _bridge_path_repair(
    render_labels: np.ndarray,
    class_index: int,
    node_map: np.ndarray,
    nodes: list[dict[str, Any]],
    parent: list[int | None],
    strategy_by_node: dict[int, str],
) -> tuple[int, int, int, dict[str, Any]] | None:
    render_mask = np.asarray(render_labels) == int(class_index)
    count, rendered_components = cv2.connectedComponents(render_mask.astype(np.uint8), connectivity=8)
    height, width = render_mask.shape
    class_nodes = [
        node_id for node_id, node in enumerate(nodes) if int(node["color_index"]) == int(class_index)
    ]
    projected = {node_id: _project_node_mask(node_map, node_id, width, height) for node_id in class_nodes}
    for component_id in range(1, int(count)):
        component = rendered_components == component_id
        overlapping = [node_id for node_id in class_nodes if np.any(component & projected[node_id])]
        for pair_index, first in enumerate(overlapping[:-1]):
            for second in overlapping[pair_index + 1 :]:
                source_path = _tree_path(first, second, parent)
                separators = [
                    node_id
                    for node_id in source_path[1:-1]
                    if int(nodes[node_id]["color_index"]) != int(class_index)
                    and strategy_by_node.get(node_id) in _ELIGIBLE_SEPARATOR_STRATEGIES
                ]
                if not separators:
                    continue
                path = _shortest_render_path(component, projected[first], projected[second])
                if not path:
                    continue
                best: tuple[float, int, int, int, int] | None = None
                for separator_node in separators:
                    separator = _project_node_mask(node_map, separator_node, width, height)
                    distance = cv2.distanceTransform((~separator).astype(np.uint8), cv2.DIST_L2, 3)
                    replacement = int(nodes[separator_node]["color_index"])
                    for y, x in path:
                        candidate = (float(distance[y, x]), int(y), int(x), replacement, int(separator_node))
                        if best is None or candidate < best:
                            best = candidate
                if best is not None:
                    distance, y, x, replacement, separator_node = best
                    return x, y, replacement, {
                        "source_node_pair": [int(first), int(second)],
                        "separator_node": separator_node,
                        "separator_strategy": strategy_by_node.get(separator_node),
                        "distance_to_separator": round(distance, 4),
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
    source_h, source_w = labels.shape
    node_map, nodes, _root, _depth, parent = _core._connected_region_graph(labels, len(colors))
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]
    repaired = list(elements)
    reports: list[dict[str, Any]] = []
    for factor in _FACTORS:
        target_w = max(16, int(round(source_w * factor)))
        target_h = max(16, int(round(source_h * factor)))
        anchors: list[dict[str, Any]] = []
        for _ in range(_MAX_BRIDGE_ANCHORS_PER_SCALE):
            rendered = _render_labels(source_w, source_h, repaired, colors, target_w, target_h)
            if rendered is None:
                break
            counts = _component_counts(rendered, len(colors))
            merged = [
                color_index
                for color_index, (expected, actual) in enumerate(zip(source_counts, counts, strict=True))
                if actual < expected
            ]
            if not merged:
                break
            applied = False
            for color_index in merged:
                repair = _bridge_path_repair(rendered, color_index, node_map, nodes, parent, strategy_by_node)
                if repair is None:
                    continue
                x, y, replacement, detail = repair
                red, green, blue = (int(value) for value in colors[replacement])
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                repaired.append(_device_anchor(x, y, target_w, target_h, source_w, source_h, fill))
                anchors.append({
                    "class_index": int(color_index),
                    "replacement_class_index": int(replacement),
                    "target_pixel": [int(x), int(y)],
                    **detail,
                })
                applied = True
                break
            if not applied:
                break
        final_render = _render_labels(source_w, source_h, repaired, colors, target_w, target_h)
        reports.append({
            "scale": f"{factor:g}x",
            "size": [target_w, target_h],
            "anchors": anchors,
            "final_component_counts": _component_counts(final_render, len(colors)) if final_render is not None else None,
        })
    return repaired, reports


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    report = _core.build_semantic_region_elements(labels, colors)
    base_elements = list(report.get("elements") or [])
    regions = list(report.get("regions") or [])
    regularized, primitive_report = _regularize_primitives(labels, colors, base_elements, regions)
    strategy_by_node = {
        int(region["node_id"]): str(region.get("strategy") or "")
        for region in regions
        if "node_id" in region
    }
    repaired, repair_report = _repair_component_merges(labels, colors, regularized, strategy_by_node)
    primitive_delta = len(regularized) - len(base_elements)
    bridge_delta = len(repaired) - len(regularized)
    total_delta = len(repaired) - len(base_elements)
    if total_delta:
        strategies = dict(report.get("strategy_counts") or {})
        if primitive_delta:
            strategies["scale_stable_primitive_reconstruction"] = primitive_delta
        if bridge_delta:
            strategies["render_feedback_bridge_cut_anchor"] = bridge_delta
        report["strategy_counts"] = dict(sorted(strategies.items()))
        report["path_count"] = int(report.get("path_count") or 0) + total_delta
        report["node_count"] = int(report.get("node_count") or 0) + 2 * total_delta
    report["elements"] = repaired
    report["scale_stable_primitive_reconstruction"] = primitive_report
    report["scale_stable_primitive_changed"] = sum(1 for item in primitive_report if item.get("changed"))
    report["scale_topology_repair"] = repair_report
    report["scale_topology_anchor_count"] = bridge_delta
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
