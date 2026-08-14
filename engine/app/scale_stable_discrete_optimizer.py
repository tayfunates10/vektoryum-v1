"""Fail-closed five-scale refinement for source-derived compact geometry.

This pass is deliberately narrower than the rejected broad discrete sweep.  It
never touches line models.  Only (a) compact leaf rectangles, (b) leaf ellipses,
and (c) analytically verified nested parent ellipses/rounded rectangles are
eligible.  Every proposal is inferred from the connected source mask, rendered
with the production renderer at 0.25x/0.5x/1x/2x/4x, and rejected when native
fidelity or component lineage regresses.  No fixture identifiers, coordinates,
threshold changes, raster embedding, or budget changes are used here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app import scale_stable_parent_geometry as _parent
from app.scale_stable_primitive_geometry import FACTORS, model_mask


def _fmt(value: float) -> str:
    return _impl._fmt(float(value))


def _flatten(parts: list[list[str]]) -> list[str]:
    return [element for part in parts for element in part]


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def _isolated_metrics(
    model: dict[str, Any],
    elements: list[str],
    fill: str,
    sw: int,
    sh: int,
) -> dict[str, Any] | None:
    colors = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    background = f'<path fill="#ffffff" d="M0 0H{sw}V{sh}H0Z"/>'
    trial = [background] + _blacken(elements, fill)
    levels: list[dict[str, Any]] = []
    for factor in FACTORS:
        tw = max(16, int(round(sw * factor)))
        th = max(16, int(round(sh * factor)))
        rendered = _impl._render(sw, sh, trial, colors, tw, th)
        if rendered is None:
            return None
        actual = rendered == 0
        expected = model_mask(model, sw, sh, tw, th)
        intersection = int(np.count_nonzero(actual & expected))
        union = int(np.count_nonzero(actual | expected))
        levels.append(
            {
                "scale": f"{factor:g}x",
                "iou": 1.0 if union == 0 else intersection / float(union),
                "mismatch": float(np.mean(actual != expected)),
                "components": int(_impl._cc(actual)),
            }
        )
    return {
        "levels": levels,
        "component_error": int(sum(abs(int(level["components"]) - 1) for level in levels)),
        "min_iou": float(min(float(level["iou"]) for level in levels)),
        "mean_iou": float(np.mean([float(level["iou"]) for level in levels])),
        "max_mismatch": float(max(float(level["mismatch"]) for level in levels)),
        "native_iou": float(levels[2]["iou"]),
    }


def _isolated_score(metrics: dict[str, Any], element_count: int) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["component_error"]),
        -float(metrics["min_iou"]),
        float(metrics["max_mismatch"]),
        -float(metrics["mean_iou"]),
        int(element_count),
    )


def _scene_metrics(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    node_map: np.ndarray,
    node_id: int,
    color_index: int,
    source_counts: list[int],
) -> dict[str, Any] | None:
    sh, sw = labels.shape
    errors: list[int] = []
    counts: list[list[int]] = []
    native_iou: float | None = None
    native_mismatch: float | None = None
    for factor in FACTORS:
        tw = max(16, int(round(sw * factor)))
        th = max(16, int(round(sh * factor)))
        rendered = _impl._render(sw, sh, elements, colors, tw, th)
        if rendered is None:
            return None
        actual = [int(value) for value in _impl._counts(rendered, len(colors))]
        counts.append(actual)
        errors.append(
            int(
                sum(
                    abs(int(current) - int(expected))
                    for current, expected in zip(actual, source_counts, strict=True)
                )
            )
        )
        if factor == 1.0:
            native_mismatch = float(np.mean(rendered != labels))
            native_iou = float(
                _impl._best_iou(
                    rendered,
                    node_map == int(node_id),
                    int(color_index),
                )
            )
    return {
        "lineage_errors": errors,
        "class_component_counts": counts,
        "native_iou": float(native_iou or 0.0),
        "native_mismatch": float(native_mismatch or 0.0),
    }


def _scene_safe(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if float(after["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
        return False
    if float(after["native_iou"]) + 1e-12 < float(before["native_iou"]):
        return False
    if float(after["native_mismatch"]) > float(before["native_mismatch"]) + 1e-12:
        return False
    before_errors = [int(value) for value in before["lineage_errors"]]
    after_errors = [int(value) for value in after["lineage_errors"]]
    return not any(
        current > previous
        for current, previous in zip(after_errors, before_errors, strict=True)
    )


def _cardinal_anchor_path(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    fill: str,
    width: float,
) -> str:
    d = " ".join(
        (
            f'M{_fmt(cx-rx)} {_fmt(cy)}h0.001',
            f'M{_fmt(cx+rx)} {_fmt(cy)}h0.001',
            f'M{_fmt(cx)} {_fmt(cy-ry)}h0.001',
            f'M{_fmt(cx)} {_fmt(cy+ry)}h0.001',
        )
    )
    return (
        f'<path d="{d}" fill="none" stroke="{fill}" stroke-width="{_fmt(width)}" '
        'vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def _ellipse_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    cx, cy, rx, ry = (
        float(model[key]) for key in ("cx", "cy", "rx", "ry")
    )
    output: list[list[str]] = []
    for center_delta in (-0.125, 0.0, 0.125):
        for radius_delta in (-0.125, 0.0, 0.125):
            nrx = max(0.5, rx + radius_delta)
            nry = max(0.5, ry + radius_delta)
            output.append(
                [
                    f'<ellipse cx="{_fmt(cx+center_delta)}" cy="{_fmt(cy+center_delta)}" '
                    f'rx="{_fmt(nrx)}" ry="{_fmt(nry)}" fill="{fill}"/>'
                ]
            )
    for center_delta, radius_delta in ((0.0, 0.0), (0.125, 0.125), (0.125, 0.0)):
        ncx = cx + center_delta
        ncy = cy + center_delta
        nrx = max(0.5, rx + radius_delta)
        nry = max(0.5, ry + radius_delta)
        base = (
            f'<ellipse cx="{_fmt(ncx)}" cy="{_fmt(ncy)}" '
            f'rx="{_fmt(nrx)}" ry="{_fmt(nry)}" fill="{fill}"/>'
        )
        for anchor_width in (0.75, 1.0):
            output.append(
                [
                    base,
                    _cardinal_anchor_path(ncx, ncy, nrx, nry, fill, anchor_width),
                ]
            )
    return output


def _compact_rect_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x, y, width, height = (
        float(model[key]) for key in ("x", "y", "width", "height")
    )
    output: list[list[str]] = []
    shifts = ((0.0, 0.0), (-0.125, -0.125), (0.125, 0.125), (-0.125, 0.0), (0.0, -0.125), (0.125, 0.0), (0.0, 0.125))
    for size_delta in (-0.25, -0.125, 0.0, 0.125, 0.25):
        rw = max(0.5, width + size_delta)
        rh = max(0.5, height + size_delta)
        for dx, dy in shifts:
            output.append(
                [
                    f'<rect x="{_fmt(x+dx)}" y="{_fmt(y+dy)}" '
                    f'width="{_fmt(rw)}" height="{_fmt(rh)}" fill="{fill}"/>'
                ]
            )
    return output


def _rounded_parent_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x0, y0, x1, y1, radius = (
        float(model[key]) for key in ("x0", "y0", "x1", "y1", "radius")
    )
    base_width = x1 - x0 + 1.0
    base_height = y1 - y0 + 1.0
    output: list[list[str]] = []
    for shift in (-0.25, -0.1875, -0.125, -0.0625, 0.0, 0.0625):
        for size_delta in (0.0, 0.125, 0.25, 0.375):
            for radius_delta in (-0.25, 0.0, 0.25):
                rr = max(0.5, radius + radius_delta)
                output.append(
                    [
                        f'<rect x="{_fmt(x0+shift)}" y="{_fmt(y0+shift)}" '
                        f'width="{_fmt(base_width+size_delta)}" '
                        f'height="{_fmt(base_height+size_delta)}" '
                        f'rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="{fill}"/>'
                    ]
                )
    return output


def _parent_ellipse_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    cx, cy, rx, ry = (
        float(model[key]) for key in ("cx", "cy", "rx", "ry")
    )
    output: list[list[str]] = []
    for center_delta in (-0.125, 0.0, 0.125):
        for radius_delta in (-0.25, -0.125, 0.0, 0.125, 0.25):
            nrx = max(0.5, rx + radius_delta)
            nry = max(0.5, ry + radius_delta)
            output.append(
                [
                    f'<ellipse cx="{_fmt(cx+center_delta)}" cy="{_fmt(cy+center_delta)}" '
                    f'rx="{_fmt(nrx)}" ry="{_fmt(nry)}" fill="{fill}"/>'
                ]
            )
    return output


def _eligible_leaf_model(model: dict[str, Any]) -> bool:
    kind = str(model.get("kind") or "")
    if kind == "ellipse":
        return True
    if kind != "rect":
        return False
    width = float(model.get("width") or 0.0)
    height = float(model.get("height") or 0.0)
    if min(width, height) <= 0.0 or max(width, height) > 8.0:
        return False
    return max(width, height) / max(1e-9, min(width, height)) <= 1.25


def refine_scale_stable_geometry(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
    primitive_report: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    node_map, nodes, _root, _depth, parent = _impl._core._connected_region_graph(
        labels, len(colors)
    )
    child_count = [0 for _ in nodes]
    for parent_id in parent:
        if parent_id is not None:
            child_count[int(parent_id)] += 1
    source_counts = [
        sum(1 for node in nodes if int(node["color_index"]) == color_index)
        for color_index in range(len(colors))
    ]

    report_by_node = {
        int(item["node_id"]): item
        for item in primitive_report
        if isinstance(item, dict) and "node_id" in item
    }
    delta_by_node = {
        node_id: int(item.get("element_delta") or 0)
        for node_id, item in report_by_node.items()
    }
    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        node_id = int(region.get("node_id") or -1)
        count = int(region.get("path_count") or 0) + delta_by_node.get(node_id, 0)
        parts.append(list(elements[cursor:cursor+count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), []

    sh, sw = labels.shape
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        node_id = int(region.get("node_id") or -1)
        if node_id < 0 or node_id >= len(nodes) or not parts[index]:
            continue
        color_index = int(region["color_index"])
        model: dict[str, Any] | None = None
        family: str | None = None

        prior = report_by_node.get(node_id)
        if prior is not None:
            proposed = dict(prior.get("model") or {})
            if _eligible_leaf_model(proposed) and child_count[node_id] == 0:
                model = proposed
                family = "leaf_compact"

        if model is None and child_count[node_id] > 0 and len(parts[index]) == 1:
            source_mask = node_map == node_id
            proposed = _parent._rounded_model(source_mask)
            if proposed is None:
                proposed = _parent._ellipse_model(source_mask)
            if proposed is not None:
                model = dict(proposed)
                family = "nested_parent"

        if model is None or family is None:
            continue

        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        base_part = list(parts[index])
        isolated_before = _isolated_metrics(model, base_part, fill, sw, sh)
        scene_before = _scene_metrics(
            labels,
            colors,
            _flatten(parts),
            node_map,
            node_id,
            color_index,
            source_counts,
        )
        if isolated_before is None or scene_before is None:
            continue

        # Healthy source-derived geometry is immutable in this pass.
        healthy_floor = 0.995 if family == "nested_parent" else float(_impl._NATIVE_MIN_IOU)
        if (
            int(isolated_before["component_error"]) == 0
            and float(isolated_before["min_iou"]) + 1e-12 >= healthy_floor
        ):
            continue

        kind = str(model["kind"])
        if family == "nested_parent" and kind == "rounded_rect":
            candidates = _rounded_parent_candidates(model, fill)
        elif family == "nested_parent" and kind == "ellipse":
            candidates = _parent_ellipse_candidates(model, fill)
        elif kind == "ellipse":
            candidates = _ellipse_candidates(model, fill)
        elif kind == "rect":
            candidates = _compact_rect_candidates(model, fill)
        else:
            continue

        best_part = base_part
        best_isolated = isolated_before
        best_scene = scene_before
        best_key = _isolated_score(isolated_before, len(base_part)) + (
            float(scene_before["native_mismatch"]),
        )
        for candidate in candidates:
            isolated = _isolated_metrics(model, candidate, fill, sw, sh)
            if isolated is None:
                continue
            if float(isolated["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            trial_parts = list(parts)
            trial_parts[index] = list(candidate)
            scene = _scene_metrics(
                labels,
                colors,
                _flatten(trial_parts),
                node_map,
                node_id,
                color_index,
                source_counts,
            )
            if scene is None or not _scene_safe(scene_before, scene):
                continue
            key = _isolated_score(isolated, len(candidate)) + (
                float(scene["native_mismatch"]),
            )
            if key < best_key:
                best_key = key
                best_part = list(candidate)
                best_isolated = isolated
                best_scene = scene

        changed = best_part != base_part
        if changed:
            parts[index] = best_part
        reports.append(
            {
                "node_id": node_id,
                "color_index": color_index,
                "family": family,
                "model": model,
                "changed": changed,
                "before": {
                    "isolated": isolated_before,
                    "scene": scene_before,
                },
                "after": {
                    "isolated": best_isolated,
                    "scene": best_scene,
                },
                "element_delta": len(best_part) - len(base_part),
            }
        )

    return _flatten(parts), reports


# Backward-compatible name for the previously experimental module.
def optimize_discrete_primitives(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
    primitive_report: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    return refine_scale_stable_geometry(
        labels, colors, elements, regions, primitive_report
    )


__all__ = ["optimize_discrete_primitives", "refine_scale_stable_geometry"]
