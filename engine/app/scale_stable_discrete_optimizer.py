"""Bounded five-scale optimizer for source-derived discrete primitives.

The established semantic fitter and primitive calibrator remain authoritative.
This pass only revisits a calibrated line/rect/ellipse whose actual production
renderer score still has min IoU below the existing 0.95 component contract.
Candidates are generic sub-pixel SVG representations derived from that region's
source model; no fixture identifiers, coordinates, corpus data, thresholds, or
budgets are imported or changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_primitive_geometry import FACTORS, model_mask


def _fmt(value: float) -> str:
    return _impl._fmt(float(value))


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def _metrics(
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
        "component_error": int(sum(abs(int(x["components"]) - 1) for x in levels)),
        "min_iou": float(min(float(x["iou"]) for x in levels)),
        "mean_iou": float(np.mean([float(x["iou"]) for x in levels])),
        "max_mismatch": float(max(float(x["mismatch"]) for x in levels)),
        "native_iou": float(levels[2]["iou"]),
    }


def _score(metrics: dict[str, Any], element_count: int) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["component_error"]),
        -float(metrics["min_iou"]),
        float(metrics["max_mismatch"]),
        -float(metrics["mean_iou"]),
        int(element_count),
    )


def _line_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x0, y0, x1, y1, width = (
        float(model[key]) for key in ("x0", "y0", "x1", "y1", "width")
    )
    axis = str(model["axis"])
    out: list[list[str]] = []
    # SVG strokes: bounded center/width sweep.  Fractions are source geometry,
    # not scale-specific branches; the production renderer chooses one geometry.
    for center_offset in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        for stroke_delta in (-0.25, 0.0, 0.25):
            stroke = max(0.5, width + stroke_delta)
            if axis == "h":
                path = f'M{_fmt(x0)} {_fmt(y0 + center_offset)}H{_fmt(x1)}'
            else:
                path = f'M{_fmt(x0 + center_offset)} {_fmt(y0)}V{_fmt(y1)}'
            for cap in ("butt", "square"):
                out.append(
                    [
                        f'<path d="{path}" fill="none" stroke="{fill}" '
                        f'stroke-width="{_fmt(stroke)}" stroke-linecap="{cap}"/>'
                    ]
                )
    # Filled strips avoid stroke-centering ambiguity.  Pillow's line endpoint is
    # inclusive, so both endpoint-distance and inclusive-width proposals are
    # considered with bounded sub-pixel placement.
    if axis == "h":
        length = max(0.001, x1 - x0)
        for y_shift in (-1.0, -0.75, -0.5, -0.25, 0.0):
            for length_add in (0.0, 0.125, 0.25, 0.5, 1.0):
                for height_delta in (-0.125, 0.0, 0.125):
                    height = max(0.5, width + height_delta)
                    out.append(
                        [
                            f'<rect x="{_fmt(x0)}" y="{_fmt(y0 + y_shift)}" '
                            f'width="{_fmt(length + length_add)}" height="{_fmt(height)}" '
                            f'fill="{fill}" shape-rendering="crispEdges"/>'
                        ]
                    )
    else:
        length = max(0.001, y1 - y0)
        for x_shift in (-1.0, -0.75, -0.5, -0.25, 0.0):
            for length_add in (0.0, 0.125, 0.25, 0.5, 1.0):
                for width_delta in (-0.125, 0.0, 0.125):
                    strip_width = max(0.5, width + width_delta)
                    out.append(
                        [
                            f'<rect x="{_fmt(x0 + x_shift)}" y="{_fmt(y0)}" '
                            f'width="{_fmt(strip_width)}" height="{_fmt(length + length_add)}" '
                            f'fill="{fill}" shape-rendering="crispEdges"/>'
                        ]
                    )
    return out


def _rect_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x, y, width, height = (
        float(model[key]) for key in ("x", "y", "width", "height")
    )
    out: list[list[str]] = []
    for pad in (-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25):
        for dx, dy in ((0.0, 0.0), (-0.0625, 0.0), (0.0625, 0.0), (0.0, -0.0625), (0.0, 0.0625)):
            w = max(0.5, width + pad)
            h = max(0.5, height + pad)
            out.append(
                [
                    f'<rect x="{_fmt(x + dx)}" y="{_fmt(y + dy)}" '
                    f'width="{_fmt(w)}" height="{_fmt(h)}" fill="{fill}" '
                    f'shape-rendering="crispEdges"/>'
                ]
            )
    return out


def _ellipse_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    cx, cy, rx, ry = (
        float(model[key]) for key in ("cx", "cy", "rx", "ry")
    )
    out: list[list[str]] = []
    for radius_delta in (-0.5, -0.375, -0.25, -0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375):
        nrx = max(0.5, rx + radius_delta)
        nry = max(0.5, ry + radius_delta)
        for center_delta in (-0.125, 0.0, 0.125):
            out.append(
                [
                    f'<ellipse cx="{_fmt(cx + center_delta)}" cy="{_fmt(cy + center_delta)}" '
                    f'rx="{_fmt(nrx)}" ry="{_fmt(nry)}" fill="{fill}"/>'
                ]
            )
    return out


def _candidate_elements(model: dict[str, Any], fill: str) -> list[list[str]]:
    kind = str(model.get("kind") or "")
    if kind == "line":
        return _line_candidates(model, fill)
    if kind == "rect":
        return _rect_candidates(model, fill)
    if kind == "ellipse":
        return _ellipse_candidates(model, fill)
    return []


def optimize_discrete_primitives(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
    primitive_report: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    labels = np.asarray(labels)
    colors = np.asarray(colors, dtype=np.uint8)
    report_by_node = {
        int(item["node_id"]): item
        for item in primitive_report
        if isinstance(item, dict) and "node_id" in item
    }
    delta_by_node = {
        int(node_id): int(item.get("element_delta") or 0)
        for node_id, item in report_by_node.items()
    }

    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        node_id = int(region.get("node_id") or -1)
        count = int(region.get("path_count") or 0) + delta_by_node.get(node_id, 0)
        parts.append(list(elements[cursor:cursor + count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), []

    sh, sw = labels.shape
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        node_id = int(region.get("node_id") or -1)
        prior = report_by_node.get(node_id)
        if not prior:
            continue
        model = prior.get("model") or {}
        if str(model.get("kind") or "") not in {"line", "rect", "ellipse"}:
            continue
        color_index = int(region["color_index"])
        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        base = list(parts[index])
        before = _metrics(model, base, fill, sw, sh)
        if before is None:
            continue
        # Healthy primitives are immutable in this pass.
        if float(before["min_iou"]) + 1e-12 >= float(_impl._NATIVE_MIN_IOU):
            continue
        best = before
        best_part = base
        best_key = _score(before, len(base))
        for candidate in _candidate_elements(model, fill):
            metrics = _metrics(model, candidate, fill, sw, sh)
            if metrics is None:
                continue
            if float(metrics["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            key = _score(metrics, len(candidate))
            if key < best_key:
                best_key = key
                best = metrics
                best_part = list(candidate)
        changed = best_part != base
        if changed:
            parts[index] = best_part
        reports.append(
            {
                "node_id": node_id,
                "color_index": color_index,
                "model": model,
                "changed": changed,
                "before": before,
                "after": best,
                "element_delta": len(best_part) - len(base),
            }
        )

    return [element for part in parts for element in part], reports


__all__ = ["optimize_discrete_primitives"]
