"""Source-derived calibration for nested parent paint regions only.

Leaf primitives and ordinary top-level shapes remain with the established
calibrator. A connected region that contains child paint may have an external
ellipse/rounded-rectangle boundary represented by raster-oriented transforms in
the core. We infer the filled external boundary from the source mask, require a
near-exact native Pillow primitive fit, and choose bounded SVG proposals with the
real production renderer across all five scales. Inclusive endpoint proposals
encode Pillow's fixed one-device-pixel term without changing any acceptance
threshold or budget.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_primitive_geometry import FACTORS, model_mask

_PARENT_NATIVE_IOU = 0.999


def _fmt(value: float) -> str:
    return _impl._fmt(float(value))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pil_mask(width: int, height: int, draw_fn: Any) -> np.ndarray:
    image = Image.new("L", (int(width), int(height)), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.asarray(image, dtype=np.uint8) > 0


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def _external(mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        _impl._core._filled_external_region(np.asarray(mask, dtype=bool)),
        dtype=bool,
    )


def _rounded_model(mask: np.ndarray) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    outer = _external(source)
    box = _bbox(outer)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if min(width, height) < 8:
        return None
    best: tuple[float, int] | None = None
    for radius in range(1, max(1, min(width, height) // 2) + 1):
        candidate = _pil_mask(
            source.shape[1],
            source.shape[0],
            lambda d, r=radius: d.rounded_rectangle(
                (x0, y0, x1, y1), radius=r, fill=255
            ),
        )
        score = _iou(candidate, outer)
        if best is None or score > best[0]:
            best = (score, radius)
        if score == 1.0:
            break
    if best is None or best[0] < _PARENT_NATIVE_IOU:
        return None
    return {
        "kind": "rounded_rect",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "radius": int(best[1]),
    }


def _ellipse_model(mask: np.ndarray) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    outer = _external(source)
    box = _bbox(outer)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if min(x1 - x0 + 1, y1 - y0 + 1) < 5:
        return None
    model = {
        "kind": "ellipse",
        "cx": (x0 + x1) / 2.0,
        "cy": (y0 + y1) / 2.0,
        "rx": (x1 - x0) / 2.0,
        "ry": (y1 - y0) / 2.0,
    }
    candidate = model_mask(
        model,
        source.shape[1],
        source.shape[0],
        source.shape[1],
        source.shape[0],
    )
    return model if _iou(candidate, outer) >= _PARENT_NATIVE_IOU else None


def _candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    if str(model["kind"]) == "rounded_rect":
        x0, y0, x1, y1, radius = (
            float(model[key]) for key in ("x0", "y0", "x1", "y1", "radius")
        )
        width = x1 - x0 + 1.0
        height = y1 - y0 + 1.0
        specs = (
            (x0, y0, width, height, radius),
            (x0 - .01, y0 - .01, width + .02, height + .02, radius),
            (x0, y0, width, height, max(.5, radius - .5)),
            (x0, y0, width, height, radius + .5),
            (x0 - .125, y0 - .125, width + .25, height + .25, radius),
        )
        output = [[
            f'<rect x="{_fmt(px)}" y="{_fmt(py)}" width="{_fmt(pw)}" '
            f'height="{_fmt(ph)}" rx="{_fmt(pr)}" ry="{_fmt(pr)}" fill="{fill}"/>'
        ] for px, py, pw, ph, pr in specs]
        base_width = max(.001, x1 - x0)
        base_height = max(.001, y1 - y0)
        for dr in (0.0, -.5, .5):
            pr = max(.5, radius + dr)
            output.append([
                f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" width="{_fmt(base_width)}" '
                f'height="{_fmt(base_height)}" rx="{_fmt(pr)}" ry="{_fmt(pr)}" '
                f'fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'
            ])
        return output

    cx, cy, rx, ry = (
        float(model[key]) for key in ("cx", "cy", "rx", "ry")
    )
    return [
        [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1" vector-effect="non-scaling-stroke"/>'],
        [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="0.75" vector-effect="non-scaling-stroke"/>'],
        [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx)}" ry="{_fmt(ry)}" fill="{fill}" stroke="{fill}" stroke-width="1.25" vector-effect="non-scaling-stroke"/>'],
        [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(rx+.25)}" ry="{_fmt(ry+.25)}" fill="{fill}"/>'],
        [f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" rx="{_fmt(max(.5,rx-.25))}" ry="{_fmt(max(.5,ry-.25))}" fill="{fill}"/>'],
    ]


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def _metrics(
    model: dict[str, Any],
    elements: list[str],
    fill: str,
    source_width: int,
    source_height: int,
) -> dict[str, float] | None:
    colors = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    background = (
        f'<path fill="#ffffff" d="M0 0H{source_width}V{source_height}H0Z"/>'
    )
    trial = [background] + _blacken(elements, fill)
    ious: list[float] = []
    mismatches: list[float] = []
    component_error = 0
    for factor in FACTORS:
        target_width = max(16, int(round(source_width * factor)))
        target_height = max(16, int(round(source_height * factor)))
        rendered = _impl._render(
            source_width,
            source_height,
            trial,
            colors,
            target_width,
            target_height,
        )
        if rendered is None:
            return None
        actual = rendered == 0
        expected = model_mask(
            model,
            source_width,
            source_height,
            target_width,
            target_height,
        )
        intersection = int(np.count_nonzero(actual & expected))
        union = int(np.count_nonzero(actual | expected))
        ious.append(1.0 if union == 0 else intersection / float(union))
        mismatches.append(float(np.mean(actual != expected)))
        component_error += abs(int(_impl._cc(actual)) - 1)
    return {
        "component_error": float(component_error),
        "min_iou": float(min(ious)),
        "mean_iou": float(np.mean(ious)),
        "max_mismatch": float(max(mismatches)),
        "native_iou": float(ious[2]),
    }


def _score(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["component_error"],
        -metrics["min_iou"],
        metrics["max_mismatch"],
        -metrics["mean_iou"],
    )


def calibrate_parent_regions(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
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

    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        count = int(region.get("path_count") or 0)
        parts.append(list(elements[cursor:cursor + count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), []

    source_height, source_width = labels.shape
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        if len(parts[index]) != 1:
            continue
        node_id = int(region["node_id"])
        if child_count[node_id] <= 0:
            continue
        source_mask = node_map == node_id
        model = _rounded_model(source_mask)
        if model is None:
            model = _ellipse_model(source_mask)
        if model is None:
            continue
        color_index = int(region["color_index"])
        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        base = list(parts[index])
        before = _metrics(model, base, fill, source_width, source_height)
        if before is None:
            continue
        best = before
        best_part = base
        best_key = _score(before)
        for candidate in _candidates(model, fill):
            metrics = _metrics(model, candidate, fill, source_width, source_height)
            if metrics is None:
                continue
            if metrics["native_iou"] + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            key = _score(metrics)
            if key < best_key:
                best_key = key
                best = metrics
                best_part = candidate
        changed = best_part != base
        if changed:
            parts[index] = list(best_part)
        reports.append({
            "node_id": node_id,
            "color_index": color_index,
            "model": model,
            "changed": changed,
            "before": before,
            "after": best,
        })

    return [element for part in parts for element in part], reports


__all__ = ["calibrate_parent_regions"]
