"""Source-derived five-scale phase refinement for exact Pillow primitives.

Tiny integer-grid primitives can match at 1x yet drift when raster-cell centres
or Pillow-inclusive endpoints are scaled.  This pass recognizes only generic
geometry from connected source masks, evaluates a small deterministic vector
candidate set with the production renderer at 0.25x/0.5x/1x/2x/4x, and keeps
an edit only when native fidelity remains healthy and component lineage does
not regress.  It contains no fixture IDs, regression builders, policy changes,
threshold changes, budget changes, or raster payloads.  Lines are out of scope.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app import semantic_region_geometry_impl as _impl
from app.scale_stable_primitive_geometry import FACTORS


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


def _scaled(value: float, target: int, source: int) -> int:
    return int(round(float(value) * float(target) / float(source)))


def _scaled_width(value: float, target: int, source: int) -> int:
    return max(1, _scaled(value, target, source))


def _external(mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        _impl._core._filled_external_region(np.asarray(mask, dtype=bool)),
        dtype=bool,
    )


def _exact_leaf_ellipse(mask: np.ndarray) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    box = _bbox(source)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if min(x1 - x0 + 1, y1 - y0 + 1) < 3:
        return None
    expected = _pil_mask(
        source.shape[1], source.shape[0],
        lambda draw: draw.ellipse((x0, y0, x1, y1), fill=255),
    )
    if not np.array_equal(expected, source):
        return None
    return {
        "kind": "center_radius_ellipse",
        "cx": (x0 + x1) / 2.0,
        "cy": (y0 + y1) / 2.0,
        "rx": (x1 - x0) / 2.0,
        "ry": (y1 - y0) / 2.0,
    }


def _exact_compact_rect(mask: np.ndarray) -> dict[str, Any] | None:
    source = np.asarray(mask, dtype=bool)
    box = _bbox(source)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if max(width, height) > 8:
        return None
    if int(np.count_nonzero(source)) != width * height:
        return None
    if not bool(np.all(source[y0:y1 + 1, x0:x1 + 1])):
        return None
    return {
        "kind": "origin_size_rect",
        "x": float(x0), "y": float(y0),
        "width": float(width), "height": float(height),
    }


def _endpoint_parent_ellipse(mask: np.ndarray) -> dict[str, Any] | None:
    outer = _external(mask)
    box = _bbox(outer)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if min(x1 - x0 + 1, y1 - y0 + 1) < 5:
        return None
    expected = _pil_mask(
        outer.shape[1], outer.shape[0],
        lambda draw: draw.ellipse((x0, y0, x1, y1), fill=255),
    )
    if not np.array_equal(expected, outer):
        return None
    return {
        "kind": "endpoint_ellipse",
        "x0": float(x0), "y0": float(y0),
        "x1": float(x1), "y1": float(y1),
    }


def _endpoint_parent_rounded(mask: np.ndarray) -> dict[str, Any] | None:
    outer = _external(mask)
    box = _bbox(outer)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    if min(width, height) < 8:
        return None
    for radius in range(1, max(1, min(width, height) // 2) + 1):
        expected = _pil_mask(
            outer.shape[1], outer.shape[0],
            lambda draw, r=radius: draw.rounded_rectangle(
                (x0, y0, x1, y1), radius=r, fill=255
            ),
        )
        if np.array_equal(expected, outer):
            return {
                "kind": "endpoint_rounded_rect",
                "x0": float(x0), "y0": float(y0),
                "x1": float(x1), "y1": float(y1),
                "radius": float(radius),
            }
    return None


def _model_mask(
    model: dict[str, Any], sw: int, sh: int, tw: int, th: int
) -> np.ndarray:
    kind = str(model["kind"])
    if kind == "center_radius_ellipse":
        cx = _scaled(model["cx"], tw, sw)
        cy = _scaled(model["cy"], th, sh)
        rx = _scaled_width(model["rx"], tw, sw)
        ry = _scaled_width(model["ry"], th, sh)
        return _pil_mask(
            tw, th,
            lambda draw: draw.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry), fill=255
            ),
        )
    if kind == "origin_size_rect":
        x = _scaled(model["x"], tw, sw)
        y = _scaled(model["y"], th, sh)
        width = _scaled_width(model["width"], tw, sw)
        height = _scaled_width(model["height"], th, sh)
        return _pil_mask(
            tw, th,
            lambda draw: draw.rectangle(
                (x, y, x + width - 1, y + height - 1), fill=255
            ),
        )
    if kind == "endpoint_ellipse":
        x0 = _scaled(model["x0"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        x1 = _scaled(model["x1"], tw, sw)
        y1 = _scaled(model["y1"], th, sh)
        return _pil_mask(
            tw, th, lambda draw: draw.ellipse((x0, y0, x1, y1), fill=255)
        )
    if kind == "endpoint_rounded_rect":
        x0 = _scaled(model["x0"], tw, sw)
        y0 = _scaled(model["y0"], th, sh)
        x1 = _scaled(model["x1"], tw, sw)
        y1 = _scaled(model["y1"], th, sh)
        radius = _scaled_width(model["radius"], min(tw, th), min(sw, sh))
        return _pil_mask(
            tw, th,
            lambda draw: draw.rounded_rectangle(
                (x0, y0, x1, y1), radius=radius, fill=255
            ),
        )
    raise ValueError(f"unsupported source-phase model: {kind}")


def _ellipse_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    if str(model["kind"]) == "center_radius_ellipse":
        cx = float(model["cx"]); cy = float(model["cy"])
        rx = float(model["rx"]); ry = float(model["ry"])
    else:
        x0 = float(model["x0"]); y0 = float(model["y0"])
        x1 = float(model["x1"]); y1 = float(model["y1"])
        cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
        rx = (x1 - x0) / 2.0; ry = (y1 - y0) / 2.0

    output: list[list[str]] = []
    for delta in (-0.0625, -0.01, 0.0, 0.01, 0.0625):
        output.append([
            f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" '
            f'rx="{_fmt(max(.125, rx + delta))}" '
            f'ry="{_fmt(max(.125, ry + delta))}" fill="{fill}"/>'
        ])
    for radius_delta, stroke_width in ((0.0, 1.0), (-0.5, 1.0), (-0.25, 0.5)):
        output.append([
            f'<ellipse cx="{_fmt(cx)}" cy="{_fmt(cy)}" '
            f'rx="{_fmt(max(.125, rx + radius_delta))}" '
            f'ry="{_fmt(max(.125, ry + radius_delta))}" fill="{fill}" '
            f'stroke="{fill}" stroke-width="{_fmt(stroke_width)}" '
            'vector-effect="non-scaling-stroke"/>'
        ])
    return output


def _rect_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x = float(model["x"]); y = float(model["y"])
    width = float(model["width"]); height = float(model["height"])
    specs = (
        (0.0, 0.0),
        (-0.01, 0.02),
        (0.01, 0.0),
        (-0.0625, 0.125),
        (0.0625, -0.125),
        (-0.125, 0.25),
        (0.125, -0.25),
    )
    output = [[
        f'<rect x="{_fmt(x + shift)}" y="{_fmt(y + shift)}" '
        f'width="{_fmt(max(.001, width + delta))}" '
        f'height="{_fmt(max(.001, height + delta))}" fill="{fill}"/>'
    ] for shift, delta in specs]
    for shift in (0.0, 0.5):
        output.append([
            f'<rect x="{_fmt(x + shift)}" y="{_fmt(y + shift)}" '
            f'width="{_fmt(max(.001, width - 1.0))}" '
            f'height="{_fmt(max(.001, height - 1.0))}" fill="{fill}" '
            f'stroke="{fill}" stroke-width="1" '
            'vector-effect="non-scaling-stroke"/>'
        ])
    return output


def _rounded_candidates(model: dict[str, Any], fill: str) -> list[list[str]]:
    x0 = float(model["x0"]); y0 = float(model["y0"])
    x1 = float(model["x1"]); y1 = float(model["y1"])
    radius = float(model["radius"])
    output: list[list[str]] = []
    for radius_delta in (-0.5, 0.0, 0.5):
        rr = max(0.5, radius + radius_delta)
        output.append([
            f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" '
            f'width="{_fmt(max(.001, x1 - x0))}" '
            f'height="{_fmt(max(.001, y1 - y0))}" '
            f'rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="{fill}" '
            f'stroke="{fill}" stroke-width="1" '
            'vector-effect="non-scaling-stroke"/>'
        ])
        output.append([
            f'<rect x="{_fmt(x0)}" y="{_fmt(y0)}" '
            f'width="{_fmt(x1 - x0 + 1.0)}" '
            f'height="{_fmt(y1 - y0 + 1.0)}" '
            f'rx="{_fmt(rr)}" ry="{_fmt(rr)}" fill="{fill}"/>'
        ])
    return output


def _blacken(elements: list[str], fill: str) -> list[str]:
    return [element.replace(fill, "#000000") for element in elements]


def _isolated_metrics(
    model: dict[str, Any], elements: list[str], fill: str, sw: int, sh: int
) -> dict[str, Any] | None:
    colors = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    trial = [f'<path fill="#ffffff" d="M0 0H{sw}V{sh}H0Z"/>'] + _blacken(
        elements, fill
    )
    levels: list[dict[str, Any]] = []
    for factor in FACTORS:
        tw = max(16, int(round(sw * factor)))
        th = max(16, int(round(sh * factor)))
        rendered = _impl._render(sw, sh, trial, colors, tw, th)
        if rendered is None:
            return None
        actual = rendered == 0
        expected = _model_mask(model, sw, sh, tw, th)
        intersection = int(np.count_nonzero(actual & expected))
        union = int(np.count_nonzero(actual | expected))
        levels.append({
            "scale": f"{factor:g}x",
            "iou": 1.0 if union == 0 else intersection / float(union),
            "mismatch": float(np.mean(actual != expected)),
            "components": int(_impl._cc(actual)),
        })
    return {
        "levels": levels,
        "component_error": int(sum(abs(int(level["components"]) - 1) for level in levels)),
        "min_iou": float(min(float(level["iou"]) for level in levels)),
        "mean_iou": float(np.mean([float(level["iou"]) for level in levels])),
        "max_mismatch": float(max(float(level["mismatch"]) for level in levels)),
        "native_iou": float(levels[2]["iou"]),
    }


def _isolated_score(metrics: dict[str, Any], count: int) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["component_error"]),
        -float(metrics["min_iou"]),
        float(metrics["max_mismatch"]),
        -float(metrics["mean_iou"]),
        int(count),
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
    lineage_errors: list[int] = []
    native_iou: float | None = None
    native_mismatch: float | None = None
    for factor in FACTORS:
        tw = max(16, int(round(sw * factor)))
        th = max(16, int(round(sh * factor)))
        rendered = _impl._render(sw, sh, elements, colors, tw, th)
        if rendered is None:
            return None
        counts = [int(value) for value in _impl._counts(rendered, len(colors))]
        lineage_errors.append(int(sum(
            abs(current - int(expected))
            for current, expected in zip(counts, source_counts, strict=True)
        )))
        if factor == 1.0:
            native_mismatch = float(np.mean(rendered != labels))
            native_iou = float(_impl._best_iou(
                rendered, node_map == int(node_id), int(color_index)
            ))
    return {
        "lineage_errors": lineage_errors,
        "native_iou": float(native_iou or 0.0),
        "native_mismatch": float(native_mismatch or 0.0),
    }


def _scene_safe(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if float(after["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
        return False
    if float(after["native_mismatch"]) > float(before["native_mismatch"]) + 1e-12:
        return False
    return not any(
        int(current) > int(previous)
        for current, previous in zip(
            after["lineage_errors"], before["lineage_errors"], strict=True
        )
    )


def _flatten(parts: list[list[str]]) -> list[str]:
    return [element for part in parts for element in part]


def refine_source_phase_geometry(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
    regions: list[dict[str, Any]],
    primitive_report: list[dict[str, Any]],
    prior_refinement_report: list[dict[str, Any]],
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

    delta_by_node: dict[int, int] = {}
    for collection in (primitive_report, prior_refinement_report):
        for item in collection:
            if not isinstance(item, dict) or "node_id" not in item:
                continue
            node_id = int(item["node_id"])
            delta_by_node[node_id] = delta_by_node.get(node_id, 0) + int(
                item.get("element_delta") or 0
            )

    parts: list[list[str]] = []
    cursor = 0
    for region in regions:
        node_id = int(region.get("node_id", -1))
        count = int(region.get("path_count") or 0) + delta_by_node.get(node_id, 0)
        parts.append(list(elements[cursor:cursor + count]))
        cursor += count
    if cursor != len(elements):
        return list(elements), []

    sh, sw = labels.shape
    reports: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        node_id = int(region.get("node_id", -1))
        if node_id < 0 or node_id >= len(nodes) or not parts[index]:
            continue
        source_mask = node_map == node_id
        model: dict[str, Any] | None = None
        family: str | None = None
        if child_count[node_id] == 0:
            model = _exact_leaf_ellipse(source_mask)
            if model is not None:
                family = "leaf_center_radius_ellipse"
            else:
                model = _exact_compact_rect(source_mask)
                if model is not None:
                    family = "leaf_origin_size_rect"
        elif len(parts[index]) == 1:
            model = _endpoint_parent_rounded(source_mask)
            if model is not None:
                family = "parent_endpoint_rounded_rect"
            else:
                model = _endpoint_parent_ellipse(source_mask)
                if model is not None:
                    family = "parent_endpoint_ellipse"
        if model is None or family is None:
            continue

        color_index = int(region["color_index"])
        rgb = colors[color_index]
        fill = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        base_part = list(parts[index])
        isolated_before = _isolated_metrics(model, base_part, fill, sw, sh)
        scene_before = _scene_metrics(
            labels, colors, _flatten(parts), node_map, node_id,
            color_index, source_counts,
        )
        if isolated_before is None or scene_before is None:
            continue
        if (
            int(isolated_before["component_error"]) == 0
            and float(isolated_before["min_iou"]) >= 0.995
        ):
            continue

        kind = str(model["kind"])
        if kind in {"center_radius_ellipse", "endpoint_ellipse"}:
            candidates = _ellipse_candidates(model, fill)
        elif kind == "origin_size_rect":
            candidates = _rect_candidates(model, fill)
        else:
            candidates = _rounded_candidates(model, fill)

        base_key = _isolated_score(isolated_before, len(base_part))
        ranked: list[tuple[tuple[float, float, float, float, int], list[str], dict[str, Any]]] = []
        for candidate in candidates:
            isolated = _isolated_metrics(model, candidate, fill, sw, sh)
            if isolated is None:
                continue
            if float(isolated["native_iou"]) + 1e-12 < float(_impl._NATIVE_MIN_IOU):
                continue
            key = _isolated_score(isolated, len(candidate))
            if key < base_key:
                ranked.append((key, list(candidate), isolated))
        ranked.sort(key=lambda item: item[0])

        best_part = base_part
        best_isolated = isolated_before
        best_scene = scene_before
        for _key, candidate, isolated in ranked:
            trial_parts = list(parts)
            trial_parts[index] = candidate
            scene = _scene_metrics(
                labels, colors, _flatten(trial_parts), node_map, node_id,
                color_index, source_counts,
            )
            if scene is not None and _scene_safe(scene_before, scene):
                best_part = candidate
                best_isolated = isolated
                best_scene = scene
                break

        changed = best_part != base_part
        if changed:
            parts[index] = best_part
        reports.append({
            "node_id": node_id,
            "color_index": color_index,
            "family": family,
            "model": model,
            "changed": changed,
            "element_delta": len(best_part) - len(base_part),
            "before": {"isolated": isolated_before, "scene": scene_before},
            "after": {"isolated": best_isolated, "scene": best_scene},
        })

    return _flatten(parts), reports


__all__ = ["refine_source_phase_geometry"]
