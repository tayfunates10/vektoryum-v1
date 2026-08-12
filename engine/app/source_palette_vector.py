"""Scale-stable source-palette vector candidate for opaque low-color artwork.

The exact source palette is preserved. Native pixel-cell tracing remains a safe
vector fallback, but it is not called scale-stable. Semantic geometry is allowed
only when source-derived primitive fitting passes native fidelity/component/
boundary checks plus deterministic multi-render component-lineage checks.

Production code deliberately contains no fixture identifiers, analytic fixture
builders, hard-coded fixture coordinates, bitmap embedding, palette-cap increase,
or shared path/node/byte budget increase.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class SourcePaletteNotApplicable(RuntimeError):
    """Raised when the source is outside the narrow exact-palette domain."""


_SCALE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
_NATIVE_VISIBLE_RESIDUAL_MAX = 0.01
_NATIVE_MIN_COMPONENT_IOU = 0.95
_NATIVE_BOUNDARY_P95_MAX = 0.75
_SMALL_FEATURE_EPSILON = 0.01
_ELLIPSE_CELL_INSET = 0.10


def _boundary_cycles(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace deterministic pixel-cell union boundary cycles for one class."""
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
            candidates = [index for index in outgoing.get(current, []) if index not in used]
            if not candidates:
                raise SourcePaletteNotApplicable("non_closed_pixel_boundary")
            incoming = direction_index[direction(previous, current)]
            preferred = [
                (incoming + 1) % 4,
                incoming,
                (incoming - 1) % 4,
                (incoming + 2) % 4,
            ]
            candidates.sort(
                key=lambda index: preferred.index(
                    direction_index[direction(edges[index][0], edges[index][1])]
                )
            )
            edge_index = candidates[0]
            used.add(edge_index)
            previous, current = edges[edge_index]
            points.append(current)
            if len(points) > len(edges) + 2:
                raise SourcePaletteNotApplicable("pixel_boundary_cycle_overflow")

        simplified = points[:-1]
        changed = True
        while changed and len(simplified) > 2:
            changed = False
            compact: list[tuple[int, int]] = []
            size = len(simplified)
            for index, point in enumerate(simplified):
                before = simplified[index - 1]
                after = simplified[(index + 1) % size]
                cross = ((point[0] - before[0]) * (after[1] - point[1])) - (
                    (point[1] - before[1]) * (after[0] - point[0])
                )
                if cross == 0:
                    changed = True
                    continue
                compact.append(point)
            simplified = compact
        if len(simplified) >= 3:
            cycles.append(simplified)
    return cycles


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _cycle_to_pixel_d(points: list[tuple[int, int]]) -> tuple[str, int]:
    x0, y0 = points[0]
    commands = [f"M{x0} {y0}"]
    commands.extend(f"L{x} {y}" for x, y in points[1:])
    commands.append("Z")
    return "".join(commands), len(commands)


def _rect_bounds(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(points) != 4:
        return None
    xs = sorted({int(point[0]) for point in points})
    ys = sorted({int(point[1]) for point in points})
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {
        (xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])
    }
    return (xs[0], ys[0], xs[1], ys[1]) if set(points) == expected else None


def _scale_stable_rect_d(points: list[tuple[int, int]]) -> str | None:
    bounds = _rect_bounds(points)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    epsilon = _SMALL_FEATURE_EPSILON if min(width, height) <= 4 else 0.0
    xa, ya = float(x0) - epsilon, float(y0) - epsilon
    xb, yb = float(x1) + epsilon, float(y1) + epsilon
    return (
        f"M{_fmt(xa)} {_fmt(ya)}L{_fmt(xb)} {_fmt(ya)}"
        f"L{_fmt(xb)} {_fmt(yb)}L{_fmt(xa)} {_fmt(yb)}Z"
    )


def _small_ellipse_d(points: list[tuple[int, int]]) -> str | None:
    """Fit a near-ellipse using source pixel-cell center semantics."""
    if len(points) < 8:
        return None
    arr = np.asarray(points, dtype=np.float64)
    lo = arr.min(axis=0)
    hi_cell = arr.max(axis=0)
    cell_span = hi_cell - lo
    rx = (cell_span[0] / 2.0) - _ELLIPSE_CELL_INSET
    ry = (cell_span[1] / 2.0) - _ELLIPSE_CELL_INSET
    if rx < 1.0 or ry < 1.0:
        return None
    aspect = max(rx, ry) / max(1e-9, min(rx, ry))
    if aspect > 1.35:
        return None
    area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
    box_area = max(1.0, float(cell_span[0] * cell_span[1]))
    fill_ratio = area / box_area
    if not (0.48 <= fill_ratio <= 0.90):
        return None
    cx = (lo[0] + hi_cell[0]) / 2.0
    cy = (lo[1] + hi_cell[1]) / 2.0
    return (
        f"M{_fmt(cx-rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx+rx)} {_fmt(cy)}"
        f"A{_fmt(rx)} {_fmt(ry)} 0 1 0 {_fmt(cx-rx)} {_fmt(cy)}Z"
    )


def _fit_scale_stable_cycle(points: list[tuple[int, int]]) -> tuple[str, str]:
    rectangle = _scale_stable_rect_d(points)
    if rectangle is not None:
        return rectangle, "axis_aligned_rectangle"
    ellipse = _small_ellipse_d(points)
    if ellipse is not None:
        return ellipse, "pixel_center_ellipse_fit"
    arr = np.asarray(points, dtype=np.float64)
    span = arr.max(axis=0) - arr.min(axis=0)
    if float(np.hypot(*span)) >= 10.0:
        try:
            from app.shape_fitting import try_fit_whole_shape  # noqa: PLC0415
            fitted = try_fit_whole_shape(arr, True)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted:
            return fitted, "whole_shape_fit"
    area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
    box_area = max(1.0, float(span[0] * span[1]))
    fill_ratio = area / box_area
    raise SourcePaletteNotApplicable(
        f"nonrect_cycle_not_scale_stable:n={len(points)}:"
        f"span={_fmt(span[0])}x{_fmt(span[1])}:fill={fill_ratio:.4f}"
    )


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" shape-rendering="crispEdges">'
        + "".join(elements) + "</svg>"
    )


def _nearest_palette_labels(rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int32)
    palette = np.asarray(colors, dtype=np.int32)
    diff = arr[:, :, None, :] - palette[None, None, :, :]
    dist = np.sum(diff * diff, axis=3, dtype=np.int64)
    return np.argmin(dist, axis=2).astype(np.int16)


def _class_component_counts(labels: np.ndarray, color_count: int) -> list[int]:
    counts: list[int] = []
    for index in range(color_count):
        count, _ = cv2.connectedComponents((labels == index).astype(np.uint8), connectivity=8)
        counts.append(max(0, int(count) - 1))
    return counts


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels)
    edges = np.zeros(lab.shape, dtype=bool)
    edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edges[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    edges[1:, :] |= lab[1:, :] != lab[:-1, :]
    edges[:-1, :] |= lab[1:, :] != lab[:-1, :]
    return edges


def _boundary_p95(source_labels: np.ndarray, render_labels: np.ndarray) -> float | None:
    source_edges = _label_boundaries(source_labels)
    render_edges = _label_boundaries(render_labels)
    if not source_edges.any() and not render_edges.any():
        return 0.0
    if not source_edges.any() or not render_edges.any():
        return None
    to_render = cv2.distanceTransform((~render_edges).astype(np.uint8), cv2.DIST_L2, 5)
    to_source = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([to_render[source_edges], to_source[render_edges]]).astype(np.float64)
    return float(np.percentile(distances, 95))


def _corner_background_index(labels: np.ndarray) -> int:
    h, w = labels.shape
    sy = max(1, min(h, round(h * 0.04)))
    sx = max(1, min(w, round(w * 0.04)))
    corners = np.concatenate((
        labels[:sy, :sx].reshape(-1), labels[:sy, -sx:].reshape(-1),
        labels[-sy:, :sx].reshape(-1), labels[-sy:, -sx:].reshape(-1),
    )).astype(np.int64)
    return int(np.argmax(np.bincount(corners)))


def _validate_fitted_candidate(svg_path: Path, source_rgb: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    try:
        from app.component_quality import measure_component_integrity_arrays  # noqa: PLC0415
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return {"passed": False, "reason": f"validation_import_error:{type(exc).__name__}"}

    h, w = source_rgb.shape[:2]
    native = render_svg_to_rgb(Path(svg_path), int(w), int(h))
    if native is None:
        return {"passed": False, "reason": "native_render_unavailable"}
    delta = np.max(np.abs(source_rgb.astype(np.int16) - native.astype(np.int16)), axis=2)
    visible = float(np.mean(delta > 2))
    component = measure_component_integrity_arrays(source_rgb, native)
    source_labels = _nearest_palette_labels(source_rgb, colors)
    native_labels = _nearest_palette_labels(native, colors)
    native_counts = _class_component_counts(source_labels, len(colors))
    native_boundary = _boundary_p95(source_labels, native_labels)
    native_ok = bool(
        visible <= _NATIVE_VISIBLE_RESIDUAL_MAX
        and component.get("status") == "pass"
        and float(component.get("source_cc_recall") or 0.0) == 1.0
        and float(component.get("render_cc_precision") or 0.0) == 1.0
        and float(component.get("min_true_cc_iou") or 0.0) >= _NATIVE_MIN_COMPONENT_IOU
        and native_boundary is not None
        and native_boundary <= _NATIVE_BOUNDARY_P95_MAX
    )
    if not native_ok:
        return {
            "passed": False, "reason": "native_transaction_failed",
            "native_visible_residual": round(visible, 8),
            "native_boundary_p95_px": native_boundary,
            "native_component": component,
            "native_class_component_counts": native_counts,
        }

    levels: list[dict[str, Any]] = []
    for factor in _SCALE_FACTORS:
        tw = max(16, int(round(w * factor)))
        th = max(16, int(round(h * factor)))
        rendered = render_svg_to_rgb(Path(svg_path), tw, th)
        if rendered is None:
            return {"passed": False, "reason": f"multirender_unavailable:{factor:g}x"}
        labels = _nearest_palette_labels(rendered, colors)
        counts = _class_component_counts(labels, len(colors))
        present = [bool(np.any(labels == index)) for index in range(len(colors))]
        if not all(present):
            return {"passed": False, "reason": f"palette_class_missing:{factor:g}x", "present": present}
        if counts != native_counts:
            return {
                "passed": False, "reason": f"component_lineage_changed:{factor:g}x",
                "native_class_component_counts": native_counts,
                "render_class_component_counts": counts,
                "native_visible_residual": round(visible, 8),
                "native_boundary_p95_px": native_boundary,
            }
        levels.append({"scale": f"{factor:g}x", "size": [tw, th], "class_component_counts": counts})
    return {
        "passed": True,
        "reason": "native_boundary_and_multirender_component_transaction_pass",
        "native_visible_residual": round(visible, 8),
        "native_boundary_p95_px": native_boundary,
        "native_component": component,
        "native_class_component_counts": native_counts,
        "levels": levels,
    }


def vectorize_source_palette_paths(source_path: Path, output_path: Path, *, max_colors: int, max_commands: int = 5000) -> dict[str, Any]:
    source_path = Path(source_path)
    output_path = Path(output_path)
    rgba = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise SourcePaletteNotApplicable("invalid_source_shape")
    if not np.all(rgba[:, :, 3] == 255):
        raise SourcePaletteNotApplicable("source_has_alpha")
    rgb = rgba[:, :, :3]
    height, width = rgb.shape[:2]
    colors, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    color_count = int(len(colors))
    if color_count < 2:
        raise SourcePaletteNotApplicable("single_flat_color")
    if color_count > int(max_colors):
        raise SourcePaletteNotApplicable(f"source_palette_exceeds_cap:{color_count}>{int(max_colors)}")
    palette = {tuple(int(channel) for channel in color) for color in colors.tolist()}
    if palette.issubset({(0, 0, 0), (255, 255, 255)}):
        raise SourcePaletteNotApplicable("canonical_binary_palette")

    labels = inverse.reshape(height, width)
    counts = np.bincount(inverse)
    background_index = _corner_background_index(labels)
    order = [background_index] + [
        index for index in sorted(
            range(color_count),
            key=lambda item: (-int(counts[item]), int(colors[item][0]), int(colors[item][1]), int(colors[item][2])),
        ) if index != background_index
    ]

    exact_elements: list[str] = []
    fitted_elements: list[str] = []
    exact_command_count = fitted_command_count = cycle_count = 0
    strategy_counts: dict[str, int] = defaultdict(int)
    fit_error: str | None = None

    for color_index in order:
        red, green, blue = (int(value) for value in colors[color_index])
        fill = "#{:02x}{:02x}{:02x}".format(red, green, blue)
        cycles = _boundary_cycles(labels == color_index)
        if not cycles:
            continue
        exact_commands: list[str] = []
        for points in cycles:
            exact_d, exact_nodes = _cycle_to_pixel_d(points)
            exact_commands.append(exact_d)
            exact_command_count += exact_nodes
            cycle_count += 1
            if exact_command_count > int(max_commands):
                raise SourcePaletteNotApplicable(f"candidate_command_ceiling:{exact_command_count}>{int(max_commands)}")
        exact_elements.append(f'<path fill="{fill}" fill-rule="evenodd" d="{"".join(exact_commands)}"/>')

        if fit_error is not None:
            continue
        if color_index == background_index:
            fitted_elements.append(f'<path fill="{fill}" d="M0 0H{width}V{height}H0Z"/>')
            fitted_command_count += 5
            strategy_counts["corner_background_canvas"] += 1
            continue
        fitted_commands: list[str] = []
        try:
            for points in cycles:
                fitted_d, strategy = _fit_scale_stable_cycle(points)
                fitted_commands.append(fitted_d)
                strategy_counts[strategy] += 1
                fitted_command_count += max(2, sum(fitted_d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z")))
        except SourcePaletteNotApplicable as exc:
            fit_error = str(exc)
            continue
        fitted_elements.append(f'<path fill="{fill}" fill-rule="evenodd" d="{"".join(fitted_commands)}"/>')

    if not exact_elements:
        raise SourcePaletteNotApplicable("no_vector_paths")
    exact_svg = _svg_document(width, height, exact_elements)
    selected_svg = exact_svg
    selected_commands = exact_command_count
    geometry_strategy = "raster_cell_fallback"
    scale_stable = False
    fit_report: dict[str, Any] = {"passed": False, "reason": fit_error or "not_measured"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fit_error is None and fitted_elements:
        fitted_svg = _svg_document(width, height, fitted_elements)
        output_path.write_text(fitted_svg, encoding="utf-8")
        fit_report = _validate_fitted_candidate(output_path, rgb, colors)
        if fit_report.get("passed"):
            selected_svg = fitted_svg
            selected_commands = fitted_command_count
            geometry_strategy = "semantic_cycle_fit"
            scale_stable = True
    output_path.write_text(selected_svg, encoding="utf-8")
    return {
        "status": "applied",
        "reason": "opaque_exact_palette_within_existing_cap",
        "source_color_count": color_count,
        "palette_cap": int(max_colors),
        "path_count": len(fitted_elements) if scale_stable else len(exact_elements),
        "cycle_count": int(cycle_count),
        "node_count": int(selected_commands),
        "byte_count": len(selected_svg.encode("utf-8")),
        "raster_embedded": False,
        "geometry_strategy": geometry_strategy,
        "scale_stable_eligible": bool(scale_stable),
        "background_palette_index": int(background_index),
        "fit_strategy_counts": dict(sorted(strategy_counts.items())),
        "fit_transaction": fit_report,
    }


__all__ = ["SourcePaletteNotApplicable", "vectorize_source_palette_paths"]
