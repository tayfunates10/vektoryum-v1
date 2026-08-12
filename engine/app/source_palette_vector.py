"""Source-palette vector candidate for opaque low-color artwork.

The production candidate preserves the exact source palette, but no longer
claims scale-stable eligibility merely because a native raster-cell trace is
pixel-perfect.  Pixel boundaries are first decomposed into semantic cycles.
Axis-aligned rectangles stay exact; curved/geometric cycles must be fitted to
resolution-independent primitives.  The fitted artifact is accepted only after
native fidelity/component checks and deterministic multi-render self-consistency.
If that transaction fails, the byte-exact raster-cell trace remains available as
an explicitly non-scale-stable fallback.

No quantization, bitmap embedding, fixture identifiers, palette-cap increase or
shared path/node/byte budget change occurs here.
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


def _boundary_cycles(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace deterministic clockwise pixel-cell boundary cycles for one mask."""
    h, w = mask.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
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

    def _direction(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
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
            incoming_index = direction_index[_direction(previous, current)]
            preferred = [
                (incoming_index + 1) % 4,
                incoming_index,
                (incoming_index - 1) % 4,
                (incoming_index + 2) % 4,
            ]
            candidates.sort(
                key=lambda index: preferred.index(
                    direction_index[_direction(edges[index][0], edges[index][1])]
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


def _cycle_to_pixel_d(points: list[tuple[int, int]]) -> tuple[str, int]:
    x0, y0 = points[0]
    commands = [f"M{x0} {y0}"]
    for x, y in points[1:]:
        commands.append(f"L{x} {y}")
    commands.append("Z")
    return "".join(commands), len(commands)


def _axis_aligned_rectangle(points: list[tuple[int, int]]) -> bool:
    if len(points) != 4:
        return False
    xs = sorted({int(p[0]) for p in points})
    ys = sorted({int(p[1]) for p in points})
    if len(xs) != 2 or len(ys) != 2:
        return False
    return set(points) == {
        (xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])
    }


def _ellipse_d_from_pixel_centers(points: list[tuple[int, int]]) -> str | None:
    """Infer a small circular/elliptic semantic primitive from a raster cycle."""
    arr = np.asarray(points, dtype=np.float64)
    x0, y0 = arr.min(axis=0)
    x1_cell, y1_cell = arr.max(axis=0)
    x1 = x1_cell - 1.0
    y1 = y1_cell - 1.0
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    if rx < 1.0 or ry < 1.0:
        return None
    aspect = max(rx, ry) / max(1e-9, min(rx, ry))
    if aspect > 1.35:
        return None

    area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
    cell_w = max(1.0, x1_cell - x0)
    cell_h = max(1.0, y1_cell - y0)
    fill_ratio = area / (cell_w * cell_h)
    if not (0.48 <= fill_ratio <= 0.90):
        return None

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    fmt = lambda value: f"{float(value):.4f}".rstrip("0").rstrip(".")
    return (
        f"M{fmt(cx-rx)} {fmt(cy)}"
        f"A{fmt(rx)} {fmt(ry)} 0 1 0 {fmt(cx+rx)} {fmt(cy)}"
        f"A{fmt(rx)} {fmt(ry)} 0 1 0 {fmt(cx-rx)} {fmt(cy)}Z"
    )


def _fit_scale_stable_cycle(
    points: list[tuple[int, int]],
) -> tuple[str, str]:
    """Return ``(d, strategy)`` or raise for an unfittable non-rect cycle."""
    if _axis_aligned_rectangle(points):
        d, _count = _cycle_to_pixel_d(points)
        return d, "axis_aligned_rectangle"

    arr = np.asarray(points, dtype=np.float64)
    span = arr.max(axis=0) - arr.min(axis=0)
    if float(np.hypot(*span)) >= 10.0:
        try:
            from app.shape_fitting import try_fit_whole_shape  # noqa: PLC0415

            fitted = try_fit_whole_shape(arr, True)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted:
            return fitted.replace(" ", ""), "whole_shape_fit"

    ellipse = _ellipse_d_from_pixel_centers(points)
    if ellipse:
        return ellipse, "small_ellipse_fit"

    raise SourcePaletteNotApplicable("nonrect_cycle_not_scale_stable")


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
    dist = np.sum(diff * diff, axis=3, dtype=np.int64)
    return np.argmin(dist, axis=2).astype(np.int16)


def _class_component_counts(labels: np.ndarray, color_count: int) -> list[int]:
    counts: list[int] = []
    for index in range(color_count):
        mask = (labels == index).astype(np.uint8)
        count, _ = cv2.connectedComponents(mask, connectivity=8)
        counts.append(max(0, int(count) - 1))
    return counts


def _validate_fitted_candidate(
    svg_path: Path,
    source_rgb: np.ndarray,
    colors: np.ndarray,
) -> dict[str, Any]:
    """Fail-closed native + multi-render transaction for fitted geometry."""
    try:
        from app.component_quality import measure_component_integrity_arrays  # noqa: PLC0415
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return {"passed": False, "reason": f"validation_import_error:{type(exc).__name__}"}

    h, w = source_rgb.shape[:2]
    native = render_svg_to_rgb(Path(svg_path), int(w), int(h))
    if native is None:
        return {"passed": False, "reason": "native_render_unavailable"}

    delta = np.max(
        np.abs(source_rgb.astype(np.int16) - native.astype(np.int16)),
        axis=2,
    )
    visible = float(np.mean(delta > 2))
    component = measure_component_integrity_arrays(source_rgb, native)
    official_pass = bool(
        component.get("status") == "pass"
        and float(component.get("source_cc_recall") or 0.0) == 1.0
        and float(component.get("render_cc_precision") or 0.0) == 1.0
        and float(component.get("min_true_cc_iou") or 0.0) >= _NATIVE_MIN_COMPONENT_IOU
    )
    if visible > _NATIVE_VISIBLE_RESIDUAL_MAX or not official_pass:
        return {
            "passed": False,
            "reason": "native_transaction_failed",
            "native_visible_residual": round(visible, 8),
            "native_component": component,
        }

    native_labels = _nearest_palette_labels(source_rgb, colors)
    native_counts = _class_component_counts(native_labels, len(colors))
    levels: list[dict[str, Any]] = []
    for factor in _SCALE_FACTORS:
        tw = max(16, int(round(w * factor)))
        th = max(16, int(round(h * factor)))
        rendered = render_svg_to_rgb(Path(svg_path), tw, th)
        if rendered is None:
            return {"passed": False, "reason": f"multirender_unavailable:{factor:g}x"}
        labels = _nearest_palette_labels(rendered, colors)
        present = [bool(np.any(labels == index)) for index in range(len(colors))]
        if not all(present):
            return {
                "passed": False,
                "reason": f"palette_class_missing:{factor:g}x",
                "present": present,
            }
        levels.append(
            {
                "scale": f"{factor:g}x",
                "size": [tw, th],
                "class_component_counts": _class_component_counts(labels, len(colors)),
            }
        )

    return {
        "passed": True,
        "reason": "native_and_multirender_transaction_pass",
        "native_visible_residual": round(visible, 8),
        "native_component": component,
        "native_class_component_counts": native_counts,
        "levels": levels,
    }


def vectorize_source_palette_paths(
    source_path: Path,
    output_path: Path,
    *,
    max_colors: int,
    max_commands: int = 5000,
) -> dict[str, Any]:
    """Serialize an opaque low-color source as genuine SVG path geometry."""
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
        raise SourcePaletteNotApplicable(
            f"source_palette_exceeds_cap:{color_count}>{int(max_colors)}"
        )
    palette = {tuple(int(channel) for channel in color) for color in colors.tolist()}
    if palette.issubset({(0, 0, 0), (255, 255, 255)}):
        raise SourcePaletteNotApplicable("canonical_binary_palette")

    labels = inverse.reshape(height, width)
    counts = np.bincount(inverse)
    order = sorted(
        range(color_count),
        key=lambda index: (
            -int(counts[index]),
            int(colors[index][0]),
            int(colors[index][1]),
            int(colors[index][2]),
        ),
    )

    exact_elements: list[str] = []
    fitted_elements: list[str] = []
    exact_command_count = 0
    fitted_command_count = 0
    cycle_count = 0
    strategy_counts: dict[str, int] = defaultdict(int)
    fit_error: str | None = None

    for color_index in order:
        cycles = _boundary_cycles(labels == color_index)
        if not cycles:
            continue
        exact_commands: list[str] = []
        fitted_commands: list[str] = []
        for points in cycles:
            cycle_count += 1
            exact_d, exact_nodes = _cycle_to_pixel_d(points)
            exact_commands.append(exact_d)
            exact_command_count += exact_nodes
            if exact_command_count > int(max_commands):
                raise SourcePaletteNotApplicable(
                    f"candidate_command_ceiling:{exact_command_count}>{int(max_commands)}"
                )
            if fit_error is None:
                try:
                    fitted_d, strategy = _fit_scale_stable_cycle(points)
                    fitted_commands.append(fitted_d)
                    strategy_counts[strategy] += 1
                    fitted_command_count += max(
                        2,
                        sum(fitted_d.count(token) for token in ("M", "L", "H", "V", "A", "C", "Z")),
                    )
                except SourcePaletteNotApplicable as exc:
                    fit_error = str(exc)

        red, green, blue = (int(value) for value in colors[color_index])
        fill = "#{:02x}{:02x}{:02x}".format(red, green, blue)
        exact_elements.append(
            f'<path fill="{fill}" fill-rule="evenodd" d="{"".join(exact_commands)}"/>'
        )
        if fit_error is None:
            fitted_elements.append(
                f'<path fill="{fill}" fill-rule="evenodd" d="{"".join(fitted_commands)}"/>'
            )

    if not exact_elements:
        raise SourcePaletteNotApplicable("no_vector_paths")

    exact_svg = _svg_document(width, height, exact_elements)
    fitted_report: dict[str, Any] = {
        "passed": False,
        "reason": fit_error or "not_measured",
    }
    scale_stable = False
    selected_svg = exact_svg
    selected_commands = exact_command_count
    geometry_strategy = "raster_cell_fallback"

    if fit_error is None and fitted_elements:
        fitted_svg = _svg_document(width, height, fitted_elements)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(fitted_svg, encoding="utf-8")
        fitted_report = _validate_fitted_candidate(output_path, rgb, colors)
        if fitted_report.get("passed"):
            scale_stable = True
            selected_svg = fitted_svg
            selected_commands = fitted_command_count
            geometry_strategy = "semantic_cycle_fit"
        else:
            output_path.write_text(exact_svg, encoding="utf-8")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(exact_svg, encoding="utf-8")

    if scale_stable:
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
        "fit_strategy_counts": dict(sorted(strategy_counts.items())),
        "fit_transaction": fitted_report,
    }


__all__ = ["SourcePaletteNotApplicable", "vectorize_source_palette_paths"]
