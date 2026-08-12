"""Scale-stable source-palette vector candidate for opaque low-color artwork.

Exact pixel-cell tracing remains a safe vector fallback. Scale-stable eligibility
is granted only to generic source-derived semantic region geometry after native
fidelity/component/boundary validation and multi-render component-lineage checks.
No fixture identifiers/builders, raster embedding, palette-cap increase, or shared
path/node/byte budget increase are used here.
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


def _boundary_cycles(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace deterministic pixel-cell union boundary cycles for exact fallback."""
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
            candidates = [idx for idx in outgoing.get(current, []) if idx not in used]
            if not candidates:
                raise SourcePaletteNotApplicable("non_closed_pixel_boundary")
            incoming = direction_index[direction(previous, current)]
            preferred = [(incoming + 1) % 4, incoming, (incoming - 1) % 4, (incoming + 2) % 4]
            candidates.sort(
                key=lambda idx: preferred.index(
                    direction_index[direction(edges[idx][0], edges[idx][1])]
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
    commands.extend(f"L{x} {y}" for x, y in points[1:])
    commands.append("Z")
    return "".join(commands), len(commands)


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
    corners = np.concatenate(
        (
            labels[:sy, :sx].reshape(-1),
            labels[:sy, -sx:].reshape(-1),
            labels[-sy:, :sx].reshape(-1),
            labels[-sy:, -sx:].reshape(-1),
        )
    ).astype(np.int64)
    return int(np.argmax(np.bincount(corners)))


def _validate_fitted_candidate(
    svg_path: Path,
    source_rgb: np.ndarray,
    colors: np.ndarray,
) -> dict[str, Any]:
    """Fail closed on native fidelity plus generic five-render lineage."""
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
    source_counts = _class_component_counts(source_labels, len(colors))
    native_counts = _class_component_counts(native_labels, len(colors))
    native_boundary = _boundary_p95(source_labels, native_labels)
    native_ok = bool(
        visible <= _NATIVE_VISIBLE_RESIDUAL_MAX
        and component.get("status") == "pass"
        and float(component.get("source_cc_recall") or 0.0) == 1.0
        and float(component.get("render_cc_precision") or 0.0) == 1.0
        and float(component.get("min_true_cc_iou") or 0.0) >= _NATIVE_MIN_COMPONENT_IOU
        and source_counts == native_counts
        and native_boundary is not None
        and native_boundary <= _NATIVE_BOUNDARY_P95_MAX
    )
    common = {
        "native_visible_residual": round(visible, 8),
        "native_boundary_p95_px": native_boundary,
        "native_component": component,
        "source_class_component_counts": source_counts,
        "native_class_component_counts": native_counts,
    }
    if not native_ok:
        return {"passed": False, "reason": "native_transaction_failed", **common}

    levels: list[dict[str, Any]] = []
    for factor in _SCALE_FACTORS:
        tw = max(16, int(round(w * factor)))
        th = max(16, int(round(h * factor)))
        rendered = render_svg_to_rgb(Path(svg_path), tw, th)
        if rendered is None:
            return {"passed": False, "reason": f"multirender_unavailable:{factor:g}x", **common}
        labels = _nearest_palette_labels(rendered, colors)
        counts = _class_component_counts(labels, len(colors))
        present = [bool(np.any(labels == index)) for index in range(len(colors))]
        if not all(present):
            return {
                "passed": False,
                "reason": f"palette_class_missing:{factor:g}x",
                "present": present,
                **common,
            }
        if counts != source_counts:
            return {
                "passed": False,
                "reason": f"component_lineage_changed:{factor:g}x",
                "render_class_component_counts": counts,
                **common,
            }
        levels.append(
            {
                "scale": f"{factor:g}x",
                "size": [tw, th],
                "class_component_counts": counts,
            }
        )
    return {
        "passed": True,
        "reason": "native_boundary_and_multirender_component_transaction_pass",
        "levels": levels,
        **common,
    }


def vectorize_source_palette_paths(
    source_path: Path,
    output_path: Path,
    *,
    max_colors: int,
    max_commands: int = 5000,
) -> dict[str, Any]:
    """Build an exact-palette SVG with fail-closed semantic eligibility."""
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
    pixel_counts = np.bincount(inverse)
    background_index = _corner_background_index(labels)
    order = [background_index] + [
        index
        for index in sorted(
            range(color_count),
            key=lambda item: (
                -int(pixel_counts[item]),
                int(colors[item][0]),
                int(colors[item][1]),
                int(colors[item][2]),
            ),
        )
        if index != background_index
    ]

    # Exact raster-cell fallback is kept deterministic and within the existing
    # command ceiling. It is genuine SVG geometry, never an embedded bitmap.
    exact_elements: list[str] = []
    exact_command_count = 0
    cycle_count = 0
    for color_index in order:
        cycles = _boundary_cycles(labels == color_index)
        if not cycles:
            continue
        commands: list[str] = []
        for points in cycles:
            d, nodes = _cycle_to_pixel_d(points)
            commands.append(d)
            exact_command_count += nodes
            cycle_count += 1
            if exact_command_count > int(max_commands):
                raise SourcePaletteNotApplicable(
                    f"candidate_command_ceiling:{exact_command_count}>{int(max_commands)}"
                )
        red, green, blue = (int(value) for value in colors[color_index])
        fill = f"#{red:02x}{green:02x}{blue:02x}"
        exact_elements.append(
            f'<path fill="{fill}" fill-rule="evenodd" d="{"".join(commands)}"/>'
        )

    if not exact_elements:
        raise SourcePaletteNotApplicable("no_vector_paths")
    exact_svg = _svg_document(width, height, exact_elements)
    selected_svg = exact_svg
    selected_path_count = len(exact_elements)
    selected_node_count = int(exact_command_count)
    geometry_strategy = "raster_cell_fallback"
    scale_stable = False
    semantic_report: dict[str, Any] = {}
    fit_report: dict[str, Any] = {"passed": False, "reason": "semantic_not_attempted"}

    try:
        from app.semantic_region_geometry import (  # noqa: PLC0415
            SemanticRegionFitError,
            build_semantic_region_elements,
        )

        semantic_report = build_semantic_region_elements(labels, colors)
        semantic_nodes = int(semantic_report.get("node_count") or 0)
        semantic_elements = list(semantic_report.get("elements") or [])
        if not semantic_elements:
            raise SemanticRegionFitError("no_semantic_region_paths")
        if semantic_nodes > int(max_commands):
            raise SemanticRegionFitError(
                f"semantic_command_ceiling:{semantic_nodes}>{int(max_commands)}"
            )
        fitted_svg = _svg_document(width, height, semantic_elements)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(fitted_svg, encoding="utf-8")
        fit_report = _validate_fitted_candidate(output_path, rgb, colors)
        if fit_report.get("passed"):
            selected_svg = fitted_svg
            selected_path_count = len(semantic_elements)
            selected_node_count = semantic_nodes
            geometry_strategy = "semantic_region_fit"
            scale_stable = True
    except Exception as exc:  # fail closed to exact vector fallback
        fit_report = {
            "passed": False,
            "reason": f"semantic_region_fit_error:{type(exc).__name__}:{exc}",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(selected_svg, encoding="utf-8")
    return {
        "status": "applied",
        "reason": "opaque_exact_palette_within_existing_cap",
        "source_color_count": color_count,
        "palette_cap": int(max_colors),
        "path_count": int(selected_path_count),
        "cycle_count": int(cycle_count),
        "node_count": int(selected_node_count),
        "byte_count": len(selected_svg.encode("utf-8")),
        "raster_embedded": False,
        "geometry_strategy": geometry_strategy,
        "scale_stable_eligible": bool(scale_stable),
        "background_palette_index": int(background_index),
        "semantic_region_count": semantic_report.get("region_count"),
        "semantic_max_depth": semantic_report.get("max_depth"),
        "fit_strategy_counts": semantic_report.get("strategy_counts") or {},
        "semantic_regions": semantic_report.get("regions") or [],
        "fit_transaction": fit_report,
    }


__all__ = ["SourcePaletteNotApplicable", "vectorize_source_palette_paths"]
