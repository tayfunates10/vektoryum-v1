"""Exact source-palette vector candidate for opaque low-color artwork.

The candidate is intentionally narrow: it is applicable only when the source is
fully opaque and its exact RGB palette already fits the caller's existing
palette cap. No quantization, bitmap embedding, threshold relaxation, or
palette-cap increase occurs. Each source color is serialized as deterministic
closed SVG path geometry along the source pixel-cell boundaries.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SourcePaletteNotApplicable(RuntimeError):
    """Raised when the source is outside the narrow exact-palette domain."""


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


def vectorize_source_palette_paths(
    source_path: Path,
    output_path: Path,
    *,
    max_colors: int,
    max_commands: int = 5000,
) -> dict[str, Any]:
    """Serialize an exact opaque low-color source as genuine SVG path geometry.

    ``max_colors`` must come from the existing production palette policy. The
    function never raises it. ``max_commands`` is a candidate-local applicability
    ceiling: exceeding it rejects this candidate instead of changing any shared
    byte/path/node budget.
    """
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

    path_elements: list[str] = []
    command_count = 0
    cycle_count = 0
    for color_index in order:
        cycles = _boundary_cycles(labels == color_index)
        if not cycles:
            continue
        commands: list[str] = []
        for points in cycles:
            cycle_count += 1
            x0, y0 = points[0]
            commands.append(f"M{x0} {y0}")
            command_count += 1
            for x, y in points[1:]:
                commands.append(f"L{x} {y}")
                command_count += 1
            commands.append("Z")
            command_count += 1
            if command_count > int(max_commands):
                raise SourcePaletteNotApplicable(
                    f"candidate_command_ceiling:{command_count}>{int(max_commands)}"
                )
        red, green, blue = (int(value) for value in colors[color_index])
        path_elements.append(
            '<path fill="#{:02x}{:02x}{:02x}" fill-rule="evenodd" d="{}"/>'.format(
                red, green, blue, "".join(commands)
            )
        )

    if not path_elements:
        raise SourcePaletteNotApplicable("no_vector_paths")

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" shape-rendering="crispEdges">'
        + "".join(path_elements)
        + "</svg>"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return {
        "status": "applied",
        "reason": "opaque_exact_palette_within_existing_cap",
        "source_color_count": color_count,
        "palette_cap": int(max_colors),
        "path_count": len(path_elements),
        "cycle_count": int(cycle_count),
        "node_count": int(command_count),
        "byte_count": len(svg.encode("utf-8")),
        "raster_embedded": False,
    }


__all__ = ["SourcePaletteNotApplicable", "vectorize_source_palette_paths"]
