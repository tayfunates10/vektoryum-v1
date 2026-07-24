"""Vector-only paint-deficit reconstruction for source-alpha painter fallback.

This candidate is attempted only after the existing painter encodings fail.
It measures source-foreground pixels that the preserved artwork renders white,
overlays a compact source-derived vector surface above that artwork, and applies
the source alpha once through a fixed 24-level polygon mask.  Acceptance remains
owned by the unchanged alpha evaluator and TransformJournal gates.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from typing import Any

import cv2
import numpy as np

from app.alpha_artwork_identity import (
    ROLE_ARTWORK_CONTAINER,
    ROLE_MASK_APPLICATION,
    ROLE_MASK_DEFINITION,
    ROLE_MASK_GEOMETRY,
    tag_transform_node,
)

_SVG_NS = "http://www.w3.org/2000/svg"
_ALPHA_LEVELS = 24
_PALETTE_LIMIT = 8


def _composite_on_white(rgba: np.ndarray) -> np.ndarray:
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        array[:, :, :3].astype(np.float32) * alpha
        + 255.0 * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def _fixed_alpha_levels(
    alpha: np.ndarray,
    levels: int = _ALPHA_LEVELS,
) -> tuple[np.ndarray, dict[int, float]]:
    """Round source alpha onto a fixed deterministic level lattice."""
    steps = max(1, int(levels) - 1)
    quantized = np.rint(
        np.asarray(alpha, dtype=np.float32) * steps / 255.0
    ).astype(np.uint8)
    opacity = {
        int(value): int(value) / float(steps)
        for value in np.unique(quantized)
        if int(value) > 0
    }
    return quantized, opacity


def _dominant_opaque_palette(
    source_rgba: np.ndarray,
    limit: int = _PALETTE_LIMIT,
) -> np.ndarray:
    """Stable source palette: 8-step buckets, count-descending, mean colour."""
    rgba = np.asarray(source_rgba, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    selected = alpha >= 240
    if not bool(selected.any()):
        selected = alpha > 0
    colors = rgba[:, :, :3][selected].astype(np.int32)
    if not len(colors):
        return np.asarray([[0, 0, 0]], dtype=np.uint8)
    buckets = colors // 8
    unique, inverse, counts = np.unique(
        buckets,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = sorted(
        range(len(unique)),
        key=lambda index: (
            -int(counts[index]),
            tuple(int(value) for value in unique[index]),
        ),
    )[: max(1, int(limit))]
    palette = []
    for index in order:
        members = colors[inverse == index]
        palette.append(
            np.rint(members.mean(axis=0))
            .clip(0, 255)
            .astype(np.uint8)
        )
    return np.asarray(palette, dtype=np.uint8)


def _anchored_source_component_mask(
    source_alpha: np.ndarray,
    artwork_alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Select only source-alpha components grounded in preserved artwork.

    Missing paint may be completed anywhere inside a source component only when
    that same deterministic 8-connected component overlaps existing artwork by at
    least one pixel. Fully detached source components remain excluded, preventing
    the fallback from synthesising unrelated objects. The unchanged alpha and
    TransformJournal gates remain the final acceptance authority.
    """
    source_positive = (
        np.asarray(source_alpha, dtype=np.uint8) > 0
    ).astype(np.uint8)
    artwork_occupied = np.asarray(artwork_alpha, dtype=np.uint8) > 0
    component_count, component_labels = cv2.connectedComponents(
        source_positive,
        connectivity=8,
    )
    anchored = np.zeros(source_positive.shape, dtype=bool)
    anchored_count = 0
    for component_id in range(1, int(component_count)):
        component = component_labels == component_id
        if bool(np.any(component & artwork_occupied)):
            anchored |= component
            anchored_count += 1
    total = max(0, int(component_count) - 1)
    return anchored, {
        "source_component_count": total,
        "anchored_source_component_count": int(anchored_count),
        "detached_source_component_count": int(total - anchored_count),
    }


def _paint_deficit_labels(
    source_rgba: np.ndarray,
    artwork_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    source = np.asarray(source_rgba, dtype=np.uint8)
    artwork = np.asarray(artwork_rgba, dtype=np.uint8)
    source_white = _composite_on_white(source)
    artwork_white = _composite_on_white(artwork)
    source_foreground = np.any(
        np.abs(source_white.astype(np.int16) - 255) > 12,
        axis=2,
    )
    artwork_missing = np.all(artwork_white > 244, axis=2)
    anchored_components, component_stats = _anchored_source_component_mask(
        source[:, :, 3], artwork[:, :, 3]
    )
    deficit = source_foreground & artwork_missing & anchored_components
    if not bool(deficit.any()):
        return (
            np.zeros(deficit.shape, dtype=np.int32),
            np.asarray([[0, 0, 0]], dtype=np.uint8),
            {
                "paint_deficit_pixel_count": 0,
                "paint_deficit_opaque_artwork_count": 0,
                "paint_deficit_alpha_residual_overlap": 0,
                **component_stats,
            },
        )

    palette = _dominant_opaque_palette(source)
    source_rgb = source[:, :, :3].astype(np.int32)
    distances = (
        source_rgb[:, :, None, :]
        - palette.astype(np.int32)[None, None, :, :]
    ) ** 2
    nearest = np.argmin(distances.sum(axis=3), axis=2).astype(np.int32)
    labels = np.zeros(deficit.shape, dtype=np.int32)
    labels[deficit] = nearest[deficit] + 1
    alpha_residual = (source[:, :, 3] > 0) & (artwork[:, :, 3] < 255)
    return labels, palette, {
        "paint_deficit_pixel_count": int(np.count_nonzero(deficit)),
        "paint_deficit_opaque_artwork_count": int(
            np.count_nonzero(deficit & (artwork[:, :, 3] == 255))
        ),
        "paint_deficit_alpha_residual_overlap": int(
            np.count_nonzero(deficit & alpha_residual)
        ),
        **component_stats,
    }


def build_paint_deficit_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element | None,
    source_rgba_grid: np.ndarray,
    transaction_id: str,
) -> tuple[ET.Element, dict[str, Any]]:
    """Build one q24 source-alpha mask plus a compact deficit overlay."""
    from app.alpha_candidate_knockout import (  # noqa: PLC0415
        _local_name,
        _render_root,
        _unique_id,
        _viewbox,
    )
    from app.alpha_candidate_painter import (  # noqa: PLC0415
        _painter_loops,
        _simplify_rectilinear_loop,
    )
    from app.alpha_candidate_support import (  # noqa: PLC0415
        _PROTECTED_ROOT_TAGS,
        _strip_content_alpha,
    )
    from app.alpha_svg_mask import _merged_rectangles_by_level  # noqa: PLC0415

    source = np.asarray(source_rgba_grid, dtype=np.uint8)
    grid_height, grid_width = source.shape[:2]
    canvas_index = (
        list(original_root).index(canvas_element)
        if canvas_element is not None
        else None
    )

    artwork_root = copy.deepcopy(original_root)
    if canvas_index is not None:
        artwork_root.remove(list(artwork_root)[canvas_index])
    artwork_rgba = _render_root(artwork_root, grid_width, grid_height)
    if artwork_rgba is None:
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_render_unmeasured"
        )
    labels, palette, deficit_stats = _paint_deficit_labels(
        source,
        artwork_rgba,
    )
    if int(deficit_stats["paint_deficit_pixel_count"]) <= 0:
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_empty"
        )

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    root = copy.deepcopy(original_root)
    target_canvas = (
        list(root)[canvas_index] if canvas_index is not None else None
    )
    defs = next(
        (
            child
            for child in list(root)
            if _local_name(str(child.tag)).lower() == "defs"
        ),
        None,
    )
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)

    paint_id = _unique_id(root, "vektoryum-alpha-deficit-artwork")
    paint = ET.SubElement(
        defs,
        qname("g"),
        {
            "id": paint_id,
            "data-vektoryum-alpha-candidate-paint": "preserved-v2",
        },
    )
    tag_transform_node(paint, ROLE_ARTWORK_CONTAINER, transaction_id)
    artwork_count = 0
    movable = [
        child
        for child in list(root)
        if child is not defs
        and _local_name(str(child.tag)).lower()
        not in _PROTECTED_ROOT_TAGS
    ]
    for child in movable:
        root.remove(child)
        if child is target_canvas:
            continue
        _strip_content_alpha(child)
        paint.append(child)
        artwork_count += 1
    if artwork_count <= 0:
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_no_artwork"
        )

    view_x, view_y, view_width, view_height = _viewbox(root)
    quantized, opacity = _fixed_alpha_levels(source[:, :, 3])
    loops = _painter_loops(quantized, opacity)
    if not loops:
        raise RuntimeError(
            "source_alpha_candidate_painter_paint_deficit_empty_alpha"
        )
    mask_id = _unique_id(root, "vektoryum-alpha-deficit-mask")
    mask = ET.SubElement(
        defs,
        qname("mask"),
        {
            "id": mask_id,
            "maskUnits": "userSpaceOnUse",
            "maskContentUnits": "userSpaceOnUse",
            "x": f"{view_x:g}",
            "y": f"{view_y:g}",
            "width": f"{view_width:g}",
            "height": f"{view_height:g}",
            "data-vektoryum-source-alpha-reconstruction": (
                "paint-deficit-q24-v1"
            ),
        },
    )
    tag_transform_node(mask, ROLE_MASK_DEFINITION, transaction_id)
    content = ET.SubElement(
        mask,
        qname("g"),
        {
            "transform": (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / grid_width:.12g} "
                f"{view_height / grid_height:.12g})"
            )
        },
    )
    ET.SubElement(
        content,
        qname("rect"),
        {
            "x": "0",
            "y": "0",
            "width": str(grid_width),
            "height": str(grid_height),
            "fill": "rgb(0,0,0)",
        },
    )
    mask_polygon_count = 0
    for _area, corners, gray in loops:
        simplified = _simplify_rectilinear_loop(corners)
        if len(simplified) < 3:
            continue
        ET.SubElement(
            content,
            qname("polygon"),
            {
                "points": " ".join(
                    f"{x},{y}" for x, y in simplified
                ),
                "fill": f"rgb({gray},{gray},{gray})",
            },
        )
        mask_polygon_count += 1

    layer = ET.SubElement(
        root,
        qname("g"),
        {
            "data-vektoryum-source-alpha-reconstruction": (
                "paint-deficit-q24-v1"
            )
        },
    )
    tag_transform_node(layer, ROLE_MASK_APPLICATION, transaction_id)
    ET.SubElement(
        layer,
        qname("use"),
        {"href": f"#{paint_id}"},
    )

    support = ET.SubElement(
        layer,
        qname("g"),
        {
            "transform": (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / grid_width:.12g} "
                f"{view_height / grid_height:.12g})"
            ),
            "data-vektoryum-paint-deficit": "source-palette-v1",
        },
    )
    tag_transform_node(support, ROLE_MASK_GEOMETRY, transaction_id)
    rectangles = _merged_rectangles_by_level(labels)
    support_rect_count = 0
    used_palette_count = 0
    for label in sorted(rectangles):
        if label <= 0 or label > len(palette):
            continue
        level_rectangles = rectangles[label]
        if not level_rectangles:
            continue
        color = palette[label - 1]
        group = ET.SubElement(
            support,
            qname("g"),
            {
                "fill": (
                    f"rgb({int(color[0])},{int(color[1])},"
                    f"{int(color[2])})"
                )
            },
        )
        for x, y, width, height in level_rectangles:
            ET.SubElement(
                group,
                qname("rect"),
                {
                    "x": str(x),
                    "y": str(y),
                    "width": str(width),
                    "height": str(height),
                },
            )
            support_rect_count += 1
        used_palette_count += 1

    layer.set("mask", f"url(#{mask_id})")
    return root, {
        "reconstruction_mask_encoding": "paint-deficit-q24-polygon",
        "reconstruction_loop_count": int(len(loops)),
        "reconstruction_mask_polygon_count": int(mask_polygon_count),
        "encoded_alpha_level_count": int(
            len([value for value in opacity if int(value) > 0])
        ),
        "paint_deficit_palette_count": int(used_palette_count),
        "paint_deficit_support_rect_count": int(support_rect_count),
        "comparison_canvas_knocked_out": bool(target_canvas is not None),
        "comparison_canvas_retained_under_mask": False,
        "candidate_support_expanded_geometry_count": 0,
        **deficit_stats,
    }
