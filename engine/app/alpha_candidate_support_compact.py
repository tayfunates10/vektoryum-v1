"""Compact renderer-native alpha geometry for RFV-3D2 support fallback.

Repeated native-grid rectangles are defined once and referenced with short SVG
IDs. This preserves exact alpha cells while keeping the unchanged
TransformJournal byte, path and path-command limits authoritative.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_knockout import _local_name, _viewbox
from app.alpha_candidate_paint_selection import expand_non_canvas_paint
from app.alpha_candidate_support import (
    _PROTECTED_ROOT_TAGS,
    _SVG_NS,
    _merged_rectangles_by_level,
    _strip_content_alpha,
)


def _private_prefix(root: ET.Element) -> str:
    used = {str(element.get("id")) for element in root.iter() if element.get("id")}
    prefix = "z"
    while any(identifier.startswith(prefix) for identifier in used):
        prefix += "z"
    return prefix


def build_compact_native_use_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    stroke_width: float,
) -> tuple[ET.Element, dict[str, int | list[int] | None]]:
    """Build exact native-grid clips with short reusable rectangle references."""
    root = copy.deepcopy(original_root)
    original_children = list(original_root)
    canvas_index = original_children.index(canvas_element)
    target_canvas = list(root)[canvas_index]
    root.remove(target_canvas)

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    defs = next(
        (child for child in list(root) if _local_name(str(child.tag)).lower() == "defs"),
        None,
    )
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)

    prefix = _private_prefix(root)
    archive = ET.SubElement(
        defs,
        qname("g"),
        {
            "display": "none",
            "data-vektoryum-candidate-geometry-knockout": "comparison-canvas-v1",
        },
    )
    archive.append(target_canvas)

    paint_id = f"{prefix}p"
    paint = ET.SubElement(
        defs,
        qname("g"),
        {"id": paint_id, "data-vektoryum-alpha-candidate-paint": "preserved-v1"},
    )
    movable = [
        child
        for child in list(root)
        if child is not defs
        and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not movable:
        raise RuntimeError("source_alpha_candidate_support_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)
    expanded_count, canvas_matched_skip_count, canvas_rgb = expand_non_canvas_paint(
        paint,
        stroke_width,
        target_canvas,
    )
    if expanded_count <= 0:
        raise RuntimeError("source_alpha_candidate_support_no_contrasting_fill_geometry")

    rectangles = _merged_rectangles_by_level(quantized)
    if not rectangles:
        raise RuntimeError("source_alpha_candidate_support_empty_source")
    dimensions = sorted(
        {
            (width, height)
            for level_rectangles in rectangles.values()
            for _x, _y, width, height in level_rectangles
            if width > 0 and height > 0
        }
    )
    if not dimensions:
        raise RuntimeError("source_alpha_candidate_support_no_rectangles")

    symbol_by_size: dict[tuple[int, int], str] = {}
    for index, (width, height) in enumerate(dimensions):
        symbol_id = f"{prefix}r{index:x}"
        symbol_by_size[(width, height)] = symbol_id
        ET.SubElement(
            defs,
            qname("rect"),
            {"id": symbol_id, "width": str(width), "height": str(height)},
        )

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_height, raster_width = quantized.shape
    sx = view_width / float(raster_width)
    sy = view_height / float(raster_height)
    transform = (
        f"translate({view_x:.12g} {view_y:.12g}) "
        f"scale({sx:.12g} {sy:.12g})"
    )
    reconstruction = ET.SubElement(
        root,
        qname("g"),
        {"data-vektoryum-source-alpha-reconstruction": "native-grid-use-v1"},
    )

    clip_count = 0
    rectangle_count = 0
    use_count = 0
    for level in sorted(rectangles):
        level_rectangles = rectangles[level]
        if not level_rectangles:
            continue
        clip_id = f"{prefix}c{int(level):x}"
        clip = ET.SubElement(
            defs,
            qname("clipPath"),
            {
                "id": clip_id,
                "clipPathUnits": "userSpaceOnUse",
                "transform": transform,
            },
        )
        for x, y, width, height in level_rectangles:
            if width <= 0 or height <= 0:
                continue
            ET.SubElement(
                clip,
                qname("use"),
                {
                    "href": f"#{symbol_by_size[(width, height)]}",
                    "x": str(x),
                    "y": str(y),
                },
            )
            rectangle_count += 1
        if len(clip) == 0:
            defs.remove(clip)
            continue
        clip_count += 1
        layer = ET.SubElement(
            reconstruction,
            qname("g"),
            {"opacity": f"{float(opacity_by_level[level]):.8f}".rstrip("0").rstrip(".")},
        )
        ET.SubElement(
            layer,
            qname("use"),
            {"href": f"#{paint_id}", "clip-path": f"url(#{clip_id})"},
        )
        use_count += 1

    if rectangle_count == 0 or use_count == 0:
        raise RuntimeError("source_alpha_candidate_support_no_reconstruction")
    return root, {
        "reconstruction_clip_count": int(clip_count),
        "reconstruction_rectangle_count": int(rectangle_count),
        "reconstruction_use_count": int(use_count),
        "reconstruction_rect_symbol_count": int(len(symbol_by_size)),
        "candidate_support_expanded_geometry_count": int(expanded_count),
        "candidate_support_canvas_matched_skip_count": int(canvas_matched_skip_count),
        "candidate_support_canvas_rgb": list(canvas_rgb) if canvas_rgb is not None else None,
        "reconstruction_compact_id_prefix_length": int(len(prefix)),
    }
