"""Compact renderer-stable encoding for candidate-geometry alpha knockout.

This module changes only the serialization of the existing reconstruction:
repeated equal-sized clip rectangles become short reusable SVG ``<use>`` nodes.
Candidate paint, path data, path/node counts, alpha levels and evaluator gates are
unchanged. The encoder uses direct user-space coordinates (no clipPath transform)
to preserve the renderer agreement established by the original implementation.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _local_name, _viewbox

_SVG_NS = "http://www.w3.org/2000/svg"
_PROTECTED_ROOT_TAGS = {"defs", "title", "desc", "metadata", "style"}


def _short_prefix(root: ET.Element) -> str:
    used = {str(element.get("id")) for element in root.iter() if element.get("id")}
    prefix = "q"
    while any(identifier.startswith(prefix) for identifier in used):
        prefix += "q"
    return prefix


def _strip_content_alpha(element: ET.Element) -> None:
    from app.alpha_svg_mask import _strip_content_alpha as existing

    existing(element)


def _merged_rectangles(quantized: np.ndarray) -> dict[int, list[tuple[int, int, int, int]]]:
    from app.alpha_svg_mask import _merged_rectangles_by_level

    return _merged_rectangles_by_level(quantized)


def build_compact_knockout_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> tuple[ET.Element, dict[str, Any]]:
    """Build the original knockout contract with compact reusable rectangles."""
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

    prefix = _short_prefix(root)
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
        raise RuntimeError("source_alpha_candidate_knockout_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)

    rectangles = _merged_rectangles(quantized)
    if not rectangles:
        raise RuntimeError("source_alpha_candidate_knockout_empty_source")

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_height, raster_width = quantized.shape
    sx = view_width / float(raster_width)
    sy = view_height / float(raster_height)

    dimensions = sorted(
        {
            (width, height)
            for level_rectangles in rectangles.values()
            for _x, _y, width, height in level_rectangles
            if width > 0 and height > 0
        }
    )
    if not dimensions:
        raise RuntimeError("source_alpha_candidate_knockout_no_reconstruction")

    symbol_by_size: dict[tuple[int, int], str] = {}
    for index, (width, height) in enumerate(dimensions):
        symbol_id = f"{prefix}r{index:x}"
        symbol_by_size[(width, height)] = symbol_id
        ET.SubElement(
            defs,
            qname("rect"),
            {
                "id": symbol_id,
                "width": f"{width * sx:.12g}",
                "height": f"{height * sy:.12g}",
            },
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
            {"id": clip_id, "clipPathUnits": "userSpaceOnUse"},
        )
        for x, y, width, height in level_rectangles:
            if width <= 0 or height <= 0:
                continue
            ET.SubElement(
                clip,
                qname("use"),
                {
                    "href": f"#{symbol_by_size[(width, height)]}",
                    "x": f"{view_x + x * sx:.12g}",
                    "y": f"{view_y + y * sy:.12g}",
                },
            )
            rectangle_count += 1
        if len(clip) == 0:
            defs.remove(clip)
            continue
        clip_count += 1
        layer = ET.SubElement(
            root,
            qname("g"),
            {
                "opacity": f"{float(opacity_by_level[level]):.8f}".rstrip("0").rstrip("."),
                "data-vektoryum-source-alpha-reconstruction": "compact-clip-use-v1",
            },
        )
        ET.SubElement(
            layer,
            qname("use"),
            {"href": f"#{paint_id}", "clip-path": f"url(#{clip_id})"},
        )
        use_count += 1

    if rectangle_count == 0 or use_count == 0:
        raise RuntimeError("source_alpha_candidate_knockout_no_reconstruction")
    return root, {
        "reconstruction_clip_count": int(clip_count),
        "reconstruction_rectangle_count": int(rectangle_count),
        "reconstruction_use_count": int(use_count),
        "reconstruction_rect_symbol_count": int(len(symbol_by_size)),
        "reconstruction_encoding": "compact-user-space-use-v1",
        "reconstruction_compact_id_prefix_length": int(len(prefix)),
    }
