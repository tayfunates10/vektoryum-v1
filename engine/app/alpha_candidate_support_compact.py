"""Compact renderer-native alpha geometry for RFV-3D2 support fallback.

Repeated native-grid rectangles are defined once and referenced with short SVG
IDs. This preserves exact alpha cells while keeping the unchanged
TransformJournal byte, path and path-command limits authoritative.
"""
from __future__ import annotations

import copy
import hashlib
import os
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

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


_DIRECT_DIAGNOSTIC_ATTRIBUTES = {
    "data-vektoryum-source-alpha-archive",
    "data-vektoryum-source-alpha-direct",
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _rect_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height or x1 < x0 or y1 < y0:
        return np.zeros(shape, dtype=bool)
    result = np.zeros(shape, dtype=bool)
    result[y0 : y1 + 1, x0 : x1 + 1] = True
    return result


def _ellipse_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    height, width = shape
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height or x1 < x0 or y1 < y0:
        return np.zeros(shape, dtype=bool)
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).ellipse((x0, y0, x1, y1), fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _candidate_boxes(
    target: np.ndarray,
    residual: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for mask in (residual, target):
        box = _bbox(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        for delta in (0, -1, 1):
            candidate = (x0 - delta, y0 - delta, x1 + delta, y1 + delta)
            if candidate not in boxes:
                boxes.append(candidate)

    # A higher-alpha rectangle can cover the top and bottom of an underlying
    # circle/ellipse, so the residual bounding box loses those extrema. Recover
    # the symmetric square candidate from the still-visible horizontal span and
    # the residual centroid; exact pixel equality below remains authoritative.
    ys, xs = np.nonzero(residual)
    if len(xs):
        residual_box = _bbox(residual)
        assert residual_box is not None
        x0, y0, x1, y1 = residual_box
        center_x = int(round(float(xs.mean())))
        center_y = int(round(float(ys.mean())))
        radius_x = (x1 - x0) / 2.0
        radius_y = (y1 - y0) / 2.0
        inferred = [
            (x0, int(round(center_y - radius_x)), x1, int(round(center_y + radius_x))),
            (int(round(center_x - radius_y)), y0, int(round(center_x + radius_y)), y1),
        ]
        for candidate in inferred:
            if candidate not in boxes:
                boxes.append(candidate)
    return boxes


def _fit_single_primitive(
    target: np.ndarray,
    higher: np.ndarray,
) -> tuple[str, tuple[int, int, int, int]] | None:
    residual = np.asarray(target, dtype=bool) & ~np.asarray(higher, dtype=bool)
    if not bool(residual.any()):
        return None
    boxes = _candidate_boxes(target, residual)
    for kind, factory in (("rect", _rect_mask), ("ellipse", _ellipse_mask)):
        for box in boxes:
            primitive = factory(target.shape, box)
            if np.array_equal(primitive | higher, target):
                return kind, box
    return None


def _primitive_layers(alpha: np.ndarray) -> list[tuple[int, str, tuple[int, int, int, int]]] | None:
    levels = sorted(int(value) for value in np.unique(alpha) if int(value) > 0)
    if not levels or len(levels) > 4:
        return None
    layers: list[tuple[int, str, tuple[int, int, int, int]]] = []
    alpha_i = np.asarray(alpha, dtype=np.uint8)
    for level in levels:
        target = alpha_i >= level
        higher = alpha_i > level
        fitted = _fit_single_primitive(target, higher)
        if fitted is None:
            return None
        kind, box = fitted
        layers.append((level, kind, box))
    return layers


def _build_primitive_mask_tree(
    original_root: ET.Element,
    alpha: np.ndarray,
    layers: list[tuple[int, str, tuple[int, int, int, int]]],
    transaction_id: str,
) -> tuple[ET.Element, dict[str, Any]]:
    from app.alpha_artwork_identity import (
        ROLE_ARTWORK_CONTAINER,
        ROLE_MASK_APPLICATION,
        ROLE_MASK_DEFINITION,
        tag_transform_node,
    )
    from app.alpha_candidate_knockout import _unique_id

    root = copy.deepcopy(original_root)
    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    defs = next(
        (child for child in list(root) if _local_name(str(child.tag)).lower() == "defs"),
        None,
    )
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)

    paint_id = _unique_id(root, "vektoryum-alpha-primitive-paint")
    paint = ET.SubElement(defs, qname("g"), {"id": paint_id})
    tag_transform_node(paint, ROLE_ARTWORK_CONTAINER, transaction_id)
    movable = [
        child
        for child in list(root)
        if child is not defs
        and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not movable:
        raise RuntimeError("source_alpha_primitive_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_height, raster_width = alpha.shape
    mask_id = _unique_id(root, "vektoryum-alpha-primitive-mask")
    mask = ET.SubElement(
        defs,
        qname("mask"),
        {
            "id": mask_id,
            "maskUnits": "userSpaceOnUse",
            "x": f"{view_x:g}",
            "y": f"{view_y:g}",
            "width": f"{view_width:g}",
            "height": f"{view_height:g}",
        },
    )
    tag_transform_node(mask, ROLE_MASK_DEFINITION, transaction_id)
    content = ET.SubElement(
        mask,
        qname("g"),
        {
            "transform": (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / float(raster_width):.12g} "
                f"{view_height / float(raster_height):.12g})"
            )
        },
    )
    ET.SubElement(
        content,
        qname("rect"),
        {
            "x": "0",
            "y": "0",
            "width": str(raster_width),
            "height": str(raster_height),
            "fill": "rgb(0,0,0)",
        },
    )
    for level, kind, (x0, y0, x1, y1) in layers:
        attributes = {
            "fill": f"rgb({level},{level},{level})",
            "x": str(x0),
            "y": str(y0),
            "width": str(x1 - x0 + 1),
            "height": str(y1 - y0 + 1),
        }
        if kind == "ellipse":
            attributes = {
                "fill": f"rgb({level},{level},{level})",
                "cx": f"{(x0 + x1) / 2.0:g}",
                "cy": f"{(y0 + y1) / 2.0:g}",
                "rx": f"{(x1 - x0) / 2.0:g}",
                "ry": f"{(y1 - y0) / 2.0:g}",
            }
        ET.SubElement(content, qname(kind), attributes)

    layer = ET.SubElement(root, qname("g"))
    tag_transform_node(layer, ROLE_MASK_APPLICATION, transaction_id)
    ET.SubElement(
        layer,
        qname("use"),
        {"href": f"#{paint_id}", "mask": f"url(#{mask_id})"},
    )
    return root, {
        "primitive_count": int(len(layers)),
        "primitive_kinds": [kind for _level, kind, _box in layers],
        "primitive_alpha_levels": [int(level) for level, _kind, _box in layers],
    }


def _apply_compact_primitive_alpha(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    from PIL import Image
    from app.alpha_artwork_identity import alpha_transaction_id
    from app.alpha_candidate_knockout import _path_node_counts, _render_root, _write_tree_to_temp
    from app.alpha_candidate_validation import validate_alpha_reconstruction_contract
    from app.alpha_mask_budget import _journal_limits
    from app.alpha_preprocess import _rgba_from_source_at_size

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    original_root = ET.fromstring(before_bytes)
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_primitive_root_not_svg")
    parent_counts = _path_node_counts(original_root)
    _vx, _vy, view_width, view_height = _viewbox(original_root)
    scale = min(1.0, 512.0 / float(max(view_width, view_height)))
    grid_width = max(1, int(round(view_width * scale)))
    grid_height = max(1, int(round(view_height * scale)))
    source_grid = _rgba_from_source_at_size(source, (grid_width, grid_height))
    alpha = np.asarray(source_grid[:, :, 3], dtype=np.uint8)
    if bool(np.all(alpha == 255)) or not bool(np.any(alpha > 0)):
        raise RuntimeError("source_alpha_primitive_not_applicable")
    collapsed = _render_root(original_root, grid_width, grid_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_primitive_parent_render_unmeasured")
    coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if coverage < 0.98:
        raise RuntimeError("source_alpha_primitive_parent_not_collapsed")
    layers = _primitive_layers(alpha)
    if layers is None:
        raise RuntimeError("source_alpha_primitive_geometry_not_exact")

    parent_sha = hashlib.sha256(before_bytes).hexdigest()
    alpha_sha = hashlib.sha256(np.ascontiguousarray(alpha).tobytes()).hexdigest()
    transaction_id = alpha_transaction_id(parent_sha, alpha_sha, mode, "primitive")
    candidate_root, geometry = _build_primitive_mask_tree(
        original_root, alpha, layers, transaction_id
    )
    temporary = _write_tree_to_temp(candidate_root, target)
    try:
        limits = _journal_limits(original_root, before_size)
        candidate_size = int(temporary.stat().st_size)
        if candidate_size > int(limits["byte_limit"]):
            raise RuntimeError(
                "source_alpha_primitive_byte_budget_rejected:"
                f"{candidate_size}>{limits['byte_limit']}"
            )
        with Image.open(source) as source_image:
            source_size = source_image.size
        source_full = _rgba_from_source_at_size(source, source_size)
        validation = validate_alpha_reconstruction_contract(
            temporary, source_full, mode, parent_counts
        )
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "status": "accepted",
        "applied": True,
        "schema": "rfv3d3-compact-primitive-alpha-v1",
        "mask_encoding": "compact_primitive_luminance_mask",
        "before_byte_size": int(before_size),
        "after_byte_size": int(target.stat().st_size),
        "before_sha256": parent_sha,
        "after_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "candidate_geometry_preserved": True,
        "candidate_path_data_preserved": True,
        "candidate_identity_preserved": True,
        "preflight_parent_path_count": int(parent_counts[0]),
        "preflight_parent_node_count": int(parent_counts[1]),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_node_limit": int(limits["node_limit"]),
        "preflight_byte_limit": int(limits["byte_limit"]),
        **geometry,
        **validation,
    }


def make_compact_primitive_alpha_first(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    if getattr(guarded_builder, "__vektoryum_compact_primitive_alpha_first__", False):
        return guarded_builder

    @wraps(guarded_builder)
    def primitive_first(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        parent = target.read_bytes()
        try:
            return _apply_compact_primitive_alpha(target, Path(source_path), mode)
        except Exception as exc:
            if target.read_bytes() != parent:
                target.write_bytes(parent)
            report = guarded_builder(target, Path(source_path), mode)
            if isinstance(report, dict):
                report = dict(report)
                report["compact_primitive_alpha"] = {
                    "status": "not_applicable_fallback_used",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            return report

    primitive_first.__vektoryum_compact_primitive_alpha_first__ = True
    return primitive_first


def _compact_direct_artifact(
    original_apply: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    @wraps(original_apply)
    def compacted(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        from PIL import Image
        from app.alpha_candidate_knockout import _path_node_counts
        from app.alpha_candidate_validation import validate_alpha_reconstruction_contract
        from app.alpha_preprocess import _rgba_from_source_at_size

        target = Path(svg_path)
        source = Path(source_path)
        parent_root = ET.fromstring(target.read_bytes())
        parent_counts = _path_node_counts(parent_root)
        report = original_apply(target, source, mode)
        accepted_bytes = target.read_bytes()
        try:
            root = ET.fromstring(accepted_bytes)
            removed = 0
            seen_diagnostics: set[tuple[str, str]] = set()
            for element in root.iter():
                for name in _DIRECT_DIAGNOSTIC_ATTRIBUTES:
                    if name not in element.attrib:
                        continue
                    marker = (name, str(element.attrib[name]))
                    if marker in seen_diagnostics:
                        element.attrib.pop(name, None)
                        removed += 1
                    else:
                        seen_diagnostics.add(marker)
            if removed == 0:
                return report
            candidate = ET.tostring(
                root,
                encoding="utf-8",
                xml_declaration=True,
                short_empty_elements=True,
            )
            if len(candidate) >= len(accepted_bytes):
                return report
            temporary = target.parent / f".{target.name}.direct-compact.svg"
            temporary.write_bytes(candidate)
            try:
                with Image.open(source) as source_image:
                    source_size = source_image.size
                source_full = _rgba_from_source_at_size(source, source_size)
                validate_alpha_reconstruction_contract(
                    temporary, source_full, mode, parent_counts
                )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        except Exception:
            target.write_bytes(accepted_bytes)
            return report

        result = dict(report)
        result["after_byte_size"] = int(target.stat().st_size)
        result["after_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        result["direct_diagnostic_attributes_removed"] = int(removed)
        result["direct_artifact_compacted"] = True
        return result

    compacted.__vektoryum_direct_artifact_compacted__ = True
    return compacted


def _install_runtime_compactors() -> None:
    from app import alpha_candidate_direct as direct_module
    from app import alpha_mask_adaptive as adaptive_module

    direct_apply = direct_module.apply_direct_element_alpha
    if not getattr(direct_apply, "__vektoryum_direct_artifact_compacted__", False):
        direct_module.apply_direct_element_alpha = _compact_direct_artifact(direct_apply)

    adaptive_factory = adaptive_module.make_adaptive_apply_source_alpha_mask
    if not getattr(adaptive_factory, "__vektoryum_primitive_factory_wrapped__", False):
        original_factory = adaptive_factory

        @wraps(original_factory)
        def compact_factory(rect_builder):
            return make_compact_primitive_alpha_first(original_factory(rect_builder))

        compact_factory.__vektoryum_primitive_factory_wrapped__ = True
        adaptive_module.make_adaptive_apply_source_alpha_mask = compact_factory


_install_runtime_compactors()
