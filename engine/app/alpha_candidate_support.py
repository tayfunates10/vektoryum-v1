"""Renderer-native candidate-paint support reconstruction.

This is the final narrow RFV-3D2 fallback. It runs only after the ordinary
vector mask and the comparison-canvas knockout both fail the unchanged alpha
IoU/MAE gates. The selected candidate's path data and candidate identity remain
unchanged. Existing filled geometry receives the smallest measured same-color
stroke, painted behind its fill, and is clipped by source alpha sampled on the
primary SVG renderer's native output grid.

The native grid matters for aspect ratios whose requested evaluation height is
rounded differently by the renderer. Reconstructing on a nominal raster and
then letting the renderer resample it created a second alpha-boundary error.
This module probes the renderer output dimensions, builds exact cell geometry
on that grid and encodes repeated rectangles through reusable SVG ``<use>``
primitives. Path and path-command counts therefore remain unchanged, while the
exact existing TransformJournal byte limit still decides admissibility.
"""
from __future__ import annotations

import copy
import io
import os
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import (
    _alpha_encodings,
    _comparison_canvas_candidate,
    _local_name,
    _path_node_counts,
    _render_root,
    _unique_id,
    _validate_reconstruction,
    _viewbox,
    _write_tree_to_temp,
)
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_SUPPORT_STROKE_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
_SUPPORT_FAILURE_PREFIXES = (
    "source_alpha_candidate_knockout_iou_gate_failed:",
    "source_alpha_candidate_knockout_mae_gate_failed:",
)
_GEOMETRY_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline"}
_ALPHA_STYLE_NAMES = {"opacity", "fill-opacity", "stroke-opacity"}
_PROTECTED_ROOT_TAGS = {"defs", "title", "desc", "metadata", "style"}
_NATIVE_GRID_MAX_SIDE = 512


def _style_declarations(style: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in str(style or "").split(";"):
        if ":" not in declaration:
            continue
        key, value = declaration.split(":", 1)
        result[key.strip().lower()] = value.strip()
    return result


def _write_style(node: ET.Element, declarations: dict[str, str]) -> None:
    if declarations:
        node.set("style", ";".join(f"{key}:{value}" for key, value in declarations.items()))
    else:
        node.attrib.pop("style", None)


def _expand_candidate_paint(
    node: ET.Element,
    stroke_width: float,
    inherited_fill: str | None = None,
) -> int:
    """Expand fill support without changing path data or adding path nodes."""
    declarations = _style_declarations(node.get("style"))
    fill = declarations.get("fill", node.get("fill", inherited_fill))
    if fill is None:
        fill = "#000000"
    local = _local_name(str(node.tag)).lower()
    count = 0
    if local in _GEOMETRY_TAGS and fill.strip().lower() not in {
        "none",
        "transparent",
        "rgba(0,0,0,0)",
    }:
        for key in (
            "stroke",
            "stroke-width",
            "stroke-linejoin",
            "stroke-linecap",
            "stroke-dasharray",
            "stroke-dashoffset",
            "stroke-opacity",
            "paint-order",
        ):
            declarations.pop(key, None)
            node.attrib.pop(key, None)
        for key in _ALPHA_STYLE_NAMES:
            declarations.pop(key, None)
            node.attrib.pop(key, None)
        _write_style(node, declarations)
        node.set("stroke", fill)
        node.set("stroke-width", f"{stroke_width:.8f}".rstrip("0").rstrip("."))
        node.set("stroke-linejoin", "round")
        node.set("stroke-linecap", "round")
        node.set("paint-order", "stroke fill markers")
        count = 1
    for child in list(node):
        count += _expand_candidate_paint(child, stroke_width, fill)
    return count


def _renderer_native_grid(
    svg_path: Path,
    view_width: float,
    view_height: float,
) -> tuple[int, int, int, int, str]:
    """Return requested and primary-renderer-native evaluation dimensions."""
    scale = min(1.0, _NATIVE_GRID_MAX_SIDE / float(max(view_width, view_height)))
    requested_width = max(1, int(round(view_width * scale)))
    requested_height = max(1, int(round(view_height * scale)))
    native_width = requested_width
    native_height = requested_height
    renderer = "requested_grid_fallback"
    try:
        import resvg_py  # type: ignore  # noqa: PLC0415

        data = bytes(
            resvg_py.svg_to_bytes(
                svg_path=str(svg_path),
                width=requested_width,
                height=requested_height,
            )
        )
        with Image.open(io.BytesIO(data)) as rendered:
            native_width, native_height = rendered.size
        renderer = "resvg_native_grid"
    except Exception:
        # The existing source-truth renderer already falls back to CairoSVG. In
        # that environment the requested grid is the only deterministic contract.
        pass
    if native_width <= 0 or native_height <= 0:
        raise RuntimeError("source_alpha_candidate_support_native_grid_invalid")
    if max(native_width, native_height) > _NATIVE_GRID_MAX_SIDE + 1:
        raise RuntimeError(
            "source_alpha_candidate_support_native_grid_unbounded:"
            f"{native_width}x{native_height}"
        )
    return (
        requested_width,
        requested_height,
        int(native_width),
        int(native_height),
        renderer,
    )


def _strip_content_alpha(element: ET.Element) -> None:
    """Remove candidate opacity so source alpha is applied exactly once."""
    for node in element.iter():
        for name in _ALPHA_STYLE_NAMES:
            node.attrib.pop(name, None)
        declarations = _style_declarations(node.get("style"))
        for name in _ALPHA_STYLE_NAMES:
            declarations.pop(name, None)
        _write_style(node, declarations)


def _merged_rectangles_by_level(
    quantized: np.ndarray,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Run-length encode equal-alpha pixels and merge identical runs vertically."""
    height, width = quantized.shape
    completed: dict[int, list[tuple[int, int, int, int]]] = {}
    active: dict[tuple[int, int, int], list[int]] = {}
    for y in range(height):
        row = quantized[y]
        runs: list[tuple[int, int, int]] = []
        x = 0
        while x < width:
            level = int(row[x])
            start = x
            x += 1
            while x < width and int(row[x]) == level:
                x += 1
            if level > 0:
                runs.append((level, start, x))
        current = {(level, x0, x1) for level, x0, x1 in runs}
        for key in list(active):
            if key in current:
                continue
            level, x0, x1 = key
            y0, y1 = active.pop(key)
            completed.setdefault(level, []).append((x0, y0, x1 - x0, y1 - y0))
        for level, x0, x1 in runs:
            key = (level, x0, x1)
            if key in active:
                active[key][1] = y + 1
            else:
                active[key] = [y, y + 1]
    for (level, x0, x1), (y0, y1) in active.items():
        completed.setdefault(level, []).append((x0, y0, x1 - x0, y1 - y0))
    return completed


def _build_native_use_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    stroke_width: float,
) -> tuple[ET.Element, dict[str, int]]:
    """Build exact native-grid alpha clips using reusable rectangle symbols."""
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

    archive = ET.SubElement(
        defs,
        qname("g"),
        {
            "data-vektoryum-candidate-geometry-knockout": "comparison-canvas-v1",
            "display": "none",
        },
    )
    archive.append(target_canvas)

    paint_id = _unique_id(root, "vektoryum-alpha-candidate-paint")
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
    expanded_count = _expand_candidate_paint(paint, stroke_width)
    if expanded_count <= 0:
        raise RuntimeError("source_alpha_candidate_support_no_fill_geometry")

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
        symbol_id = _unique_id(root, f"vektoryum-alpha-cell-{index}")
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
    clip_count = 0
    rectangle_count = 0
    use_count = 0
    for level in sorted(rectangles):
        level_rectangles = rectangles[level]
        if not level_rectangles:
            continue
        clip_id = _unique_id(root, f"vektoryum-alpha-level-{level}")
        clip = ET.SubElement(
            defs,
            qname("clipPath"),
            {
                "id": clip_id,
                "clipPathUnits": "userSpaceOnUse",
                "transform": transform,
                "data-vektoryum-alpha-level": str(level),
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
            root,
            qname("g"),
            {
                "opacity": f"{float(opacity_by_level[level]):.8f}".rstrip("0").rstrip("."),
                "data-vektoryum-source-alpha-reconstruction": "native-grid-use-v1",
                "data-vektoryum-alpha-level": str(level),
            },
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
    }


def apply_candidate_support_reconstruction(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Choose the smallest admissible native-grid support expansion."""
    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    original_root = ET.fromstring(before_bytes)
    parent_counts = _path_node_counts(original_root)
    _view_x, _view_y, view_width, view_height = _viewbox(original_root)
    (
        requested_width,
        requested_height,
        native_width,
        native_height,
        native_grid_source,
    ) = _renderer_native_grid(target, view_width, view_height)

    source_rgba_native = _rgba_from_source_at_size(
        source, (native_width, native_height)
    )
    source_alpha_native = np.asarray(
        source_rgba_native[:, :, 3], dtype=np.uint8
    ).copy()
    if bool(np.all(source_alpha_native == 255)):
        raise RuntimeError("source_alpha_candidate_support_opaque_source")
    if not np.any(source_alpha_native > 0):
        raise RuntimeError("source_alpha_candidate_support_empty_source")

    collapsed = _render_root(original_root, native_width, native_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_support_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_support_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    canvas = _comparison_canvas_candidate(
        original_root,
        source_rgba_native,
        native_width,
        native_height,
    )
    limits = _journal_limits(original_root, before_size)

    with Image.open(source) as source_image:
        source_size = source_image.size
    source_rgba_full = _rgba_from_source_at_size(source, source_size)
    last_error: RuntimeError | None = None

    for encoding_name, quantized, opacity_by_level in _alpha_encodings(
        source_alpha_native
    ):
        for stroke_width in _SUPPORT_STROKE_PIXELS:
            candidate_root, geometry = _build_native_use_reconstruction_tree(
                original_root,
                canvas,
                quantized,
                opacity_by_level,
                stroke_width,
            )
            temporary = _write_tree_to_temp(candidate_root, target)
            try:
                projected_size = temporary.stat().st_size
                if projected_size > int(limits["byte_limit"]):
                    last_error = RuntimeError(
                        "source_alpha_candidate_support_byte_budget_rejected:"
                        f"{projected_size}>{limits['byte_limit']}"
                    )
                    continue
                try:
                    validation = _validate_reconstruction(
                        temporary,
                        source_rgba_full,
                        mode,
                        parent_counts,
                    )
                except RuntimeError as exc:
                    if str(exc).startswith(_SUPPORT_FAILURE_PREFIXES):
                        last_error = exc
                        continue
                    raise
                os.replace(temporary, target)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

            return {
                "status": "accepted",
                "applied": True,
                "schema": "rfv3d2-candidate-support-reconstruction-v2",
                "before_byte_size": int(before_size),
                "after_byte_size": int(target.stat().st_size),
                "mask_encoding": "candidate_support_native_grid_use",
                "reconstruction_alpha_encoding": encoding_name,
                "candidate_support_stroke_width_pixels": float(stroke_width),
                "candidate_geometry_preserved": True,
                "candidate_path_data_preserved": True,
                "trace_rgb_bytes_preserved": True,
                "candidate_identity_preserved": True,
                "renderer_grid_source": native_grid_source,
                "renderer_requested_alpha_width": int(requested_width),
                "renderer_requested_alpha_height": int(requested_height),
                "renderer_native_alpha_width": int(native_width),
                "renderer_native_alpha_height": int(native_height),
                "preflight_parent_path_count": int(parent_counts[0]),
                "preflight_parent_node_count": int(parent_counts[1]),
                "preflight_path_limit": int(limits["path_limit"]),
                "preflight_node_limit": int(limits["node_limit"]),
                "preflight_byte_limit": int(limits["byte_limit"]),
                **geometry,
                **validation,
            }

    raise last_error or RuntimeError(
        "source_alpha_candidate_support_no_admissible_reconstruction"
    )


def make_candidate_support_reconstruction_fallback(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Retry only a rolled-back candidate-knockout alpha rejection."""
    if getattr(guarded_builder, "__vektoryum_candidate_support_fallback__", False):
        return guarded_builder

    @wraps(guarded_builder)
    def fallback(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        source = Path(source_path)
        try:
            return guarded_builder(target, source, mode)
        except RuntimeError as first_error:
            trigger = str(first_error)
            if not trigger.startswith(_SUPPORT_FAILURE_PREFIXES):
                raise

        from app.alpha_mask_budget import (  # noqa: PLC0415
            _create_atomic_backup,
            _restore_atomic_backup,
        )

        backup = _create_atomic_backup(target)
        try:
            report = apply_candidate_support_reconstruction(target, source, mode)
        except BaseException:
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)

        report["mask_fallback_reason"] = "candidate_knockout_exact_alpha_failure"
        report["mask_fallback_trigger"] = trigger
        report["rollback_guard"] = "armed_and_committed"
        return report

    fallback.__vektoryum_candidate_support_fallback__ = True
    return fallback
