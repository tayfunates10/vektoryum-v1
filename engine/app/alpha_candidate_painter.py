"""Scale-stable painter-mask alpha reconstruction (final RFV-3D2 fallback).

The renderer-native 512 cell grid reproduces source alpha exactly on the 512
measurement grid, but its constant 2x2-blocky fade ribbon fragments the real
TransformJournal's native-resolution topology refinement (live class_reklam:
components 91->379, holes 15->217). A source-native grid fixes topology but
per-cell tilings then lose alpha under the renderer's 512 coverage compositing.

This module resolves both scales with one deterministic, vector-only geometry:

- alpha is sampled on the source-native raster (bounded), quantized through the
  unchanged ``_quantize_alpha`` contract;
- every alpha level INCLUDING zero becomes exact union contours (shared cell
  edges cancel), so enclosed fully-transparent lakes are explicitly repainted;
- all loops are painted once, largest-area first (containers before contained,
  equal areas resolved darker-first), as opaque grayscale ``<polygon>`` fills
  over an opaque black base inside a single luminance ``<mask>``;
- opaque-over-opaque compositing makes boundary pixels a linear coverage mix of
  neighbouring level grays at every render scale, so junction error scales with
  the local alpha gradient exactly like real image resampling;
- the candidate's paths, path data, colors and identity stay untouched: paint
  moves into a reusable group, receives the smallest admissible same-color
  support stroke behind its fills, and is drawn once through the mask.

Validation is fail-closed with unchanged thresholds: the native-grid render
must reproduce the staged alpha plane, the INTER_AREA-downscaled native render
must match the identically downscaled source on the bounded evaluation grid,
FinalArtifactEvaluator's alpha-plane codes are enforced against the full-
resolution source, and candidate path/node counts must be preserved. RGB
appearance and every parent-relative regression stay owned by the real
TransformJournal stage that follows.
"""
from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.alpha_mask_contour import loop_signed_area, trace_cell_contours
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_PAINTER_GRID_MAX_SIDE = 1600
_PAINTER_STROKE_PIXELS = (1.0, 1.5, 2.0, 3.0)
_PAINTER_EVAL_SIDE = 512.0
_ALPHA_PLANE_FAILURE_CODES = {"alpha_iou_below_min", "alpha_mae_above_max"}


def _painter_loops(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> list[tuple[float, list[tuple[int, int]], int]]:
    """Deterministic painter order: (area desc, gray asc, start corner)."""
    height, width = quantized.shape
    loops: list[tuple[float, list[tuple[int, int]], int]] = []
    for level in sorted(int(value) for value in np.unique(quantized)):
        for corners in trace_cell_contours(quantized == level):
            corner_x, corner_y = corners[0]
            sampled = int(
                quantized[min(corner_y, height - 1), min(corner_x, width - 1)]
            )
            gray = (
                int(round(float(opacity_by_level.get(sampled, 0.0)) * 255))
                if sampled > 0
                else 0
            )
            loops.append((abs(loop_signed_area(corners)), corners, gray))
    loops.sort(key=lambda item: (-item[0], item[2], item[1][0][1], item[1][0][0]))
    return loops


def build_painter_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    stroke_width: float,
) -> tuple[ET.Element, dict[str, int]]:
    """Knockout the comparison canvas and mask the stroked paint once."""
    from app.alpha_candidate_knockout import _local_name, _unique_id, _viewbox  # noqa: PLC0415
    from app.alpha_candidate_support import (  # noqa: PLC0415
        _PROTECTED_ROOT_TAGS,
        _expand_candidate_paint,
        _strip_content_alpha,
    )

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    root = copy.deepcopy(original_root)
    canvas_index = list(original_root).index(canvas_element)
    target_canvas = list(root)[canvas_index]
    root.remove(target_canvas)

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
        raise RuntimeError("source_alpha_candidate_painter_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)
    expanded_count = _expand_candidate_paint(paint, stroke_width)
    if expanded_count <= 0:
        raise RuntimeError("source_alpha_candidate_painter_no_fill_geometry")

    loops = _painter_loops(quantized, opacity_by_level)
    if not loops:
        raise RuntimeError("source_alpha_candidate_painter_empty_source")

    raster_height, raster_width = quantized.shape
    view_x, view_y, view_width, view_height = _viewbox(root)
    mask_id = _unique_id(root, "vektoryum-alpha-painter")
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
            "data-vektoryum-source-alpha-reconstruction": "painter-luminance-v1",
        },
    )
    content = ET.SubElement(mask, qname("g"))
    content.set(
        "transform",
        f"translate({view_x:.12g} {view_y:.12g}) "
        f"scale({view_width / float(raster_width):.12g} "
        f"{view_height / float(raster_height):.12g})",
    )
    # Opak siyah taban: maske kanvasının alfası her yerde 1 kalır; sınır
    # pikselleri komşu gri değerlerinin lineer kapsama karışımına düşer.
    # Griler standart ``rgb(v,v,v)`` gösterimiyle yazılır: bunlar sanat
    # paletinin renkleri değil maske-içi luminance kodlamasıdır ve palet
    # sayımı sanat eserinin hex fill'lerini ölçmeye devam eder.
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
    for _area, corners, gray in loops:
        ET.SubElement(
            content,
            qname("polygon"),
            {
                "points": " ".join(f"{x},{y}" for x, y in corners),
                "fill": f"rgb({gray},{gray},{gray})",
            },
        )

    layer = ET.SubElement(
        root,
        qname("g"),
        {"data-vektoryum-source-alpha-reconstruction": "painter-luminance-v1"},
    )
    ET.SubElement(
        layer,
        qname("use"),
        {"href": f"#{paint_id}", "mask": f"url(#{mask_id})"},
    )
    return root, {
        "reconstruction_loop_count": int(len(loops)),
        "candidate_support_expanded_geometry_count": int(expanded_count),
    }


def validate_painter_reconstruction(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    grid_alpha: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
) -> dict[str, Any]:
    """Fail-closed dual-scale alpha validation with unchanged thresholds.

    The direct gate renders once at the reconstruction's native grid — where
    the geometry is pixel-aligned and must reproduce the staged alpha plane —
    and compares the bounded evaluation view through the same INTER_AREA
    downscale the source reference uses, so both sides of the comparison pass
    through identical filters. Thresholds are the existing image-class hard
    gates; nothing is loosened. FinalArtifactEvaluator then independently
    confirms the alpha plane against the full-resolution source, exactly like
    the established dual contract, and appearance/topology/seam stay owned by
    the following real TransformJournal parent-delta stage.
    """
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.alpha_candidate_knockout import _source_rgb_on_white  # noqa: PLC0415
    from app.final_artifact_evaluator import (  # noqa: PLC0415
        _structure_check,
        _thresholds,
        evaluate_final_svg,
    )

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    iou_min = float(thresholds["alpha_iou_min"])
    mae_max = float(thresholds["alpha_mae_max"])

    grid_height, grid_width = grid_alpha.shape
    native = render_svg_to_rgba(candidate_path, grid_width, grid_height)
    if native is None:
        raise RuntimeError("source_alpha_candidate_painter_render_unmeasured")
    if native.shape[:2] != (grid_height, grid_width):
        native = resize_rgba(native, grid_width, grid_height)
    native_metrics = alpha_plane_metrics(grid_alpha, native[:, :, 3])
    if float(native_metrics["alpha_iou"]) < iou_min:
        raise RuntimeError(
            "source_alpha_candidate_painter_native_iou_gate_failed:"
            f"{native_metrics['alpha_iou']:.6f}<{iou_min}"
        )
    if float(native_metrics["alpha_mae"]) > mae_max:
        raise RuntimeError(
            "source_alpha_candidate_painter_native_mae_gate_failed:"
            f"{native_metrics['alpha_mae']:.6f}>{mae_max}"
        )

    source_height, source_width = source_rgba_full.shape[:2]
    eval_scale = min(1.0, _PAINTER_EVAL_SIDE / float(max(source_width, source_height)))
    eval_width = max(1, int(round(source_width * eval_scale)))
    eval_height = max(1, int(round(source_height * eval_scale)))
    source_eval = resize_rgba(source_rgba_full, eval_width, eval_height)
    rendered_eval_alpha = cv2.resize(
        native[:, :, 3], (eval_width, eval_height), interpolation=cv2.INTER_AREA
    )
    direct_metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered_eval_alpha)
    if float(direct_metrics["alpha_iou"]) < iou_min:
        raise RuntimeError(
            "source_alpha_candidate_painter_iou_gate_failed:"
            f"{direct_metrics['alpha_iou']:.6f}<{iou_min}"
        )
    if float(direct_metrics["alpha_mae"]) > mae_max:
        raise RuntimeError(
            "source_alpha_candidate_painter_mae_gate_failed:"
            f"{direct_metrics['alpha_mae']:.6f}>{mae_max}"
        )

    report = evaluate_final_svg(
        candidate_path,
        _source_rgb_on_white(source_rgba_full),
        source_alpha=source_rgba_full[:, :, 3],
        image_class=image_class,
        required_metrics={"alpha_fidelity"},
    )
    alpha_group = report.metrics.get("G_gradient_alpha") or {}
    evaluator_alpha_iou = alpha_group.get("alpha_iou")
    evaluator_alpha_mae = alpha_group.get("alpha_mae")
    if evaluator_alpha_iou is None or evaluator_alpha_mae is None:
        raise RuntimeError(
            "source_alpha_candidate_painter_evaluator_rejected:alpha_plane_unmeasured"
        )
    plane_failure_codes = [
        code for code in report.hard_fail_codes if code in _ALPHA_PLANE_FAILURE_CODES
    ]
    if float(evaluator_alpha_iou) < iou_min and "alpha_iou_below_min" not in plane_failure_codes:
        plane_failure_codes.append("alpha_iou_below_min")
    if float(evaluator_alpha_mae) > mae_max and "alpha_mae_above_max" not in plane_failure_codes:
        plane_failure_codes.append("alpha_mae_above_max")
    if plane_failure_codes:
        raise RuntimeError(
            "source_alpha_candidate_painter_evaluator_rejected:"
            + ",".join(plane_failure_codes)
        )

    structure, _messages, structure_codes, root = _structure_check(
        Path(candidate_path).read_bytes()
    )
    if structure_codes or root is None:
        raise RuntimeError(
            "source_alpha_candidate_painter_structure_failed:"
            + ",".join(structure_codes or ["parse_failed"])
        )
    after_counts = (
        int(structure.get("path_count") or 0),
        int(structure.get("node_count") or 0),
    )
    if after_counts != parent_counts:
        raise RuntimeError(
            "source_alpha_candidate_painter_candidate_geometry_changed:"
            f"{parent_counts[0]}/{parent_counts[1]}->"
            f"{after_counts[0]}/{after_counts[1]}"
        )

    return {
        "painter_native_alpha_iou": float(native_metrics["alpha_iou"]),
        "painter_native_alpha_mae": float(native_metrics["alpha_mae"]),
        "source_truth_alpha_iou": float(direct_metrics["alpha_iou"]),
        "source_truth_alpha_mae": float(direct_metrics["alpha_mae"]),
        "source_truth_source_coverage": float(direct_metrics["source_coverage"]),
        "source_truth_render_coverage": float(direct_metrics["render_coverage"]),
        "source_truth_alignment": "native_render_inter_area_downscale",
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": float(evaluator_alpha_iou),
        "final_evaluator_alpha_mae": float(evaluator_alpha_mae),
        "final_evaluator_alpha_source_resolution": f"{source_width}x{source_height}",
        "appearance_regression_authority": "transform_journal_parent_delta",
        "non_alpha_regression_authority": "transform_journal_parent_delta",
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }


def apply_candidate_painter_reconstruction(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Build, validate and atomically publish the painter reconstruction."""
    from app.alpha_candidate_knockout import (  # noqa: PLC0415
        _comparison_canvas_candidate,
        _local_name,
        _path_node_counts,
        _render_root,
        _viewbox,
        _write_tree_to_temp,
    )
    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415
    from app.alpha_svg_mask import _quantize_alpha  # noqa: PLC0415

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    ET.register_namespace("", _SVG_NS)
    original_root = ET.fromstring(before_bytes)
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_candidate_painter_root_not_svg")
    parent_counts = _path_node_counts(original_root)
    _view_x, _view_y, view_width, view_height = _viewbox(original_root)

    grid_scale = min(
        1.0, _PAINTER_GRID_MAX_SIDE / float(max(view_width, view_height))
    )
    grid_width = max(1, int(round(view_width * grid_scale)))
    grid_height = max(1, int(round(view_height * grid_scale)))
    grid_rgba = _rgba_from_source_at_size(source, (grid_width, grid_height))
    grid_alpha = np.asarray(grid_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(grid_alpha == 255)):
        raise RuntimeError("source_alpha_candidate_painter_opaque_source")
    if not np.any(grid_alpha > 0):
        raise RuntimeError("source_alpha_candidate_painter_empty_source")
    quantized, opacity_by_level = _quantize_alpha(grid_alpha)

    collapsed = _render_root(original_root, grid_width, grid_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_painter_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_painter_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    canvas = _comparison_canvas_candidate(
        original_root, grid_rgba, grid_width, grid_height
    )
    limits = _journal_limits(original_root, before_size)

    with Image.open(source) as source_image:
        source_size = source_image.size
    source_rgba_full = _rgba_from_source_at_size(source, source_size)

    last_error: RuntimeError | None = None
    for stroke_width in _PAINTER_STROKE_PIXELS:
        candidate_root, geometry = build_painter_reconstruction_tree(
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
                    "source_alpha_candidate_painter_byte_budget_rejected:"
                    f"{projected_size}>{limits['byte_limit']}"
                )
                continue
            try:
                validation = validate_painter_reconstruction(
                    temporary,
                    source_rgba_full,
                    grid_alpha,
                    mode,
                    parent_counts,
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return {
            "status": "accepted",
            "applied": True,
            "schema": "rfv3d2-candidate-painter-reconstruction-v1",
            "before_byte_size": int(before_size),
            "after_byte_size": int(target.stat().st_size),
            "mask_encoding": "candidate_painter_luminance_mask",
            "candidate_support_stroke_width_pixels": float(stroke_width),
            "candidate_geometry_preserved": True,
            "candidate_path_data_preserved": True,
            "trace_rgb_bytes_preserved": True,
            "candidate_identity_preserved": True,
            "painter_grid_width": int(grid_width),
            "painter_grid_height": int(grid_height),
            "preflight_parent_path_count": int(parent_counts[0]),
            "preflight_parent_node_count": int(parent_counts[1]),
            "preflight_path_limit": int(limits["path_limit"]),
            "preflight_node_limit": int(limits["node_limit"]),
            "preflight_byte_limit": int(limits["byte_limit"]),
            **geometry,
            **validation,
        }

    raise last_error or RuntimeError(
        "source_alpha_candidate_painter_no_admissible_reconstruction"
    )
