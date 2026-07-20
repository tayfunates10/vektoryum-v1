"""Candidate-paint support reconstruction for alpha-boundary undercoverage.

This is the final narrow RFV-3D2 fallback. It runs only after the ordinary
vector mask and the comparison-canvas knockout both fail the unchanged alpha
IoU/MAE gates. The selected candidate's path data and candidate identity remain
unchanged. Existing filled geometry receives the smallest measured same-color
stroke (painted behind its fill) and is clipped by the exact source-alpha
reconstruction. This creates missing boundary support without introducing a
new raster, image element, threshold, or evaluator exception.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import (
    _alpha_encodings,
    _build_reconstruction_tree,
    _comparison_canvas_candidate,
    _local_name,
    _path_node_counts,
    _render_root,
    _validate_reconstruction,
    _viewbox,
    _write_tree_to_temp,
)
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import resize_rgba

_SUPPORT_STROKE_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
_SUPPORT_FAILURE_PREFIXES = (
    "source_alpha_candidate_knockout_iou_gate_failed:",
    "source_alpha_candidate_knockout_mae_gate_failed:",
)
_GEOMETRY_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline"}
_ALPHA_STYLE_NAMES = {"opacity", "fill-opacity", "stroke-opacity"}


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
        # Candidate opacity was already stripped by the base reconstruction.
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


def apply_candidate_support_reconstruction(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Choose the smallest admissible candidate-paint support expansion."""
    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    original_root = ET.fromstring(before_bytes)
    parent_counts = _path_node_counts(original_root)
    _view_x, _view_y, view_width, view_height = _viewbox(original_root)

    scale = min(1.0, 1600.0 / float(max(view_width, view_height)))
    raster_width = max(1, int(round(view_width * scale)))
    raster_height = max(1, int(round(view_height * scale)))
    source_rgba = _rgba_from_source_at_size(source, (raster_width, raster_height))
    source_alpha = np.asarray(source_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(source_alpha == 255)):
        raise RuntimeError("source_alpha_candidate_support_opaque_source")
    if not np.any(source_alpha > 0):
        raise RuntimeError("source_alpha_candidate_support_empty_source")

    eval_scale = min(1.0, 256.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    collapsed = _render_root(original_root, eval_width, eval_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_support_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_support_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    canvas = _comparison_canvas_candidate(
        original_root, source_eval, eval_width, eval_height
    )
    limits = _journal_limits(original_root, before_size)

    with Image.open(source) as source_image:
        source_size = source_image.size
    source_rgba_full = _rgba_from_source_at_size(source, source_size)
    last_error: RuntimeError | None = None

    for encoding_name, quantized, opacity_by_level in _alpha_encodings(source_alpha):
        for stroke_width in _SUPPORT_STROKE_PIXELS:
            candidate_root, geometry = _build_reconstruction_tree(
                original_root, canvas, quantized, opacity_by_level
            )
            paint = next(
                (
                    element
                    for element in candidate_root.iter()
                    if element.get("data-vektoryum-alpha-candidate-paint")
                    == "preserved-v1"
                ),
                None,
            )
            if paint is None:
                raise RuntimeError("source_alpha_candidate_support_paint_missing")
            expanded_count = _expand_candidate_paint(paint, stroke_width)
            if expanded_count <= 0:
                raise RuntimeError("source_alpha_candidate_support_no_fill_geometry")

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
                "schema": "rfv3d2-candidate-support-reconstruction-v1",
                "before_byte_size": int(before_size),
                "after_byte_size": int(target.stat().st_size),
                "mask_encoding": "candidate_support_reconstruction",
                "reconstruction_alpha_encoding": encoding_name,
                "candidate_support_stroke_width_pixels": float(stroke_width),
                "candidate_support_expanded_geometry_count": int(expanded_count),
                "candidate_geometry_preserved": True,
                "candidate_path_data_preserved": True,
                "trace_rgb_bytes_preserved": True,
                "candidate_identity_preserved": True,
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
