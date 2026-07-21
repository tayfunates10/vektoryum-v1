"""Adaptive vector encoding for source-alpha masks.

The default builder emits editable ``<rect>`` primitives. When that exact
representation cannot fit the unchanged TransformJournal byte budget, preflight
may authorize a compact, directly emitted contour-path representation only if
the parent artifact has enough existing path/node/byte budget. No gate is
relaxed. The legacy rect-to-path converter remains for direct compatibility
fixtures, but production contour paths are never materialized as oversized rect
XML first.
"""
from __future__ import annotations

import hashlib
import inspect
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_MASK_ID = "vektoryum-source-alpha"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _rect_path(rectangles: list[ET.Element]) -> str:
    commands: list[str] = []
    for rect in rectangles:
        x = rect.get("x", "0")
        y = rect.get("y", "0")
        width = rect.get("width", "0")
        height = rect.get("height", "0")
        commands.append(f"M{x} {y}h{width}v{height}h-{width}Z")
    return " ".join(commands)


def _compact_mask_rectangles(svg_path: Path) -> tuple[int, int]:
    """Replace each alpha-level rect set with one equivalent SVG path."""
    from app.alpha_svg_mask import _atomic_write_tree  # noqa: PLC0415

    ET.register_namespace("", _SVG_NS)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    mask = next(
        (
            element
            for element in root.iter()
            if _local_name(str(element.tag)) == "mask"
            and element.get("id") == _MASK_ID
        ),
        None,
    )
    if mask is None:
        raise RuntimeError("source_alpha_compact_mask_missing")

    path_count = 0
    rectangle_count = 0
    for group in list(mask.iter()):
        if _local_name(str(group.tag)) != "g":
            continue
        if group.get("data-vektoryum-alpha-level") is None:
            continue
        rectangles = [
            child
            for child in list(group)
            if _local_name(str(child.tag)) == "rect"
        ]
        if not rectangles:
            continue
        path_data = _rect_path(rectangles)
        if not path_data:
            raise RuntimeError("source_alpha_compact_path_empty")
        for rect in rectangles:
            group.remove(rect)
        ET.SubElement(group, f"{{{_SVG_NS}}}path", {"d": path_data})
        path_count += 1
        rectangle_count += len(rectangles)

    if path_count == 0 or rectangle_count == 0:
        raise RuntimeError("source_alpha_compact_path_not_generated")
    _atomic_write_tree(tree, svg_path)
    return path_count, rectangle_count


def _validate_compact_alpha(
    svg_path: Path,
    source_path: Path,
    mode: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    raster_width = int(report["mask_raster_width"])
    raster_height = int(report["mask_raster_height"])
    source_rgba = _rgba_from_source_at_size(
        source_path, (raster_width, raster_height)
    )
    eval_scale = min(1.0, 512.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    rendered = render_svg_to_rgba(svg_path, eval_width, eval_height)
    if rendered is None:
        raise RuntimeError("source_alpha_compact_render_unmeasured")
    metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered[:, :, 3])

    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import _thresholds  # noqa: PLC0415

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    if float(metrics["alpha_iou"]) < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_compact_iou_gate_failed:"
            f"{metrics['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if float(metrics["alpha_mae"]) > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_compact_mae_gate_failed:"
            f"{metrics['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )

    report.update({
        "alpha_iou": float(metrics["alpha_iou"]),
        "alpha_mae": float(metrics["alpha_mae"]),
        "source_coverage": float(metrics["source_coverage"]),
        "render_coverage": float(metrics["render_coverage"]),
        "threshold_image_class": image_class,
    })
    return report


def make_adaptive_apply_source_alpha_mask(
    rect_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Use rects by default and compact paths only when preflight authorizes."""
    if getattr(rect_builder, "__vektoryum_adaptive_encoding__", False):
        return rect_builder

    @wraps(rect_builder)
    def adaptive(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        from app.alpha_mask_budget import current_alpha_mask_encoding  # noqa: PLC0415

        encoding = current_alpha_mask_encoding()
        report = rect_builder(Path(svg_path), Path(source_path), mode)
        if not report.get("applied"):
            report["mask_encoding"] = "not_applicable"
            return report
        if encoding == "rect":
            report["mask_encoding"] = "rect"
            return report
        if encoding != "path":
            raise RuntimeError(f"source_alpha_mask_encoding_invalid:{encoding}")

        # The production builder consumes the preflight contour plan directly,
        # avoiding the exact OOM condition this adaptive layer is meant to guard.
        if report.get("mask_encoding") == "path" and int(
            report.get("mask_path_count") or 0
        ) > 0:
            return report

        # Legacy/direct fixtures may still supply a rect mask under a forced path
        # context. Preserve that compatibility without using it in production.
        path_count, rectangle_count = _compact_mask_rectangles(Path(svg_path))
        report["mask_encoding"] = "path"
        report["mask_path_count"] = int(path_count)
        report["mask_rectangle_count"] = int(rectangle_count)
        report["after_byte_size"] = Path(svg_path).stat().st_size
        report["after_sha256"] = hashlib.sha256(
            Path(svg_path).read_bytes()
        ).hexdigest()
        return _validate_compact_alpha(
            Path(svg_path), Path(source_path), mode, report
        )

    adaptive.__vektoryum_adaptive_encoding__ = True
    return adaptive


def _contour_fallback_plan(
    svg_path: Path,
    source_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build and budget-check a contour plan after a rect alpha-gate failure."""
    from defusedxml import ElementTree as SafeET  # noqa: PLC0415
    from app.alpha_mask_budget import (  # noqa: PLC0415
        _FIXED_MARKUP_OVERHEAD,
        _MAX_MASK_SIDE,
        _build_contour_plan,
        _journal_limits,
        _quantize_alpha,
        _viewbox_size,
    )

    target = Path(svg_path)
    before_size = target.stat().st_size
    root = SafeET.fromstring(target.read_bytes())
    limits = _journal_limits(root, before_size)
    width, height = _viewbox_size(root)
    scale = min(1.0, _MAX_MASK_SIDE / float(max(width, height)))
    raster_width = max(1, int(round(width * scale)))
    raster_height = max(1, int(round(height * scale)))
    rgba = _rgba_from_source_at_size(
        Path(source_path), (raster_width, raster_height)
    )
    quantized, opacity_by_level = _quantize_alpha(
        np.asarray(rgba[:, :, 3], dtype=np.uint8)
    )
    plan = _build_contour_plan(quantized, opacity_by_level)
    if plan is None:
        raise RuntimeError("source_alpha_mask_contour_fallback_unavailable")

    path_count_after = limits["parent_path_count"] + int(plan["path_count"])
    path_node_after = limits["parent_node_count"] + int(plan["command_count"])
    path_projected = (
        before_size
        + _FIXED_MARKUP_OVERHEAD
        + int(plan["path_markup_bytes"])
    )
    if not (
        path_count_after <= limits["path_limit"]
        and path_node_after <= limits["node_limit"]
        and path_projected <= limits["byte_limit"]
    ):
        raise RuntimeError(
            "source_alpha_mask_contour_fallback_budget_rejected:"
            f"path_bytes={path_projected}/{limits['byte_limit']},"
            f"path_count={path_count_after}/{limits['path_limit']},"
            f"path_nodes={path_node_after}/{limits['node_limit']}"
        )
    return plan, {
        "fallback_path_projected_byte_size": int(path_projected),
        "fallback_path_count_after": int(path_count_after),
        "fallback_path_node_count_after": int(path_node_after),
        "fallback_path_limit": int(limits["path_limit"]),
        "fallback_node_limit": int(limits["node_limit"]),
        "fallback_byte_limit": int(limits["byte_limit"]),
    }


def make_rect_fidelity_fallback(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Retry a rejected rect mask as a budget-admissible contour mask.

    The ordinary guarded builder remains the first and preferred path. It rolls
    back before raising. Only exact alpha IoU/MAE rejection activates this second
    transaction, and the retry calls the unwrapped production builder under a
    verified path plan so no threshold or journal budget is relaxed.
    """
    if getattr(guarded_builder, "__vektoryum_rect_fidelity_fallback__", False):
        return guarded_builder
    base_builder = inspect.unwrap(guarded_builder)

    @wraps(guarded_builder)
    def fallback(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        source = Path(source_path)
        try:
            return guarded_builder(target, source, mode)
        except RuntimeError as first_error:
            trigger = str(first_error)
            if not trigger.startswith((
                "source_alpha_mask_iou_gate_failed:",
                "source_alpha_mask_mae_gate_failed:",
            )):
                raise

        from app.alpha_mask_budget import (  # noqa: PLC0415
            _ALPHA_MASK_ENCODING,
            _ALPHA_MASK_PLAN,
            _create_atomic_backup,
            _restore_atomic_backup,
        )

        plan, measurements = _contour_fallback_plan(target, source)
        encoding_token = _ALPHA_MASK_ENCODING.set("path")
        plan_token = _ALPHA_MASK_PLAN.set(plan)
        backup = _create_atomic_backup(target)
        try:
            report = base_builder(target, source, mode)
        except BaseException:
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)
        finally:
            _ALPHA_MASK_PLAN.reset(plan_token)
            _ALPHA_MASK_ENCODING.reset(encoding_token)

        report.update(measurements)
        report["mask_encoding"] = "path"
        report["preflight_mask_encoding"] = "rect"
        report["mask_fallback_reason"] = "rect_exact_alpha_gate_failure"
        report["mask_fallback_trigger"] = trigger
        report["rollback_guard"] = "armed_and_committed"
        return report

    fallback.__vektoryum_rect_fidelity_fallback__ = True
    return fallback
