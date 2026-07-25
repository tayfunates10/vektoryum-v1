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
_DENSE_POLYGON_MIN_RECTANGLES = 32
_DENSE_POLYGON_MAX_SIDE = 512
_DENSE_POLYGON_MAX_LEVELS = 4


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


def _dense_polygon_plan(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> dict[str, Any] | None:
    """Plan exact level polygons before any verbose SVG rect nodes are created."""
    levels = sorted(int(level) for level in opacity_by_level)
    height, width = quantized.shape
    if (
        len(levels) < 2
        or len(levels) > _DENSE_POLYGON_MAX_LEVELS
        or max(width, height) > _DENSE_POLYGON_MAX_SIDE
    ):
        return None

    from app.alpha_candidate_support_compact import _union_polygon_points  # noqa: PLC0415
    from app.alpha_svg_mask import _merged_rectangles_by_level  # noqa: PLC0415

    rectangles_by_level = _merged_rectangles_by_level(quantized)
    rectangle_count = sum(len(items) for items in rectangles_by_level.values())
    if rectangle_count < _DENSE_POLYGON_MIN_RECTANGLES:
        return None

    layers: list[dict[str, Any]] = []
    polygon_count = 0
    emitted_rectangle_count = 0
    for level in levels:
        rectangles = list(rectangles_by_level.get(level) or [])
        if not rectangles:
            continue
        if len(rectangles) >= 8:
            points = _union_polygon_points(rectangles, width, height)
            if not points:
                return None
            layers.append({
                "level": int(level),
                "opacity": float(opacity_by_level[level]),
                "kind": "polygon",
                "points": points,
                "source_rectangle_count": int(len(rectangles)),
            })
            polygon_count += 1
        else:
            layers.append({
                "level": int(level),
                "opacity": float(opacity_by_level[level]),
                "kind": "rectangles",
                "rectangles": rectangles,
                "source_rectangle_count": int(len(rectangles)),
            })
            emitted_rectangle_count += len(rectangles)

    if not layers or polygon_count == 0:
        return None
    return {
        "layers": layers,
        "source_rectangle_count": int(rectangle_count),
        "polygon_count": int(polygon_count),
        "emitted_rectangle_count": int(emitted_rectangle_count),
    }


def _apply_dense_polygon_alpha(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Publish dense partial alpha as polygons without allocating rect XML first."""
    from app.alpha_candidate_knockout import _path_node_counts  # noqa: PLC0415
    from app.alpha_mask_budget import current_alpha_mask_encoding, _journal_limits  # noqa: PLC0415
    from app.alpha_svg_mask import (  # noqa: PLC0415
        _ALPHA_MASK_MODES,
        _MODE_IMAGE_CLASS,
        _UNDERLAY_STROKE_PIXELS,
        _alpha_diagnostics,
        _alpha_passes,
        _atomic_write_tree,
        _coverage_underlay,
        _quantize_alpha,
        _render_mask_only,
        _strip_content_alpha,
        _viewbox,
    )
    from app.final_artifact_evaluator import _thresholds  # noqa: PLC0415

    target = Path(svg_path)
    source = Path(source_path)
    if mode not in _ALPHA_MASK_MODES or current_alpha_mask_encoding() != "rect":
        raise RuntimeError("source_alpha_dense_polygon_not_applicable_context")

    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    ET.register_namespace("", _SVG_NS)
    tree = ET.ElementTree(ET.fromstring(before_bytes))
    root = tree.getroot()
    if _local_name(str(root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_dense_polygon_root_not_svg")

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_width = max(1, int(round(view_width)))
    raster_height = max(1, int(round(view_height)))
    if (
        abs(view_width - raster_width) > 1e-9
        or abs(view_height - raster_height) > 1e-9
        or max(raster_width, raster_height) > _DENSE_POLYGON_MAX_SIDE
    ):
        raise RuntimeError("source_alpha_dense_polygon_coordinate_contract")

    source_rgba = _rgba_from_source_at_size(source, (raster_width, raster_height))
    source_alpha = np.asarray(source_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(source_alpha == 255)) or not bool(np.any(source_alpha > 0)):
        raise RuntimeError("source_alpha_dense_polygon_not_partial")
    quantized, opacity_by_level = _quantize_alpha(source_alpha)
    plan = _dense_polygon_plan(quantized, opacity_by_level)
    if plan is None:
        raise RuntimeError("source_alpha_dense_polygon_plan_unavailable")

    eval_scale = min(1.0, 512.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    candidate_render = render_svg_to_rgba(target, eval_width, eval_height)
    if candidate_render is None:
        raise RuntimeError("source_alpha_dense_polygon_parent_unmeasured")
    candidate_support = alpha_plane_metrics(
        source_eval[:, :, 3], candidate_render[:, :, 3]
    )
    del candidate_render

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    defs = next(
        (child for child in list(root) if _local_name(str(child.tag)) == "defs"),
        None,
    )
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)
    for child in list(defs):
        if child.get("id") == _MASK_ID:
            defs.remove(child)

    mask = ET.SubElement(
        defs,
        qname("mask"),
        {
            "id": _MASK_ID,
            "maskUnits": "userSpaceOnUse",
            "maskContentUnits": "userSpaceOnUse",
            "x": f"{view_x:g}",
            "y": f"{view_y:g}",
            "width": f"{view_width:g}",
            "height": f"{view_height:g}",
            "style": "mask-type:alpha",
        },
    )
    content = ET.SubElement(mask, qname("g"))
    sx = view_width / float(raster_width)
    sy = view_height / float(raster_height)
    if not (
        abs(view_x) < 1e-9
        and abs(view_y) < 1e-9
        and abs(sx - 1.0) < 1e-9
        and abs(sy - 1.0) < 1e-9
    ):
        content.set(
            "transform",
            f"translate({view_x:g} {view_y:g}) scale({sx:.12g} {sy:.12g})",
        )

    group_count = 0
    polygon_count = 0
    rectangle_count = 0
    for layer in plan["layers"]:
        attributes = {"fill": "#ffffff"}
        opacity = float(layer["opacity"])
        if opacity < 1.0:
            attributes["fill-opacity"] = f"{opacity:.8f}".rstrip("0").rstrip(".")
        group = ET.SubElement(content, qname("g"), attributes)
        if layer["kind"] == "polygon":
            ET.SubElement(
                group,
                qname("polygon"),
                {"points": str(layer["points"]), "fill-rule": "evenodd"},
            )
            polygon_count += 1
        else:
            for x, y, width, height in layer["rectangles"]:
                if width <= 0 or height <= 0:
                    continue
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
                rectangle_count += 1
        if len(group):
            group_count += 1
        else:
            content.remove(group)
    if group_count == 0 or polygon_count == 0:
        raise RuntimeError("source_alpha_dense_polygon_empty_geometry")

    mask_only_render = _render_mask_only(root, defs, eval_width, eval_height)
    if mask_only_render is None:
        raise RuntimeError("source_alpha_dense_polygon_mask_unmeasured")
    mask_only_metrics = alpha_plane_metrics(
        source_eval[:, :, 3], mask_only_render[:, :, 3]
    )
    del mask_only_render

    protected = {"defs", "title", "desc", "metadata", "style"}
    movable = [
        child
        for child in list(root)
        if _local_name(str(child.tag)) not in protected
    ]
    if not movable:
        raise RuntimeError("source_alpha_dense_polygon_no_renderable_content")
    wrapper = ET.Element(
        qname("g"),
        {
            "mask": f"url(#{_MASK_ID})",
            "data-vektoryum-source-alpha": "vector-mask-v1",
        },
    )
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        wrapper.append(child)
    root.append(wrapper)
    _atomic_write_tree(tree, target)

    rendered = render_svg_to_rgba(target, eval_width, eval_height)
    if rendered is None:
        raise RuntimeError("source_alpha_dense_polygon_render_unmeasured")
    rendered_alpha = np.asarray(rendered[:, :, 3], dtype=np.uint8).copy()
    metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered_alpha)
    del rendered

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    underlay_width_pixels = 0.0
    underlay_clone_count = 0
    if not _alpha_passes(metrics, thresholds):
        for width_pixels in _UNDERLAY_STROKE_PIXELS:
            for child in list(wrapper):
                if child.get("data-vektoryum-alpha-coverage-underlay"):
                    wrapper.remove(child)
            width_user = width_pixels * max(sx, sy)
            underlay, clone_count = _coverage_underlay(movable, width_user)
            wrapper.insert(0, underlay)
            _atomic_write_tree(tree, target)
            attempted = render_svg_to_rgba(target, eval_width, eval_height)
            if attempted is None:
                continue
            attempted_alpha = np.asarray(attempted[:, :, 3], dtype=np.uint8).copy()
            attempted_metrics = alpha_plane_metrics(
                source_eval[:, :, 3], attempted_alpha
            )
            del attempted
            if _alpha_passes(attempted_metrics, thresholds):
                metrics = attempted_metrics
                rendered_alpha = attempted_alpha
                underlay_width_pixels = float(width_pixels)
                underlay_clone_count = int(clone_count)
                break
        else:
            for child in list(wrapper):
                if child.get("data-vektoryum-alpha-coverage-underlay"):
                    wrapper.remove(child)
            _atomic_write_tree(tree, target)

    if float(metrics["alpha_iou"]) < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_dense_polygon_iou_gate_failed:"
            f"{metrics['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if float(metrics["alpha_mae"]) > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_dense_polygon_mae_gate_failed:"
            f"{metrics['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )

    limits = _journal_limits(ET.fromstring(before_bytes), before_size)
    after_root = ET.parse(target).getroot()
    actual_paths, actual_nodes = _path_node_counts(after_root)
    after_size = int(target.stat().st_size)
    if after_size > int(limits["byte_limit"]):
        raise RuntimeError(
            "source_alpha_dense_polygon_byte_budget_rejected:"
            f"{after_size}>{limits['byte_limit']}"
        )
    if actual_paths > int(limits["path_limit"]):
        raise RuntimeError(
            "source_alpha_dense_polygon_path_budget_rejected:"
            f"{actual_paths}>{limits['path_limit']}"
        )
    if actual_nodes > int(limits["node_limit"]):
        raise RuntimeError(
            "source_alpha_dense_polygon_node_budget_rejected:"
            f"{actual_nodes}>{limits['node_limit']}"
        )

    after_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "status": "accepted",
        "applied": True,
        "schema": "rfv3d2-source-alpha-vector-mask-v1",
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "before_byte_size": int(before_size),
        "after_byte_size": int(after_size),
        "mask_encoding": "hybrid_polygon",
        "mask_path_count": 0,
        "mask_group_count": int(group_count),
        "mask_rectangle_count": int(rectangle_count),
        "mask_polygon_count": int(polygon_count),
        "dense_polygon_source_rectangle_count": int(plan["source_rectangle_count"]),
        "mask_raster_width": int(raster_width),
        "mask_raster_height": int(raster_height),
        "alpha_level_count": int(group_count),
        "threshold_image_class": image_class,
        "alpha_iou": float(metrics["alpha_iou"]),
        "alpha_mae": float(metrics["alpha_mae"]),
        "source_coverage": float(metrics["source_coverage"]),
        "render_coverage": float(metrics["render_coverage"]),
        "mask_only_alpha_iou": float(mask_only_metrics["alpha_iou"]),
        "mask_only_alpha_mae": float(mask_only_metrics["alpha_mae"]),
        "candidate_support_iou": float(candidate_support["alpha_iou"]),
        "candidate_support_mae": float(candidate_support["alpha_mae"]),
        "coverage_underlay_width_pixels": float(underlay_width_pixels),
        "coverage_underlay_clone_count": int(underlay_clone_count),
        "preflight_parent_path_count": int(limits["parent_path_count"]),
        "preflight_parent_node_count": int(limits["parent_node_count"]),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_node_limit": int(limits["node_limit"]),
        "preflight_byte_limit": int(limits["byte_limit"]),
    }
    report.update(_alpha_diagnostics(source_eval[:, :, 3], rendered_alpha))
    return report


def make_dense_polygon_alpha_first(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Try the exact dense-polygon fast path before heavier alpha candidates."""
    if getattr(guarded_builder, "__vektoryum_dense_polygon_alpha_first__", False):
        return guarded_builder

    @wraps(guarded_builder)
    def dense_first(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        parent = target.read_bytes()
        try:
            return _apply_dense_polygon_alpha(target, Path(source_path), mode)
        except Exception as exc:  # noqa: BLE001 - byte-identical fallback is authoritative
            if target.read_bytes() != parent:
                target.write_bytes(parent)
            report = guarded_builder(target, Path(source_path), mode)
            if isinstance(report, dict):
                report = dict(report)
                report["dense_polygon_alpha"] = {
                    "status": "not_applicable_fallback_used",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            return report

    dense_first.__vektoryum_dense_polygon_alpha_first__ = True
    return dense_first


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
        target = Path(svg_path)
        source = Path(source_path)
        report: dict[str, Any]
        if encoding == "rect":
            parent = target.read_bytes()
            try:
                report = _apply_dense_polygon_alpha(target, source, mode)
            except Exception as exc:  # noqa: BLE001 - established builder remains authoritative
                if target.read_bytes() != parent:
                    target.write_bytes(parent)
                report = rect_builder(target, source, mode)
                if isinstance(report, dict):
                    report = dict(report)
                    report["dense_polygon_alpha"] = {
                        "status": "not_applicable_fallback_used",
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
        else:
            report = rect_builder(target, source, mode)
        if not report.get("applied"):
            report["mask_encoding"] = "not_applicable"
            return report
        if encoding == "rect":
            # Üretici, ikili (tek tam-opak seviye) maskeyi render-eş bir clipPath
            # veya yoğun kısmî alfayı path büyütmeyen bir polygon maskesi olarak
            # yayınlamış olabilir; bu kompakt alt-kodlamaları koru.
            if report.get("mask_encoding") not in {"clip", "hybrid_polygon"}:
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
