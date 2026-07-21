"""Renderer-stable source-alpha reconstruction from selected candidate geometry.

The ordinary RFV-3D2 vector mask remains the preferred path. Some tracers emit
an opaque comparison-canvas path and some SVG renderers disagree on a large,
nested alpha mask. When the unchanged alpha hard gate rejects that artifact,
this module may remove only a renderer-proven opaque comparison-canvas element
and reconstruct source alpha with clip-stratified ``<use>`` instances of the
same selected candidate paint.

The candidate paths, path-command nodes, gradients, trace raster and candidate
identity are not regenerated. Source alpha controls only clipping and group
opacity. Every write is byte-transactional and must pass both the direct RGBA
source-truth gate and the exact FinalArtifactEvaluator contract without changing
any existing threshold or TransformJournal complexity allowance.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
_MAX_RECONSTRUCTION_SIDE = 1600
_MAX_EVAL_SIDE = 512
_PROTECTED_ROOT_TAGS = {"defs", "title", "desc", "metadata", "style"}
_RENDERABLE_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "g"}
_ALPHA_FAILURE_PREFIXES = (
    "source_alpha_mask_iou_gate_failed:",
    "source_alpha_mask_mae_gate_failed:",
    "source_alpha_compact_iou_gate_failed:",
    "source_alpha_compact_mae_gate_failed:",
)
_PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1] if "}" in name else name.split(":")[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    from app.alpha_svg_mask import _viewbox as existing_viewbox  # noqa: PLC0415

    return existing_viewbox(root)


def _path_node_counts(root: ET.Element) -> tuple[int, int]:
    paths = [
        element for element in root.iter()
        if _local_name(str(element.tag)).lower() == "path"
    ]
    return len(paths), sum(
        len(_PATH_COMMAND.findall(str(path.get("d") or ""))) for path in paths
    )


def _source_rgb_on_white(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        np.rint(arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)),
        0,
        255,
    ).astype(np.uint8)


def _render_root(root: ET.Element, width: int, height: int) -> np.ndarray | None:
    with tempfile.TemporaryDirectory(prefix="vektoryum-alpha-knockout-probe-") as directory:
        path = Path(directory) / "probe.svg"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return render_svg_to_rgba(path, int(width), int(height))


def _probe_root(root: ET.Element, candidate: ET.Element) -> ET.Element:
    probe = ET.Element(root.tag, dict(root.attrib))
    for child in list(root):
        if _local_name(str(child.tag)).lower() in _PROTECTED_ROOT_TAGS:
            probe.append(copy.deepcopy(child))
    probe.append(copy.deepcopy(candidate))
    return probe


def _comparison_canvas_candidate(
    root: ET.Element,
    source_eval: np.ndarray,
    eval_width: int,
    eval_height: int,
) -> ET.Element:
    """Return one direct child proven to be the opaque white trace canvas."""
    source_alpha = np.asarray(source_eval[:, :, 3], dtype=np.uint8)
    transparent = source_alpha == 0
    transparent_count = int(np.count_nonzero(transparent))
    if transparent_count == 0:
        raise RuntimeError("source_alpha_candidate_knockout_no_transparent_source")

    best: tuple[float, ET.Element] | None = None
    for child in list(root):
        local = _local_name(str(child.tag)).lower()
        if local not in _RENDERABLE_TAGS:
            continue
        rendered = _render_root(_probe_root(root, child), eval_width, eval_height)
        if rendered is None:
            continue
        alpha = np.asarray(rendered[:, :, 3], dtype=np.uint8)
        soft_coverage = float(alpha.astype(np.float32).mean() / 255.0)
        if soft_coverage < 0.97:
            continue
        visible = (alpha > 0) & transparent
        if int(np.count_nonzero(visible)) < max(16, int(0.90 * transparent_count)):
            continue
        rgb = np.asarray(rendered[:, :, :3], dtype=np.uint8)
        weights = alpha[visible].astype(np.float32) / 255.0
        if not np.any(weights > 0):
            continue
        weighted_rgb = np.average(rgb[visible].astype(np.float32), axis=0, weights=weights)
        if float(np.min(weighted_rgb)) < 240.0:
            continue
        score = soft_coverage + float(np.min(weighted_rgb)) / 2550.0
        if best is None or score > best[0]:
            best = (score, child)

    if best is None:
        raise RuntimeError("source_alpha_candidate_knockout_canvas_not_proven")
    return best[1]


def _unique_id(root: ET.Element, base: str) -> str:
    used = {str(element.get("id")) for element in root.iter() if element.get("id")}
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


def _alpha_encodings(alpha: np.ndarray):
    """Yield exact alpha first, then the established 128-level fallback."""
    exact = np.asarray(alpha, dtype=np.uint8).copy()
    exact_levels = {
        int(value): int(value) / 255.0
        for value in np.unique(exact)
        if int(value) > 0
    }
    yield "exact", exact, exact_levels

    from app.alpha_mask_budget import _quantize_alpha  # noqa: PLC0415

    quantized, opacity_by_level = _quantize_alpha(exact)
    if not np.array_equal(quantized, exact):
        yield "quantized_128", quantized, opacity_by_level


def _build_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> tuple[ET.Element, dict[str, int]]:
    from app.alpha_svg_mask import _merged_rectangles_by_level, _strip_content_alpha  # noqa: PLC0415

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

    knockout_archive = ET.SubElement(
        defs,
        qname("g"),
        {
            "data-vektoryum-candidate-geometry-knockout": "comparison-canvas-v1",
            "display": "none",
        },
    )
    knockout_archive.append(target_canvas)

    paint_id = _unique_id(root, "vektoryum-alpha-candidate-paint")
    paint = ET.SubElement(
        defs,
        qname("g"),
        {"id": paint_id, "data-vektoryum-alpha-candidate-paint": "preserved-v1"},
    )
    movable = [
        child for child in list(root)
        if child is not defs and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not movable:
        raise RuntimeError("source_alpha_candidate_knockout_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)

    rectangles = _merged_rectangles_by_level(quantized)
    if not rectangles:
        raise RuntimeError("source_alpha_candidate_knockout_empty_source")

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_height, raster_width = quantized.shape
    sx = view_width / float(raster_width)
    sy = view_height / float(raster_height)

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
                "data-vektoryum-alpha-level": str(level),
            },
        )
        for x, y, width, height in level_rectangles:
            if width <= 0 or height <= 0:
                continue
            # Resvg supports a stricter clipPath subset than Cairo. Emit exact
            # user-space rectangles directly instead of nesting a transformed
            # group inside clipPath so both evaluator renderers agree.
            ET.SubElement(
                clip,
                qname("rect"),
                {
                    "x": f"{view_x + x * sx:.12g}",
                    "y": f"{view_y + y * sy:.12g}",
                    "width": f"{width * sx:.12g}",
                    "height": f"{height * sy:.12g}",
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
                "data-vektoryum-source-alpha-reconstruction": "clip-use-v1",
                "data-vektoryum-alpha-level": str(level),
            },
        )
        ET.SubElement(
            layer,
            qname("use"),
            {
                "href": f"#{paint_id}",
                f"{{{_XLINK_NS}}}href": f"#{paint_id}",
                "clip-path": f"url(#{clip_id})",
            },
        )
        use_count += 1

    if rectangle_count == 0 or use_count == 0:
        raise RuntimeError("source_alpha_candidate_knockout_no_reconstruction")
    return root, {
        "reconstruction_clip_count": int(clip_count),
        "reconstruction_rectangle_count": int(rectangle_count),
        "reconstruction_use_count": int(use_count),
    }


def _write_tree_to_temp(root: ET.Element, target: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".alpha-knockout.svg",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _validate_reconstruction(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
) -> dict[str, Any]:
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import _structure_check, _thresholds, evaluate_final_svg  # noqa: PLC0415

    source_height, source_width = source_rgba_full.shape[:2]
    eval_scale = min(1.0, _MAX_EVAL_SIDE / float(max(source_width, source_height)))
    eval_width = max(1, int(round(source_width * eval_scale)))
    eval_height = max(1, int(round(source_height * eval_scale)))
    source_eval = resize_rgba(source_rgba_full, eval_width, eval_height)
    rendered = render_svg_to_rgba(candidate_path, eval_width, eval_height)
    if rendered is None:
        raise RuntimeError("source_alpha_candidate_knockout_render_unmeasured")
    if rendered.shape[:2] != (eval_height, eval_width):
        rendered = resize_rgba(rendered, eval_width, eval_height)
    direct_metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered[:, :, 3])

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    if float(direct_metrics["alpha_iou"]) < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_iou_gate_failed:"
            f"{direct_metrics['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if float(direct_metrics["alpha_mae"]) > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_mae_gate_failed:"
            f"{direct_metrics['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )

    source_rgb = _source_rgb_on_white(source_rgba_full)
    report = evaluate_final_svg(
        candidate_path,
        source_rgb,
        source_alpha=source_rgba_full[:, :, 3],
        image_class=image_class,
        required_metrics={"alpha_fidelity"},
    )
    alpha_group = report.metrics.get("G_gradient_alpha") or {}
    if alpha_group.get("alpha_fidelity_status") != "passed":
        codes = ",".join(report.hard_fail_codes or ["alpha_fidelity_not_passed"])
        raise RuntimeError(f"source_alpha_candidate_knockout_evaluator_rejected:{codes}")
    if report.hard_fail_codes:
        raise RuntimeError(
            "source_alpha_candidate_knockout_evaluator_hard_fail:"
            + ",".join(report.hard_fail_codes)
        )

    structure, _messages, structure_codes, root = _structure_check(candidate_path.read_bytes())
    if structure_codes or root is None:
        raise RuntimeError(
            "source_alpha_candidate_knockout_structure_failed:"
            + ",".join(structure_codes or ["parse_failed"])
        )
    after_counts = (
        int(structure.get("path_count") or 0),
        int(structure.get("node_count") or 0),
    )
    if after_counts != parent_counts:
        raise RuntimeError(
            "source_alpha_candidate_knockout_candidate_geometry_changed:"
            f"{parent_counts[0]}/{parent_counts[1]}->{after_counts[0]}/{after_counts[1]}"
        )

    return {
        "source_truth_alpha_iou": float(direct_metrics["alpha_iou"]),
        "source_truth_alpha_mae": float(direct_metrics["alpha_mae"]),
        "source_truth_source_coverage": float(direct_metrics["source_coverage"]),
        "source_truth_render_coverage": float(direct_metrics["render_coverage"]),
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_iou": float(alpha_group["alpha_iou"]),
        "final_evaluator_alpha_mae": float(alpha_group["alpha_mae"]),
        "final_evaluator_hard_fail_codes": list(report.hard_fail_codes),
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }


def apply_candidate_geometry_knockout(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Replace an opaque trace canvas with clip-stratified source alpha."""
    target = Path(svg_path)
    source = Path(source_path)
    before_sha = _sha256(target)
    before_size = target.stat().st_size
    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)
    original_tree = ET.parse(target)
    original_root = original_tree.getroot()
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_candidate_knockout_root_not_svg")
    parent_counts = _path_node_counts(original_root)

    _view_x, _view_y, view_width, view_height = _viewbox(original_root)
    scale = min(1.0, _MAX_RECONSTRUCTION_SIDE / float(max(view_width, view_height)))
    raster_width = max(1, int(round(view_width * scale)))
    raster_height = max(1, int(round(view_height * scale)))
    source_rgba = _rgba_from_source_at_size(source, (raster_width, raster_height))
    source_alpha = np.asarray(source_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(source_alpha == 255)):
        raise RuntimeError("source_alpha_candidate_knockout_opaque_source")
    if not np.any(source_alpha > 0):
        raise RuntimeError("source_alpha_candidate_knockout_empty_source")

    eval_scale = min(1.0, 256.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    collapsed = _render_root(copy.deepcopy(original_root), eval_width, eval_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_knockout_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_knockout_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    canvas = _comparison_canvas_candidate(original_root, source_eval, eval_width, eval_height)

    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415

    limits = _journal_limits(original_root, before_size)
    last_budget_error: str | None = None
    for encoding_name, quantized, opacity_by_level in _alpha_encodings(source_alpha):
        try:
            candidate_root, geometry = _build_reconstruction_tree(
                original_root, canvas, quantized, opacity_by_level
            )
            temporary = _write_tree_to_temp(candidate_root, target)
            try:
                after_size = temporary.stat().st_size
                if after_size > int(limits["byte_limit"]):
                    last_budget_error = (
                        "source_alpha_candidate_knockout_byte_budget_rejected:"
                        f"{after_size}>{limits['byte_limit']}"
                    )
                    continue
                with Image.open(source) as source_image:
                    source_size = source_image.size
                source_rgba_full = _rgba_from_source_at_size(source, source_size)
                validation = _validate_reconstruction(
                    temporary,
                    source_rgba_full,
                    mode,
                    parent_counts,
                )
                os.replace(temporary, target)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        except RuntimeError as exc:
            if "byte_budget" in str(exc):
                last_budget_error = str(exc)
                continue
            raise

        after_sha = _sha256(target)
        return {
            "status": "accepted",
            "applied": True,
            "schema": "rfv3d2-candidate-geometry-knockout-v1",
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "before_byte_size": int(before_size),
            "after_byte_size": int(target.stat().st_size),
            "threshold_image_class": {
                "geometric_logo": "geometric",
                "minimal_ai": "clean_logo",
                "flat_logo": "clean_logo",
                "logo_color": "clean_logo",
                "photo_poster": "photo",
            }.get(mode, "clean_logo"),
            "mask_encoding": "candidate_geometry_knockout",
            "reconstruction_alpha_encoding": encoding_name,
            "candidate_canvas_knockout_count": 1,
            "candidate_geometry_preserved": True,
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

    raise RuntimeError(last_budget_error or "source_alpha_candidate_knockout_no_admissible_encoding")


def make_candidate_geometry_knockout_fallback(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Retry only a rolled-back exact alpha-gate failure via candidate knockout."""
    if getattr(guarded_builder, "__vektoryum_candidate_knockout_fallback__", False):
        return guarded_builder

    @wraps(guarded_builder)
    def fallback(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        source = Path(source_path)
        try:
            return guarded_builder(target, source, mode)
        except RuntimeError as first_error:
            trigger = str(first_error)
            if not trigger.startswith(_ALPHA_FAILURE_PREFIXES):
                raise

        from app.alpha_mask_budget import _create_atomic_backup, _restore_atomic_backup  # noqa: PLC0415

        backup = _create_atomic_backup(target)
        try:
            report = apply_candidate_geometry_knockout(target, source, mode)
        except BaseException:
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)

        report["mask_fallback_reason"] = "source_alpha_exact_gate_failure"
        report["mask_fallback_trigger"] = trigger
        report["rollback_guard"] = "armed_and_committed"
        return report

    fallback.__vektoryum_candidate_knockout_fallback__ = True
    return fallback
