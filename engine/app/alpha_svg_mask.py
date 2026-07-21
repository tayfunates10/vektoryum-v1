"""Final, vector-only source-alpha mask for selected production SVG artifacts."""
from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
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
_MASK_ID = "vektoryum-source-alpha"
_MAX_ALPHA_LEVELS = 128
_MAX_MASK_SIDE = 1600
_MODE_IMAGE_CLASS = {
    "geometric_logo": "geometric",
    "minimal_ai": "clean_logo",
    "flat_logo": "clean_logo",
    "logo_color": "clean_logo",
    "photo_poster": "photo",
}
_ALPHA_MASK_MODES = frozenset(_MODE_IMAGE_CLASS)
_ALPHA_STYLE_NAMES = {"opacity", "fill-opacity", "stroke-opacity"}
_GEOMETRY_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline"}
_UNDERLAY_STROKE_PIXELS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1] if "}" in name else name


def _dimension(value: str | None) -> float | None:
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:px)?\s*",
        str(value or ""),
    )
    if not match:
        return None
    parsed = float(match.group(1))
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = root.get("viewBox") or root.get("viewbox")
    if raw:
        parts = [float(value) for value in re.split(r"[\s,]+", raw.strip()) if value]
        if (
            len(parts) == 4
            and all(np.isfinite(value) for value in parts)
            and parts[2] > 0
            and parts[3] > 0
        ):
            return parts[0], parts[1], parts[2], parts[3]
    width = _dimension(root.get("width"))
    height = _dimension(root.get("height"))
    if width is None or height is None:
        raise RuntimeError("source_alpha_mask_missing_coordinate_contract")
    return 0.0, 0.0, width, height


def _quantize_alpha(alpha: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    values = np.unique(alpha)
    nonzero = values[values > 0]
    if len(nonzero) <= _MAX_ALPHA_LEVELS:
        quantized = alpha.astype(np.uint8, copy=True)
        return quantized, {int(value): int(value) / 255.0 for value in nonzero}

    steps = _MAX_ALPHA_LEVELS - 1
    indexes = np.rint(alpha.astype(np.float32) * steps / 255.0).astype(np.uint8)
    return indexes, {
        int(value): int(value) / float(steps)
        for value in np.unique(indexes)
        if int(value) > 0
    }


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

        current_keys = {(level, x0, x1) for level, x0, x1 in runs}
        for key in list(active):
            if key not in current_keys:
                level, x0, x1 = key
                rect = active.pop(key)
                completed.setdefault(level, []).append(
                    (x0, rect[0], x1 - x0, rect[1] - rect[0])
                )
        for level, x0, x1 in runs:
            key = (level, x0, x1)
            if key in active:
                active[key][1] = y + 1
            else:
                active[key] = [y, y + 1]

    for (level, x0, x1), (y0, y1) in active.items():
        completed.setdefault(level, []).append((x0, y0, x1 - x0, y1 - y0))
    return completed


def _strip_content_alpha(element: ET.Element) -> None:
    """Make the source mask the only alpha truth; avoid multiplying alpha twice."""
    for node in element.iter():
        for name in _ALPHA_STYLE_NAMES:
            node.attrib.pop(name, None)
        style = node.get("style")
        if not style:
            continue
        declarations: list[str] = []
        for declaration in style.split(";"):
            if not declaration.strip():
                continue
            key = declaration.split(":", 1)[0].strip().lower()
            if key not in _ALPHA_STYLE_NAMES:
                declarations.append(declaration.strip())
        if declarations:
            node.set("style", ";".join(declarations))
        else:
            node.attrib.pop("style", None)


def _atomic_write_tree(tree: ET.ElementTree, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=Path(path).parent,
        prefix=f".{Path(path).name}.",
        suffix=".alpha.svg",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _journal_source_rgb(image: Image.Image) -> np.ndarray:
    """Match the pipeline journal's white-composited source RGB contract."""
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[:, :, 3].astype(np.float32)[:, :, None] / 255.0
        return np.clip(
            rgba[:, :, :3].astype(np.float32) * alpha
            + 255.0 * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _style_declarations(style: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in str(style or "").split(";"):
        if ":" not in declaration:
            continue
        key, value = declaration.split(":", 1)
        result[key.strip().lower()] = value.strip()
    return result


def _remove_style_keys(node: ET.Element, keys: set[str]) -> None:
    declarations = _style_declarations(node.get("style"))
    for key in keys:
        declarations.pop(key, None)
    if declarations:
        node.set("style", ";".join(f"{key}:{value}" for key, value in declarations.items()))
    else:
        node.attrib.pop("style", None)


def _stroke_clone_geometry(
    node: ET.Element,
    stroke_width: float,
    inherited_fill: str | None = None,
) -> int:
    declarations = _style_declarations(node.get("style"))
    fill = declarations.get("fill", node.get("fill", inherited_fill))
    if fill is None:
        fill = "#000000"
    local = _local_name(str(node.tag)).lower()
    count = 0
    if local in _GEOMETRY_TAGS and fill.strip().lower() not in {
        "none", "transparent", "rgba(0,0,0,0)",
    }:
        _remove_style_keys(
            node,
            {
                "stroke", "stroke-width", "stroke-linejoin", "stroke-linecap",
                "stroke-dasharray", "stroke-dashoffset", "stroke-opacity",
            },
        )
        node.set("stroke", fill)
        node.set("stroke-width", f"{stroke_width:.8f}".rstrip("0").rstrip("."))
        node.set("stroke-linejoin", "round")
        node.set("stroke-linecap", "round")
        node.attrib.pop("stroke-dasharray", None)
        node.attrib.pop("stroke-dashoffset", None)
        node.attrib.pop("stroke-opacity", None)
        count = 1
    for child in list(node):
        count += _stroke_clone_geometry(child, stroke_width, fill)
    return count


def _coverage_underlay(
    movable: list[ET.Element],
    stroke_width: float,
) -> tuple[ET.Element, int]:
    underlay = ET.Element(
        f"{{{_SVG_NS}}}g",
        {
            "data-vektoryum-alpha-coverage-underlay": "paint-preserving-v1",
            "pointer-events": "none",
        },
    )
    clone_count = 0
    for element in movable:
        cloned = copy.deepcopy(element)
        count = _stroke_clone_geometry(cloned, stroke_width)
        if count:
            underlay.append(cloned)
            clone_count += count
    if clone_count == 0:
        raise RuntimeError("source_alpha_coverage_underlay_no_painted_geometry")
    return underlay, clone_count


def _alpha_passes(metrics: dict[str, float], thresholds: dict[str, Any]) -> bool:
    return bool(
        float(metrics["alpha_iou"]) >= float(thresholds["alpha_iou_min"])
        and float(metrics["alpha_mae"]) <= float(thresholds["alpha_mae_max"])
    )


def _alpha_diagnostics(
    source_alpha: np.ndarray,
    rendered_alpha: np.ndarray,
) -> dict[str, int]:
    source_positive = source_alpha > 0
    rendered_positive = rendered_alpha > 0
    return {
        "missing_source_alpha_pixel_count": int(
            np.count_nonzero(source_positive & ~rendered_positive)
        ),
        "extra_alpha_pixel_count": int(
            np.count_nonzero(~source_positive & rendered_positive)
        ),
    }


def _render_mask_only(
    root: ET.Element,
    defs: ET.Element,
    eval_width: int,
    eval_height: int,
) -> np.ndarray | None:
    probe_root = ET.Element(root.tag, dict(root.attrib))
    probe_root.append(copy.deepcopy(defs))
    probe = ET.SubElement(
        probe_root,
        f"{{{_SVG_NS}}}rect",
        {
            "x": str(_viewbox(root)[0]),
            "y": str(_viewbox(root)[1]),
            "width": str(_viewbox(root)[2]),
            "height": str(_viewbox(root)[3]),
            "fill": "#ffffff",
            "mask": f"url(#{_MASK_ID})",
        },
    )
    del probe
    with tempfile.TemporaryDirectory(prefix="vektoryum-alpha-mask-probe-") as directory:
        path = Path(directory) / "mask-probe.svg"
        ET.ElementTree(probe_root).write(path, encoding="utf-8", xml_declaration=True)
        return render_svg_to_rgba(path, eval_width, eval_height)


def apply_source_alpha_mask(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Attach a vector-only alpha mask and validate it with existing hard gates."""
    svg_path = Path(svg_path)
    before_sha = _sha256(svg_path)
    before_size = svg_path.stat().st_size
    ET.register_namespace("", _SVG_NS)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    if _local_name(str(root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_mask_root_not_svg")

    view_x, view_y, view_width, view_height = _viewbox(root)
    mask_width = max(1, int(round(view_width)))
    mask_height = max(1, int(round(view_height)))
    scale = min(1.0, _MAX_MASK_SIDE / float(max(mask_width, mask_height)))
    raster_width = max(1, int(round(mask_width * scale)))
    raster_height = max(1, int(round(mask_height * scale)))
    source_rgba = _rgba_from_source_at_size(
        Path(source_path), (raster_width, raster_height)
    )
    source_alpha = np.asarray(source_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(source_alpha == 255)):
        return {
            "status": "not_applicable",
            "applied": False,
            "before_sha256": before_sha,
            "after_sha256": before_sha,
            "before_byte_size": before_size,
            "after_byte_size": before_size,
        }

    eval_scale = min(1.0, 512.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    candidate_render = render_svg_to_rgba(svg_path, eval_width, eval_height)
    if candidate_render is None:
        raise RuntimeError("source_alpha_candidate_support_unmeasured")
    candidate_support = alpha_plane_metrics(
        source_eval[:, :, 3], candidate_render[:, :, 3]
    )

    from app.alpha_mask_budget import (  # noqa: PLC0415
        current_alpha_mask_encoding,
        current_alpha_mask_plan,
    )

    encoding = current_alpha_mask_encoding()
    contour_plan = current_alpha_mask_plan()
    quantized, opacity_by_level = _quantize_alpha(source_alpha)
    rectangles: dict[int, list[tuple[int, int, int, int]]] = {}
    direct_path = bool(
        encoding == "path" and contour_plan and contour_plan.get("layers")
    )
    if encoding == "rect" or (encoding == "path" and not direct_path):
        # A forced path context without a preflight plan is retained only for
        # legacy/direct unit fixtures; production preflight always supplies the
        # bounded direct contour plan and never materializes oversized rect XML.
        rectangles = _merged_rectangles_by_level(quantized)
        if not rectangles:
            raise RuntimeError("source_alpha_mask_empty_foreground")
    elif encoding != "path":
        raise RuntimeError(f"source_alpha_mask_encoding_invalid:{encoding}")

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
    content.set(
        "transform",
        f"translate({view_x:g} {view_y:g}) scale({sx:.12g} {sy:.12g})",
    )

    group_count = 0
    rectangle_count = 0
    path_count = 0
    if not direct_path:
        for level in sorted(rectangles):
            level_rectangles = rectangles[level]
            if not level_rectangles:
                continue
            group_attributes = {
                "fill": "#ffffff",
                "data-vektoryum-alpha-level": str(level),
            }
            opacity = float(opacity_by_level[level])
            if opacity < 1.0:
                group_attributes["fill-opacity"] = (
                    f"{opacity:.8f}".rstrip("0").rstrip(".")
                )
            group = ET.SubElement(content, qname("g"), group_attributes)
            for x, y, width, height in level_rectangles:
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
        if rectangle_count == 0:
            raise RuntimeError("source_alpha_mask_no_vector_rectangles")
    else:
        stroke_width = float(contour_plan.get("stroke_width", 0.5))
        for layer in contour_plan["layers"]:
            opacity = float(layer["opacity"])
            group_attributes = {
                "opacity": f"{opacity:.8f}".rstrip("0").rstrip("."),
                "data-vektoryum-alpha-level": str(layer["level"]),
                "data-vektoryum-alpha-contours": str(layer["contour_count"]),
            }
            group = ET.SubElement(content, qname("g"), group_attributes)
            ET.SubElement(
                group,
                qname("path"),
                {
                    "fill": "#ffffff",
                    "stroke": "#ffffff",
                    "stroke-width": f"{stroke_width:g}",
                    "stroke-linejoin": "miter",
                    "stroke-linecap": "square",
                    "fill-rule": "evenodd",
                    "d": str(layer["d"]),
                },
            )
            group_count += 1
            path_count += 1
        if path_count == 0:
            raise RuntimeError("source_alpha_mask_no_vector_paths")

    mask_only_render = _render_mask_only(
        root, defs, eval_width, eval_height
    )
    if mask_only_render is None:
        raise RuntimeError("source_alpha_mask_only_render_unmeasured")
    mask_only_metrics = alpha_plane_metrics(
        source_eval[:, :, 3], mask_only_render[:, :, 3]
    )

    protected = {"defs", "title", "desc", "metadata", "style"}
    movable = [
        child for child in list(root)
        if _local_name(str(child.tag)) not in protected
    ]
    if not movable:
        raise RuntimeError("source_alpha_mask_no_renderable_content")
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
    _atomic_write_tree(tree, svg_path)

    rendered = render_svg_to_rgba(svg_path, eval_width, eval_height)
    if rendered is None:
        raise RuntimeError("source_alpha_mask_render_unmeasured")
    metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered[:, :, 3])

    from app.final_artifact_evaluator import _thresholds  # noqa: PLC0415

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    underlay_width_pixels = 0.0
    underlay_clone_count = 0
    if not _alpha_passes(metrics, thresholds):
        # A mask can remove excess candidate coverage but cannot create support
        # where the selected vector candidate stopped just inside a source edge.
        # Add the selected candidate's own paint behind itself, never an arbitrary
        # canvas color, and choose the smallest measured expansion that passes the
        # unchanged hard alpha gates. The original candidate remains above it.
        for width_pixels in _UNDERLAY_STROKE_PIXELS:
            for child in list(wrapper):
                if child.get("data-vektoryum-alpha-coverage-underlay"):
                    wrapper.remove(child)
            width_user = width_pixels * max(sx, sy)
            underlay, clone_count = _coverage_underlay(movable, width_user)
            wrapper.insert(0, underlay)
            _atomic_write_tree(tree, svg_path)
            attempted = render_svg_to_rgba(svg_path, eval_width, eval_height)
            if attempted is None:
                continue
            attempted_metrics = alpha_plane_metrics(
                source_eval[:, :, 3], attempted[:, :, 3]
            )
            if _alpha_passes(attempted_metrics, thresholds):
                rendered = attempted
                metrics = attempted_metrics
                underlay_width_pixels = float(width_pixels)
                underlay_clone_count = int(clone_count)
                break
        else:
            for child in list(wrapper):
                if child.get("data-vektoryum-alpha-coverage-underlay"):
                    wrapper.remove(child)
            _atomic_write_tree(tree, svg_path)

    if float(metrics["alpha_iou"]) < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_mask_iou_gate_failed:"
            f"{metrics['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if float(metrics["alpha_mae"]) > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_mask_mae_gate_failed:"
            f"{metrics['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )

    after_sha = _sha256(svg_path)
    report: dict[str, Any] = {
        "status": "accepted",
        "applied": True,
        "schema": "rfv3d2-source-alpha-vector-mask-v1",
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "before_byte_size": before_size,
        "after_byte_size": svg_path.stat().st_size,
        "mask_encoding": "path" if direct_path else "rect",
        "mask_path_count": path_count,
        "mask_group_count": group_count,
        "mask_rectangle_count": rectangle_count,
        "mask_raster_width": raster_width,
        "mask_raster_height": raster_height,
        "alpha_level_count": group_count,
        "threshold_image_class": image_class,
        "alpha_iou": float(metrics["alpha_iou"]),
        "alpha_mae": float(metrics["alpha_mae"]),
        "source_coverage": float(metrics["source_coverage"]),
        "render_coverage": float(metrics["render_coverage"]),
        "mask_only_alpha_iou": float(mask_only_metrics["alpha_iou"]),
        "mask_only_alpha_mae": float(mask_only_metrics["alpha_mae"]),
        "candidate_support_iou": float(candidate_support["alpha_iou"]),
        "candidate_support_mae": float(candidate_support["alpha_mae"]),
        "coverage_underlay_width_pixels": underlay_width_pixels,
        "coverage_underlay_clone_count": underlay_clone_count,
    }
    report.update(
        _alpha_diagnostics(source_eval[:, :, 3], rendered[:, :, 3])
    )
    return report


def wrap_run_pipeline_with_alpha_mask(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Finalize selected color SVGs with source alpha after every mutator stage."""
    if getattr(original, "__vektoryum_alpha_mask_finalized__", False):
        return original

    @wraps(original)
    def alpha_mask_finalized_pipeline(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ) -> dict[str, Any]:
        result = original(
            image,
            original_path,
            trace_mode,
            job_dir,
            refine=refine,
            edge_cleanup=edge_cleanup,
        )
        best = result.get("best")
        if not isinstance(best, dict) or not best.get("svg_path"):
            return result

        mode = str(result.get("mode_used") or trace_mode)
        if mode not in _ALPHA_MASK_MODES:
            result["alpha_mask_report"] = {
                "status": "not_applicable",
                "applied": False,
                "reason": "unsupported_non_color_mode",
                "mode": mode,
            }
            return result

        source_path = Path(original_path)
        with Image.open(source_path) as source:
            source_alpha = np.asarray(
                source.convert("RGBA"), dtype=np.uint8
            )[:, :, 3].copy()
        if bool(np.all(source_alpha == 255)):
            result["alpha_mask_report"] = {
                "status": "not_applicable",
                "applied": False,
                "reason": "opaque_source",
                "mode": mode,
            }
            return result

        parent_path = Path(best["svg_path"])
        finalized_path = Path(job_dir) / f"{parent_path.stem}_alpha.svg"
        shutil.copy2(parent_path, finalized_path)
        report = apply_source_alpha_mask(finalized_path, source_path, mode)

        from app.pipeline import score_candidate, score_structure_integrity  # noqa: PLC0415
        from app.transform_journal import (  # noqa: PLC0415
            TransformJournal,
            merge_journal_reports,
        )

        alpha_journal = TransformJournal(
            parent_path,
            _journal_source_rgb(image),
            image_class=_MODE_IMAGE_CLASS[mode],
            required_metrics=set(),
        )
        accepted_path, alpha_stage = alpha_journal.consider_candidate(
            "source_alpha_vector_mask",
            parent_path,
            finalized_path,
            transform_report=report,
        )
        if accepted_path != finalized_path:
            first_reasons = list(alpha_stage.get("reason_codes") or ["unknown"])
            topology_only = bool(first_reasons) and all(
                str(reason).startswith("topology_") for reason in first_reasons
            )
            if not topology_only:
                finalized_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "source_alpha_mask_transform_gate_rejected:"
                    + ",".join(first_reasons)
                )

            # Yalnız kanıtlanmış topoloji reddi: ölçek-stabil painter maskesiyle
            # aynı parent'tan yeniden inşa et ve TAZE bir journal aşamasında
            # aynı değişmemiş kapılarla yeniden ölç. Painter da reddedilirse
            # fail-closed kalınır ve seçili SVG değişmeden bırakılır.
            from app.alpha_candidate_painter import (  # noqa: PLC0415
                apply_candidate_painter_reconstruction,
            )

            shutil.copy2(parent_path, finalized_path)
            try:
                report = apply_candidate_painter_reconstruction(
                    finalized_path, source_path, mode
                )
            except BaseException:
                finalized_path.unlink(missing_ok=True)
                raise
            alpha_journal = TransformJournal(
                parent_path,
                _journal_source_rgb(image),
                image_class=_MODE_IMAGE_CLASS[mode],
                required_metrics=set(),
            )
            accepted_path, alpha_stage = alpha_journal.consider_candidate(
                "source_alpha_vector_mask",
                parent_path,
                finalized_path,
                transform_report=report,
            )
            if accepted_path != finalized_path:
                finalized_path.unlink(missing_ok=True)
                retry_reasons = ",".join(
                    alpha_stage.get("reason_codes") or ["unknown"]
                )
                raise RuntimeError(
                    f"source_alpha_mask_transform_gate_rejected:{retry_reasons}"
                )
            report["mask_fallback_reason"] = "journal_topology_rejection"
            report["mask_fallback_trigger"] = ",".join(first_reasons)

        merged_journal = merge_journal_reports(
            result.get("transform_journal"), alpha_journal.to_dict()
        )
        if not merged_journal or not merged_journal.get("chain_valid", True):
            finalized_path.unlink(missing_ok=True)
            codes = ",".join(
                (merged_journal or {}).get("chain_failure_codes") or ["missing"]
            )
            raise RuntimeError(f"source_alpha_mask_journal_chain_invalid:{codes}")
        report["journal_status"] = alpha_stage["status"]
        report["journal_reasons"] = list(alpha_stage.get("reason_codes") or [])
        report["journal_chain_valid"] = True

        candidate = {
            **best,
            "name": f"{best.get('name', parent_path.stem)}_alpha",
            "svg_path": finalized_path,
            "alpha_mask_report": report,
        }
        rescored = score_candidate(
            candidate,
            source_path,
            result.get("analysis") or {},
            mode,
        )
        if rescored is None or not rescored.get("rendered_ok"):
            finalized_path.unlink(missing_ok=True)
            raise RuntimeError("source_alpha_mask_final_rescore_unmeasured")

        result["best"] = {**rescored, "alpha_mask_report": report}
        result["scored"] = [*(result.get("scored") or []), result["best"]]
        result["selection_reason"] = (
            f"{result.get('selection_reason') or 'selected'}+source_alpha_vector_mask"
        )
        result["alpha_mask_report"] = report
        result["transform_journal"] = merged_journal
        result["structure_report"] = score_structure_integrity(
            finalized_path, source_path
        )
        result["refit_info"] = {
            **(result.get("refit_info") or {}),
            "source_alpha_vector_mask": report,
        }
        return result

    alpha_mask_finalized_pipeline.__vektoryum_alpha_mask_finalized__ = True
    return alpha_mask_finalized_pipeline
