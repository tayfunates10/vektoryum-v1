"""Compact source-alpha preservation using existing candidate elements.

The verbose raster-rectangle mask is a safe general fallback, but it is wasteful
when the selected vector candidate already contains the exact source-support
geometry. This module tries renderer-proven, byte-transactional candidates before
the existing mask chain:

* archive one color-agnostically proven comparison-canvas child in ``<defs>``;
* archive same-canvas-colour negative-space children whose measured source alpha
  support is exactly the transparent level;
* assign one measured source-alpha opacity to each remaining direct paint child.

No path data, path-command node, palette, gradient or evaluator threshold is
changed. The candidate is published only after the existing direct alpha-plane
and FinalArtifactEvaluator contract passes with identical path/node counts. Any
parse, classification, render, identity, byte or fidelity rejection leaves the
parent bytes untouched and delegates to the established guarded mask builder.
"""
from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.alpha_candidate_background import classify_comparison_background
from app.alpha_candidate_knockout import (
    _PROTECTED_ROOT_TAGS,
    _RENDERABLE_TAGS,
    _local_name,
    _path_node_counts,
    _probe_root,
    _render_root,
    _viewbox,
)
from app.alpha_candidate_paint_selection import comparison_canvas_rgb
from app.alpha_candidate_validation import validate_alpha_reconstruction_contract
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_MAX_DIRECT_CHILDREN = 32
_ALPHA_STYLE_NAMES = {"opacity", "fill-opacity", "stroke-opacity"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _style_has_alpha(style: str | None) -> bool:
    declarations = str(style or "").lower().replace(" ", "")
    return any(f"{name}:" in declarations for name in _ALPHA_STYLE_NAMES)


def _has_existing_content_alpha(node: ET.Element) -> bool:
    for element in node.iter():
        if any(element.get(name) is not None for name in _ALPHA_STYLE_NAMES):
            return True
        if _style_has_alpha(element.get("style")):
            return True
    return False


def _opacity_text(level: int) -> str:
    return f"{int(level) / 255.0:.8f}".rstrip("0").rstrip(".")


def _weighted_source_alpha_mode(
    child_alpha: np.ndarray,
    source_alpha: np.ndarray,
) -> int:
    covered = np.asarray(child_alpha, dtype=np.uint8) >= 128
    if not np.any(covered):
        raise RuntimeError("source_alpha_direct_child_render_empty")
    values = np.asarray(source_alpha, dtype=np.uint8)[covered]
    weights = np.asarray(child_alpha, dtype=np.float64)[covered]
    histogram = np.bincount(values, weights=weights, minlength=256)
    level = int(np.argmax(histogram))
    if float(histogram[level]) <= 0.0:
        raise RuntimeError("source_alpha_direct_child_support_unmeasured")
    return level


def _archive_indexes(
    root: ET.Element,
    archive_reasons: dict[int, str],
) -> None:
    if not archive_reasons:
        return
    children = list(root)
    selected: list[tuple[ET.Element, str]] = []
    for index, reason in sorted(archive_reasons.items()):
        if index < 0 or index >= len(children):
            raise RuntimeError("source_alpha_direct_archive_index_drift")
        selected.append((children[index], reason))

    defs = next(
        (
            child
            for child in list(root)
            if _local_name(str(child.tag)).lower() == "defs"
        ),
        None,
    )
    if defs is None:
        defs = ET.Element(f"{{{_SVG_NS}}}defs")
        root.insert(0, defs)

    for element, reason in selected:
        root.remove(element)
        element.set("data-vektoryum-source-alpha-archive", reason)
        defs.append(element)


def _direct_renderable_indexes(root: ET.Element) -> list[int]:
    indexes = [
        index
        for index, child in enumerate(list(root))
        if _local_name(str(child.tag)).lower() in _RENDERABLE_TAGS
        and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not indexes:
        raise RuntimeError("source_alpha_direct_no_paint_children")
    if len(indexes) > _MAX_DIRECT_CHILDREN:
        raise RuntimeError(
            "source_alpha_direct_child_budget_exceeded:"
            f"{len(indexes)}>{_MAX_DIRECT_CHILDREN}"
        )
    return indexes


def _build_direct_candidate(
    original_root: ET.Element,
    source_eval: np.ndarray,
    eval_width: int,
    eval_height: int,
) -> tuple[ET.Element, dict[str, Any]]:
    status, background = classify_comparison_background(
        original_root,
        source_eval,
        eval_width,
        eval_height,
    )
    if status == "ambiguous":
        raise RuntimeError("source_alpha_direct_background_ambiguous")

    original_children = list(original_root)
    background_index = (
        original_children.index(background)
        if status == "proven" and background is not None
        else None
    )
    canvas_rgb = comparison_canvas_rgb(background) if background is not None else None
    source_alpha = np.asarray(source_eval[:, :, 3], dtype=np.uint8)
    nonzero_levels = [int(value) for value in np.unique(source_alpha) if int(value) > 0]
    if not nonzero_levels:
        raise RuntimeError("source_alpha_direct_empty_source")

    assignments: list[tuple[int, int]] = []
    archive_reasons: dict[int, str] = {}
    if background_index is not None:
        archive_reasons[background_index] = "renderer-proven-comparison-canvas-v1"

    negative_space_count = 0
    for index in _direct_renderable_indexes(original_root):
        if background_index is not None and index == background_index:
            continue
        child = original_children[index]
        if _has_existing_content_alpha(child):
            raise RuntimeError("source_alpha_direct_existing_content_alpha")
        rendered = _render_root(
            _probe_root(original_root, child),
            eval_width,
            eval_height,
        )
        if rendered is None:
            raise RuntimeError("source_alpha_direct_child_render_unmeasured")
        if rendered.shape[:2] != (eval_height, eval_width):
            rendered = resize_rgba(rendered, eval_width, eval_height)
        level = _weighted_source_alpha_mode(rendered[:, :, 3], source_alpha)
        if level == 0:
            child_rgb = comparison_canvas_rgb(child)
            if canvas_rgb is None or child_rgb != canvas_rgb:
                raise RuntimeError(
                    "source_alpha_direct_transparent_child_not_canvas_colour"
                )
            archive_reasons[index] = "measured-canvas-colour-negative-space-v1"
            negative_space_count += 1
            continue
        assignments.append((index, level))

    if not assignments:
        raise RuntimeError("source_alpha_direct_no_supported_paint")

    candidate_root = copy.deepcopy(original_root)
    candidate_children = list(candidate_root)
    assigned_count = 0
    translucent_count = 0
    for index, level in assignments:
        if index < 0 or index >= len(candidate_children):
            raise RuntimeError("source_alpha_direct_child_index_drift")
        child = candidate_children[index]
        assigned_count += 1
        if level < 255:
            child.set("opacity", _opacity_text(level))
            child.set(
                "data-vektoryum-source-alpha-direct",
                str(level),
            )
            translucent_count += 1

    _archive_indexes(candidate_root, archive_reasons)

    return candidate_root, {
        "background_status": status,
        "comparison_canvas_archived": background_index is not None,
        "canvas_colour_negative_space_archive_count": int(negative_space_count),
        "direct_child_count": int(assigned_count),
        "direct_translucent_child_count": int(translucent_count),
        "source_nonzero_alpha_levels": nonzero_levels,
    }


def _write_candidate(root: ET.Element, target: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=Path(target).parent,
        prefix=f".{Path(target).name}.",
        suffix=".direct-alpha.svg",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ET.ElementTree(root).write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def apply_direct_element_alpha(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Publish a compact direct-element alpha candidate or raise fail-closed."""
    target = Path(svg_path)
    source = Path(source_path)
    before_sha = _sha256(target)
    before_size = target.stat().st_size
    ET.register_namespace("", _SVG_NS)

    try:
        original_root = ET.parse(target).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError("source_alpha_direct_parse_failed") from exc
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_direct_root_not_svg")
    parent_counts = _path_node_counts(original_root)

    _view_x, _view_y, view_width, view_height = _viewbox(original_root)
    eval_scale = min(1.0, 512.0 / float(max(view_width, view_height)))
    eval_width = max(1, int(round(view_width * eval_scale)))
    eval_height = max(1, int(round(view_height * eval_scale)))
    source_eval = _rgba_from_source_at_size(source, (eval_width, eval_height))
    source_alpha = np.asarray(source_eval[:, :, 3], dtype=np.uint8)
    if bool(np.all(source_alpha == 255)):
        raise RuntimeError("source_alpha_direct_opaque_source")
    if not np.any(source_alpha > 0):
        raise RuntimeError("source_alpha_direct_empty_source")

    candidate_root, direct_report = _build_direct_candidate(
        original_root,
        source_eval,
        eval_width,
        eval_height,
    )
    temporary = _write_candidate(candidate_root, target)
    try:
        from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415

        limits = _journal_limits(original_root, before_size)
        candidate_size = temporary.stat().st_size
        if candidate_size > int(limits["byte_limit"]):
            raise RuntimeError(
                "source_alpha_direct_byte_budget_rejected:"
                f"{candidate_size}>{limits['byte_limit']}"
            )

        with Image.open(source) as source_image:
            source_size = source_image.size
        source_rgba_full = _rgba_from_source_at_size(source, source_size)
        validation = validate_alpha_reconstruction_contract(
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

    after_sha = _sha256(target)
    return {
        "status": "accepted",
        "applied": True,
        "schema": "rfv3d3-direct-element-alpha-v1",
        "mask_encoding": "direct_existing_elements",
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "before_byte_size": int(before_size),
        "after_byte_size": int(target.stat().st_size),
        "candidate_geometry_preserved": True,
        "candidate_identity_preserved": True,
        "preflight_parent_path_count": int(parent_counts[0]),
        "preflight_parent_node_count": int(parent_counts[1]),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_node_limit": int(limits["node_limit"]),
        "preflight_byte_limit": int(limits["byte_limit"]),
        **direct_report,
        **validation,
    }


def make_direct_element_alpha_first(
    guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Try compact existing-element alpha, then delegate byte-identically."""
    if getattr(guarded_builder, "__vektoryum_direct_element_alpha_first__", False):
        return guarded_builder

    @wraps(guarded_builder)
    def direct_first(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        parent = target.read_bytes()
        try:
            return apply_direct_element_alpha(target, Path(source_path), mode)
        except Exception as exc:  # noqa: BLE001 - guarded fallback remains authoritative
            if target.read_bytes() != parent:
                target.write_bytes(parent)
            report = guarded_builder(target, Path(source_path), mode)
            if isinstance(report, dict):
                report = dict(report)
                report["direct_element_alpha"] = {
                    "status": "rejected_fallback_used",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            return report

    direct_first.__vektoryum_direct_element_alpha_first__ = True
    return direct_first
