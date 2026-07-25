"""Fail-closed compact source-alpha reconstruction for antialiased ellipses.

Some public badge assets contain an opaque vector design inside one antialiased
circular/elliptical alpha boundary. Expanding that boundary into hundreds of
quantized contour layers is both wasteful and may introduce topology, edge and
seam regressions. This module recognizes only a source plane that is already
measured to be a near-exact ellipse, then applies one renderer-native SVG ellipse
mask to the unchanged selected candidate.

Detection alone never authorizes publication. The normal dual alpha evaluator,
path/node identity checks, byte budget and the outer TransformJournal remain
unchanged and authoritative. Any rejection restores the original bytes and calls
the established alpha reconstruction chain.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from app.alpha_candidate_knockout import (
    _local_name,
    _path_node_counts,
    _render_root,
    _unique_id,
    _viewbox,
    _write_tree_to_temp,
)
from app.alpha_candidate_support import _PROTECTED_ROOT_TAGS, _SVG_NS, _strip_content_alpha


_MIN_BINARY_ELLIPSE_IOU = 0.998
_MAX_SOFT_PIXEL_FRACTION = 0.03
_MIN_INNER_OPAQUE_FRACTION = 0.995
_MIN_PARENT_SOURCE_SUPPORT_RECALL = 0.995


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _ellipse_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height or x1 <= x0 or y1 <= y0:
        return np.zeros(shape, dtype=bool)
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).ellipse((x0, y0, x1, y1), fill=1)
    return np.asarray(image, dtype=np.uint8).astype(bool)


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left & right) / union)


def fit_soft_ellipse(
    alpha: np.ndarray,
) -> tuple[
    tuple[int, int, int, int],
    tuple[float, float, float, float],
    float,
] | None:
    """Return measured ellipse geometry only for a simple antialiased boundary."""
    plane = np.asarray(alpha, dtype=np.uint8)
    if plane.ndim != 2 or min(plane.shape) < 16:
        return None
    if bool(np.all(plane == 255)) or not bool(np.any(plane > 0)):
        return None

    soft = (plane > 0) & (plane < 255)
    soft_fraction = float(np.mean(soft))
    if soft_fraction <= 0.0 or soft_fraction > _MAX_SOFT_PIXEL_FRACTION:
        return None

    target = plane >= 128
    box = _bbox(target)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    box_width = x1 - x0 + 1
    box_height = y1 - y0 + 1
    if box_width < 9 or box_height < 9:
        return None

    primitive = _ellipse_mask(plane.shape, box)
    overlap = _binary_iou(target, primitive)
    if overlap < _MIN_BINARY_ELLIPSE_IOU:
        return None

    # A true alpha boundary has an opaque interior. This rejects translucent
    # ellipses, internal holes and approximately oval multi-object silhouettes.
    inset = max(2, int(round(min(box_width, box_height) * 0.005)))
    inner_box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    inner = _ellipse_mask(plane.shape, inner_box)
    if not bool(np.any(inner)):
        return None
    opaque_fraction = float(np.mean(plane[inner] >= 250))
    if opaque_fraction < _MIN_INNER_OPAQUE_FRACTION:
        return None

    # The alpha integral is the coverage area of the antialiased ellipse. Derive
    # sub-pixel radii from that area while retaining the measured bounding-box
    # aspect ratio. This avoids the half/one-pixel shrink introduced by choosing a
    # radius solely from the alpha>=128 bbox.
    alpha_area = float(np.asarray(plane, dtype=np.float64).sum() / 255.0)
    aspect = float(box_width / box_height)
    if alpha_area <= 0.0 or aspect <= 0.0:
        return None
    radius_x = math.sqrt(alpha_area * aspect / math.pi)
    radius_y = math.sqrt(alpha_area / (aspect * math.pi))
    center_x = (x0 + x1 + 1) / 2.0
    center_y = (y0 + y1 + 1) / 2.0
    if radius_x > box_width / 2.0 + 1.0 or radius_y > box_height / 2.0 + 1.0:
        return None

    return box, (center_x, center_y, radius_x, radius_y), overlap


def _build_soft_ellipse_tree(
    original_root: ET.Element,
    alpha_shape: tuple[int, int],
    geometry: tuple[float, float, float, float],
    transaction_id: str,
) -> ET.Element:
    from app.alpha_artwork_identity import (
        ROLE_ARTWORK_CONTAINER,
        ROLE_MASK_APPLICATION,
        ROLE_MASK_DEFINITION,
        tag_transform_node,
    )

    root = copy.deepcopy(original_root)
    qname = lambda name: f"{{{_SVG_NS}}}{name}"
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

    paint_id = _unique_id(root, "vektoryum-alpha-soft-ellipse-paint")
    paint = ET.SubElement(defs, qname("g"), {"id": paint_id})
    tag_transform_node(paint, ROLE_ARTWORK_CONTAINER, transaction_id)
    movable = [
        child
        for child in list(root)
        if child is not defs
        and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not movable:
        raise RuntimeError("source_alpha_soft_ellipse_no_paint")
    for child in movable:
        root.remove(child)
        _strip_content_alpha(child)
        paint.append(child)

    view_x, view_y, view_width, view_height = _viewbox(root)
    raster_height, raster_width = alpha_shape
    mask_id = _unique_id(root, "vektoryum-alpha-soft-ellipse-mask")
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
            "style": "mask-type:luminance",
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

    center_x, center_y, radius_x, radius_y = geometry
    ET.SubElement(
        content,
        qname("ellipse"),
        {
            "cx": f"{center_x:.12g}",
            "cy": f"{center_y:.12g}",
            "rx": f"{radius_x:.12g}",
            "ry": f"{radius_y:.12g}",
            "fill": "white",
        },
    )

    layer = ET.SubElement(root, qname("g"))
    tag_transform_node(layer, ROLE_MASK_APPLICATION, transaction_id)
    ET.SubElement(
        layer,
        qname("use"),
        {"href": f"#{paint_id}", "mask": f"url(#{mask_id})"},
    )
    return root


def apply_soft_ellipse_alpha(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Publish one measured ellipse mask or raise without changing parent bytes."""
    from app.alpha_artwork_identity import alpha_transaction_id
    from app.alpha_candidate_validation import validate_alpha_reconstruction_contract
    from app.alpha_mask_budget import _journal_limits
    from app.alpha_preprocess import _rgba_from_source_at_size

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    original_root = ET.fromstring(before_bytes)
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_soft_ellipse_root_not_svg")
    parent_counts = _path_node_counts(original_root)

    with Image.open(source) as source_image:
        source_size = source_image.size
    source_full = _rgba_from_source_at_size(source, source_size)
    source_alpha = np.asarray(source_full[:, :, 3], dtype=np.uint8)
    fitted = fit_soft_ellipse(source_alpha)
    if fitted is None:
        raise RuntimeError("source_alpha_soft_ellipse_not_applicable")
    box, geometry, binary_overlap = fitted

    _vx, _vy, view_width, view_height = _viewbox(original_root)
    eval_scale = min(1.0, 512.0 / float(max(view_width, view_height)))
    eval_width = max(1, int(round(view_width * eval_scale)))
    eval_height = max(1, int(round(view_height * eval_scale)))
    collapsed = _render_root(original_root, eval_width, eval_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_soft_ellipse_parent_render_unmeasured")
    source_eval = _rgba_from_source_at_size(source, (eval_width, eval_height))
    source_support = np.asarray(source_eval[:, :, 3], dtype=np.uint8) >= 128
    support_pixels = int(np.count_nonzero(source_support))
    if support_pixels <= 0:
        raise RuntimeError("source_alpha_soft_ellipse_source_support_empty")
    parent_support = np.asarray(collapsed[:, :, 3], dtype=np.uint8) >= 128
    support_recall = float(
        np.count_nonzero(parent_support & source_support) / support_pixels
    )
    if support_recall < _MIN_PARENT_SOURCE_SUPPORT_RECALL:
        raise RuntimeError(
            "source_alpha_soft_ellipse_parent_source_support_recall_below_min:"
            f"{support_recall:.6f}<{_MIN_PARENT_SOURCE_SUPPORT_RECALL}"
        )

    parent_sha = hashlib.sha256(before_bytes).hexdigest()
    alpha_sha = hashlib.sha256(
        np.ascontiguousarray(source_alpha).tobytes()
    ).hexdigest()
    transaction_id = alpha_transaction_id(parent_sha, alpha_sha, mode, "soft-ellipse")
    candidate_root = _build_soft_ellipse_tree(
        original_root,
        source_alpha.shape,
        geometry,
        transaction_id,
    )
    temporary = _write_tree_to_temp(candidate_root, target)
    try:
        limits = _journal_limits(original_root, before_size)
        candidate_size = int(temporary.stat().st_size)
        if candidate_size > int(limits["byte_limit"]):
            raise RuntimeError(
                "source_alpha_soft_ellipse_byte_budget_rejected:"
                f"{candidate_size}>{limits['byte_limit']}"
            )
        validation = validate_alpha_reconstruction_contract(
            temporary,
            source_full,
            mode,
            parent_counts,
        )
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    center_x, center_y, radius_x, radius_y = geometry
    return {
        "status": "accepted",
        "applied": True,
        "schema": "rfv3d3-soft-ellipse-alpha-v1",
        "mask_encoding": "measured_antialiased_ellipse",
        "before_byte_size": int(before_size),
        "after_byte_size": int(target.stat().st_size),
        "before_sha256": parent_sha,
        "after_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "candidate_geometry_preserved": True,
        "candidate_path_data_preserved": True,
        "candidate_identity_preserved": True,
        "soft_ellipse_bbox": [int(value) for value in box],
        "soft_ellipse_geometry": [
            float(center_x),
            float(center_y),
            float(radius_x),
            float(radius_y),
        ],
        "soft_ellipse_binary_iou": float(binary_overlap),
        "soft_ellipse_parent_source_support_recall": float(support_recall),
        "preflight_parent_path_count": int(parent_counts[0]),
        "preflight_parent_node_count": int(parent_counts[1]),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_node_limit": int(limits["node_limit"]),
        "preflight_byte_limit": int(limits["byte_limit"]),
        **validation,
    }


def make_soft_ellipse_alpha_first(
    original_apply: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Try the measured ellipse and delegate byte-identically on any rejection."""
    if getattr(original_apply, "__vektoryum_soft_ellipse_alpha_first__", False):
        return original_apply

    @wraps(original_apply)
    def ellipse_first(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        parent = target.read_bytes()
        try:
            return apply_soft_ellipse_alpha(target, Path(source_path), mode)
        except Exception as exc:  # noqa: BLE001 - established fallback is authoritative
            if target.read_bytes() != parent:
                target.write_bytes(parent)
            if not str(exc).startswith("source_alpha_soft_ellipse_not_applicable"):
                print(
                    "source_alpha_soft_ellipse_attempt_rejected="
                    f"{type(exc).__name__}:{exc}",
                    flush=True,
                )
            report = original_apply(target, Path(source_path), mode)
            if isinstance(report, dict):
                report = dict(report)
                report["soft_ellipse_alpha"] = {
                    "status": "not_applicable_fallback_used",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            return report

    ellipse_first.__vektoryum_soft_ellipse_alpha_first__ = True
    return ellipse_first


def install_soft_ellipse_alpha() -> None:
    """Install once after the application package has assembled its alpha chain."""
    from app import alpha_svg_mask

    current = alpha_svg_mask.apply_source_alpha_mask
    if getattr(current, "__vektoryum_soft_ellipse_alpha_first__", False):
        return
    alpha_svg_mask.apply_source_alpha_mask = make_soft_ellipse_alpha_first(current)
