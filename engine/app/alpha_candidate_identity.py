"""Stable candidate-identity contract for post-selection alpha finalization."""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

_ALPHA_SUFFIX = "_alpha"


def _candidate_names(result: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for collection_name in ("results", "scored"):
        for candidate in result.get(collection_name) or []:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _restore_selected_candidate_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Keep artifact transforms from masquerading as new engine candidates."""
    report = result.get("alpha_mask_report")
    best = result.get("best")
    if not (
        isinstance(report, dict)
        and report.get("applied") is True
        and isinstance(best, dict)
    ):
        return result

    finalized_name = best.get("name")
    if not isinstance(finalized_name, str) or not finalized_name.endswith(_ALPHA_SUFFIX):
        raise RuntimeError("source_alpha_candidate_identity_missing_suffix")

    source_name = finalized_name[: -len(_ALPHA_SUFFIX)]
    if not source_name:
        raise RuntimeError("source_alpha_candidate_identity_empty")

    known_names = _candidate_names(result)
    if source_name not in known_names:
        raise RuntimeError(
            "source_alpha_candidate_identity_unbound:"
            f"{source_name} not in production candidates"
        )

    best["name"] = source_name
    for candidate in result.get("scored") or []:
        if not isinstance(candidate, dict):
            continue
        if candidate is best or (
            candidate.get("name") == finalized_name
            and candidate.get("alpha_mask_report") is report
        ):
            candidate["name"] = source_name

    report["source_candidate_name"] = source_name
    report["finalization_stage"] = "source_alpha_vector_mask"
    result["candidate_identity"] = {
        "status": "preserved",
        "source_candidate_name": source_name,
        "artifact_transform": "source_alpha_vector_mask",
    }
    return result


def wrap_run_pipeline_preserving_candidate_identity(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Restore the engine candidate name after the alpha artifact transform."""
    if getattr(original, "__vektoryum_candidate_identity_preserved__", False):
        return original

    @wraps(original)
    def identity_preserving_pipeline(*args, **kwargs) -> dict[str, Any]:
        return _restore_selected_candidate_identity(original(*args, **kwargs))

    identity_preserving_pipeline.__vektoryum_candidate_identity_preserved__ = True
    return identity_preserving_pipeline


def _simple_opaque_silhouette_quantization(alpha):
    """Return a one-level mask only when source alpha proves a simple silhouette.

    The compact fallback is deliberately stricter than the unchanged evaluator:
    it requires one connected, hole-free, mostly opaque support component whose
    only partial alpha is a narrow antialias fringe and whose mass-preserving
    binary cut already has near-exact alpha IoU/MAE. It therefore cannot flatten
    real translucency, shadows, holes, or multi-component artwork.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    plane = np.asarray(alpha, dtype=np.uint8)
    if plane.ndim != 2 or plane.size == 0:
        return None
    partial = (plane > 0) & (plane < 255)
    if not bool(partial.any()):
        return None

    target_area = float(plane.sum(dtype=np.float64) / 255.0)
    histogram = np.bincount(plane.ravel(), minlength=256)
    covered_at = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
    threshold = min(
        range(1, 256),
        key=lambda value: (
            abs(float(covered_at[value]) - target_area),
            abs(value - 128),
            value,
        ),
    )
    support = plane >= threshold
    support_count = int(np.count_nonzero(support))
    if support_count <= 0 or support_count >= int(plane.size):
        return None

    binary = support.astype(np.uint8)
    component_count, _labels = cv2.connectedComponents(binary, connectivity=8)
    if int(component_count) != 2:
        return None
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or len(contours) != 1:
        return None
    relation = hierarchy[0][0]
    if int(relation[2]) != -1 or int(relation[3]) != -1:
        return None

    # A fixed point cap rejected large, smooth rounded rectangles solely because
    # their perimeter scales with the source dimensions. Bound perimeter relative
    # to the raster instead; noisy/jagged or internally complex silhouettes still
    # fail while ordinary full-canvas rounded/circular supports remain eligible.
    height, width = plane.shape
    perimeter_budget = int(round(2.05 * float(width + height)))
    if len(contours[0]) > perimeter_budget:
        return None

    opaque_ratio = float(np.count_nonzero((plane >= 250) & support)) / float(
        support_count
    )
    if opaque_ratio < 0.98:
        return None

    # Partial alpha must be confined to a narrow edge ribbon. Measuring it against
    # contour length is scale-stable: a two-pixel antialias fringe remains eligible
    # on large rectangles/circles, while translucent interiors and broad shadows
    # grow with area and fail closed.
    partial_support_count = int(np.count_nonzero(partial & support))
    fringe_budget = max(16, int(round(2.5 * float(len(contours[0])))))
    if partial_support_count > fringe_budget:
        return None

    binary_alpha = binary * np.uint8(255)
    alpha_min = np.minimum(binary_alpha, plane).sum(dtype=np.float64)
    alpha_max = np.maximum(binary_alpha, plane).sum(dtype=np.float64)
    alpha_iou = float(alpha_min / alpha_max) if alpha_max > 0 else 1.0
    alpha_mae = float(
        np.abs(binary_alpha.astype(np.int16) - plane.astype(np.int16)).mean()
        / 255.0
    )
    if alpha_iou < 0.9985 or alpha_mae > 0.0015:
        return None

    return binary.astype(np.int32), {0: 0.0, 1: 1.0}


def _simple_silhouette_has_proven_canvas(
    svg_path: Path,
    source_path: Path,
):
    """Return the compact grid only when a proven full-canvas underpaint exists."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    from app.alpha_candidate_background import (  # noqa: PLC0415
        classify_comparison_background,
    )
    from app.alpha_candidate_knockout import _viewbox  # noqa: PLC0415
    from app.alpha_preprocess import _rgba_from_source_at_size  # noqa: PLC0415
    from app import alpha_candidate_painter as painter  # noqa: PLC0415

    try:
        root = ET.parse(Path(svg_path)).getroot()
        _x, _y, width, height = _viewbox(root)
    except (ET.ParseError, OSError, RuntimeError, ValueError):
        return None

    scale = min(
        1.0,
        float(painter._PAINTER_GRID_MAX_SIDE) / float(max(width, height)),
    )
    grid_width = max(1, int(round(width * scale)))
    grid_height = max(1, int(round(height * scale)))
    grid_rgba = _rgba_from_source_at_size(
        Path(source_path), (grid_width, grid_height)
    )
    compact = _simple_opaque_silhouette_quantization(grid_rgba[:, :, 3])
    if compact is None:
        return None
    status, canvas = classify_comparison_background(
        root, grid_rgba, grid_width, grid_height
    )
    if status != "proven" or canvas is None:
        return None
    return compact


def _install_simple_silhouette_quantizer() -> None:
    """Bind compact, supportless silhouette as the final q32 painter candidate."""
    from app import alpha_candidate_painter as painter  # noqa: PLC0415
    from app import alpha_candidate_support as support  # noqa: PLC0415

    original_quantizer = painter._requantize_alpha
    if getattr(
        original_quantizer,
        "__vektoryum_simple_silhouette_quantizer__",
        False,
    ):
        return

    @wraps(original_quantizer)
    def requantize_with_simple_silhouette(alpha, max_levels):
        if int(max_levels) == 32:
            compact = _simple_opaque_silhouette_quantization(alpha)
            if compact is not None:
                return compact
        return original_quantizer(alpha, max_levels)

    requantize_with_simple_silhouette.__vektoryum_simple_silhouette_quantizer__ = True
    painter._requantize_alpha = requantize_with_simple_silhouette

    original_apply = painter.apply_candidate_painter_reconstruction

    @wraps(original_apply)
    def apply_with_supportless_simple_silhouette(svg_path, source_path, mode):
        compact = _simple_silhouette_has_proven_canvas(
            Path(svg_path), Path(source_path)
        )
        if compact is None:
            return original_apply(svg_path, source_path, mode)

        original_strokes = painter._PAINTER_STROKE_PIXELS
        original_min_stroke = painter._PAINTER_UNDERPAINT_MIN_STROKE_PIXELS
        original_expand = support._expand_candidate_paint

        @wraps(original_expand)
        def expand_or_preserve(node, stroke_width, inherited_fill=None):
            if float(stroke_width) == 0.0:
                # The proven full-canvas element is retained inside the same source
                # alpha mask, so it supplies complete support. Preserve every artwork
                # stroke/fill byte instead of manufacturing topology-changing support.
                geometry_count = sum(
                    1
                    for child in node.iter()
                    if support._local_name(str(child.tag)).lower()
                    in support._GEOMETRY_TAGS
                )
                return max(1, int(geometry_count))
            return original_expand(node, stroke_width, inherited_fill)

        painter._PAINTER_STROKE_PIXELS = tuple(
            dict.fromkeys((0.0, *original_strokes))
        )
        painter._PAINTER_UNDERPAINT_MIN_STROKE_PIXELS = 0.0
        support._expand_candidate_paint = expand_or_preserve
        try:
            report = original_apply(svg_path, source_path, mode)
        finally:
            support._expand_candidate_paint = original_expand
            painter._PAINTER_UNDERPAINT_MIN_STROKE_PIXELS = original_min_stroke
            painter._PAINTER_STROKE_PIXELS = original_strokes

        if float(report.get("candidate_support_stroke_width_pixels") or 0.0) == 0.0:
            report["candidate_support_expanded_geometry_count"] = 0
            report["candidate_support_mode"] = (
                "proven_canvas_underpaint_without_artwork_expansion"
            )
        return report

    apply_with_supportless_simple_silhouette.__vektoryum_supportless_silhouette__ = True
    painter.apply_candidate_painter_reconstruction = (
        apply_with_supportless_simple_silhouette
    )


_install_simple_silhouette_quantizer()
