"""Stable candidate-identity contract for post-selection alpha finalization."""
from __future__ import annotations

from functools import wraps
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
    mass-preserving binary cut already has near-exact alpha IoU/MAE. It therefore
    cannot flatten real translucency, shadows, holes, or multi-component artwork.
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
    if len(contours[0]) > 5000:
        return None

    opaque_ratio = float(np.count_nonzero((plane >= 250) & support)) / float(
        support_count
    )
    if opaque_ratio < 0.98:
        return None

    binary_alpha = binary * np.uint8(255)
    alpha_min = np.minimum(binary_alpha, plane).sum(dtype=np.float64)
    alpha_max = np.maximum(binary_alpha, plane).sum(dtype=np.float64)
    alpha_iou = float(alpha_min / alpha_max) if alpha_max > 0 else 1.0
    alpha_mae = float(
        np.abs(binary_alpha.astype(np.int16) - plane.astype(np.int16)).mean()
        / 255.0
    )
    if alpha_iou < 0.999 or alpha_mae > 0.001:
        return None

    return binary.astype(np.int32), {0: 0.0, 1: 1.0}


def _install_simple_silhouette_quantizer() -> None:
    """Bind the compact silhouette as the final q32 painter candidate."""
    from app import alpha_candidate_painter as painter  # noqa: PLC0415

    original = painter._requantize_alpha
    if getattr(original, "__vektoryum_simple_silhouette_quantizer__", False):
        return

    @wraps(original)
    def requantize_with_simple_silhouette(alpha, max_levels):
        if int(max_levels) == 32:
            compact = _simple_opaque_silhouette_quantization(alpha)
            if compact is not None:
                return compact
        return original(alpha, max_levels)

    requantize_with_simple_silhouette.__vektoryum_simple_silhouette_quantizer__ = True
    painter._requantize_alpha = requantize_with_simple_silhouette


_install_simple_silhouette_quantizer()
