"""Geometric-logo guard for broad intentional neutral fills.

The established implementation is retained byte-for-byte in
``app.preprocess_base``. This compatibility layer changes only geometric-logo
preprocessing and only after the legacy pipeline has finished. Large, interior,
low-chroma regions that are visibly distinct from a uniform white background are
restored from source pixels, then the existing four-color dominant-palette reducer
runs again. Thin antialias films, border-connected background and chromatic
artwork remain under the legacy behavior.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app import preprocess_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy_preprocess_geometric_logo = _base.preprocess_geometric_logo


def _restore_broad_intentional_neutral_regions(
    source_rgb: np.ndarray,
    processed: np.ndarray,
    *,
    scale: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Restore only thick interior neutral fills erased by black/white hardening."""
    rgb = np.asarray(source_rgb, dtype=np.uint8)
    out = np.asarray(processed, dtype=np.uint8).copy()
    if rgb.shape != out.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        return out, []

    height, width = rgb.shape[:2]
    pw, ph = max(8, width // 12), max(8, height // 12)
    corners = np.concatenate(
        [
            rgb[:ph, :pw].reshape(-1, 3),
            rgb[:ph, -pw:].reshape(-1, 3),
            rgb[-ph:, :pw].reshape(-1, 3),
            rgb[-ph:, -pw:].reshape(-1, 3),
        ]
    ).astype(np.float32)
    background = np.median(corners, axis=0)
    background_mean = float(background.mean())
    background_spread = float(background.max() - background.min())

    # Fail closed: this guard is only for a uniform near-white canvas.
    if background_spread > 12.0 or background_mean < 245.0:
        return out, []

    source_i = rgb.astype(np.int16)
    channel_spread = source_i.max(axis=2) - source_i.min(axis=2)
    lightness = source_i.mean(axis=2)
    candidate = (
        (channel_spread <= 14)
        & (lightness >= 8.0)
        & (lightness <= background_mean - 8.0)
    ).astype(np.uint8)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    min_area = max(80 * scale * scale, round(0.0025 * height * width))
    kernel = np.ones((3, 3), np.uint8)
    accepted: list[dict[str, Any]] = []

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or area > int(0.35 * height * width):
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        ):
            continue

        mask = (labels == component_id).astype(np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=max(1, scale))
        survival = float(eroded.sum()) / float(area)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_radius = float(distance.max())
        if survival < 0.45 or max_radius < 3.0 * scale:
            continue

        representative = np.median(rgb[mask > 0], axis=0)
        if float(representative.max() - representative.min()) > 12.0:
            continue
        if float(np.linalg.norm(representative - background)) < 12.0:
            continue
        representative_mean = float(representative.mean())
        if not (8.0 <= representative_mean <= background_mean - 8.0):
            continue

        restored_rgb = np.clip(np.round(representative), 0, 255).astype(np.uint8)
        out[mask > 0] = restored_rgb
        accepted.append(
            {
                "area": area,
                "bbox": [x, y, component_width, component_height],
                "eroded_survival": round(survival, 6),
                "max_radius": round(max_radius, 4),
                "tone_class": "light" if representative_mean >= 128.0 else "dark",
                "rgb": [int(value) for value in restored_rgb],
            }
        )

    if accepted:
        # Reuse the established reducer so restored fills do not introduce an
        # extra antialias palette. The four dominant design colors remain.
        out = _base._reduce_to_dominant(
            out,
            k=4,
            erode_iters=max(1, scale),
        )
    return out, accepted


# Compatibility alias for the first targeted guard name.
def _restore_broad_light_neutral_regions(
    source_rgb: np.ndarray,
    processed: np.ndarray,
    *,
    scale: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    return _restore_broad_intentional_neutral_regions(
        source_rgb,
        processed,
        scale=scale,
    )


def preprocess_geometric_logo(arr: np.ndarray, report: dict[str, Any]) -> np.ndarray:
    processed = _legacy_preprocess_geometric_logo(arr, report)
    source_rgb = _base._rgba_to_rgb_on_white(arr)
    scale = 2 if report.get("supersampled") else 1
    restored, accepted = _restore_broad_intentional_neutral_regions(
        source_rgb,
        processed,
        scale=scale,
    )
    if accepted:
        report["steps"].append("restore_broad_intentional_neutral_regions")
        report["neutral_region_guard"] = {
            "schema": "broad-intentional-neutral-region-v2",
            "accepted_count": len(accepted),
            "components": accepted,
        }
        report["palette"] = _base._palette_list(restored)
    return restored


_base.preprocess_geometric_logo = preprocess_geometric_logo
_base._DISPATCH["geometric_logo"] = preprocess_geometric_logo
preprocess_for_mode = _base.preprocess_for_mode
