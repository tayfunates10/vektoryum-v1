"""Canonical-source multi-scale adapter for residual diagnostics.

The legacy synthetic builders were authored at 256px and several retain fixed
pixel coordinates even when passed another size. Re-running those builders at
64/128/512/1024 therefore changes or clips the source geometry. Multi-scale
fidelity must instead compare SVG rasterization against deterministic resizes of
the same canonical source raster.

The reference resize is deliberately nearest-neighbour. Area/Lanczos filtering
creates RGB values and ringing that are absent from the canonical source, which
then appear as false palette classes and false connected components. The source
reference may lose sub-pixel samples as resolution decreases, but it must never
invent new colors/components merely because the diagnostic changed scale.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from engine.regression.residual_error_metrics import (
    SCALES,
    _finite,
    measure_residual_error,
)
from app.source_truth import render_svg_to_rgba


def _resize_canonical_rgba(source: np.ndarray, size: int) -> np.ndarray:
    rgba = np.asarray(source, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("canonical multi-scale source RGBA olmalı")
    if rgba.shape[:2] == (size, size):
        return rgba.copy()
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(rgba, (int(size), int(size)), interpolation=interpolation)


def measure_multiscale_svg_from_base(
    svg_path: Any,
    source_builder: Callable[[int], Image.Image],
    *,
    base_size: int = 256,
    scales: Iterable[float] = SCALES,
) -> dict[str, Any]:
    """Measure one SVG against palette-preserving scales of one source raster."""
    requested_scales = tuple(float(scale) for scale in scales)
    canonical = np.asarray(
        source_builder(int(base_size)).convert("RGBA"),
        dtype=np.uint8,
    ).copy()
    if canonical.shape[:2] != (int(base_size), int(base_size)):
        canonical = _resize_canonical_rgba(canonical, int(base_size))

    levels: list[dict[str, Any]] = []
    missing: list[str] = []
    for scale in requested_scales:
        size = max(16, int(round(base_size * float(scale))))
        source = _resize_canonical_rgba(canonical, size)
        rendered = render_svg_to_rgba(svg_path, size, size)
        label = f"{float(scale):g}x"
        if rendered is None:
            missing.append(label)
            continue
        metrics = measure_residual_error(source, rendered)
        boundary = metrics.get("boundary_p95_px")
        quantization_allowance = max(math.sqrt(2.0), 0.75 * float(scale))
        levels.append(
            {
                "scale": label,
                "size": size,
                "de00_p95": metrics["de00_p95"],
                "visible_error_ratio": metrics["visible_error_ratio"],
                "boundary_p95_px": boundary,
                "boundary_quantization_allowance_px": quantization_allowance,
                "normalized_boundary_excess_px": (
                    max(0.0, float(boundary) - quantization_allowance) / float(scale)
                    if _finite(boundary)
                    else None
                ),
                "source_component_recall": metrics["source_component_recall"],
                "render_component_precision": metrics["render_component_precision"],
                "small_component_recall": metrics["small_component_recall"],
                "min_component_iou": metrics["min_component_iou"],
                "unmatched_source_count": metrics["unmatched_source_count"],
                "unmatched_render_count": metrics["unmatched_render_count"],
            }
        )

    def finite_values(key: str) -> list[float]:
        return [float(level[key]) for level in levels if _finite(level.get(key))]

    component_recall = finite_values("source_component_recall")
    small_recall = finite_values("small_component_recall")
    normalized_boundary_excess = finite_values("normalized_boundary_excess_px")
    visible_ratio = finite_values("visible_error_ratio")
    return {
        "scales_requested": [f"{scale:g}x" for scale in requested_scales],
        "missing_scales": missing,
        "levels": levels,
        "min_source_component_recall": min(component_recall) if component_recall else None,
        "min_small_component_recall": min(small_recall) if small_recall else None,
        "max_normalized_boundary_excess_px": (
            max(normalized_boundary_excess) if normalized_boundary_excess else None
        ),
        "max_visible_error_ratio": max(visible_ratio) if visible_ratio else None,
        "all_scales_measured": not missing and len(levels) == len(requested_scales),
        "source_scale_policy": "resize_canonical_palette_nearest_v2",
    }
