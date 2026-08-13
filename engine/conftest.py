"""Temporary Issue #152 true-contract primitive diagnostic; removed before final."""
from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    from app import semantic_region_geometry_core as core
    from app.source_palette_vector import _svg_document
    from engine.regression.output_quality_suite import _draw_small_details
    from engine.regression.residual_multiscale_source import measure_multiscale_svg_from_base

    source256 = np.asarray(_draw_small_details(256).convert("RGB"), dtype=np.uint8)
    colors, inverse = np.unique(source256.reshape(-1, 3), axis=0, return_inverse=True)
    labels = inverse.reshape(256, 256)
    original = core._ellipse_from_cycle
    results = []
    with tempfile.TemporaryDirectory(prefix="issue152-direct-ellipse-sweep-") as tmp:
        root = Path(tmp)
        try:
            for center_shift in (-0.5, -0.25, 0.0, 0.25, 0.5):
                for radius_inset in (-0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 0.75):
                    def candidate(points, cs=center_shift, ri=radius_inset):
                        if len(points) < 8:
                            return None
                        arr = np.asarray(points, dtype=np.float64)
                        low = arr.min(axis=0); high = arr.max(axis=0); span = high - low
                        if min(span) < 2.0 or max(span) / max(1e-9, min(span)) > 1.35:
                            return None
                        area = abs(float(cv2.contourArea(arr.astype(np.float32).reshape(-1, 1, 2))))
                        occupancy = area / max(1.0, float(span[0] * span[1]))
                        if not (0.48 <= occupancy <= 0.90):
                            return None
                        rx = float(span[0]) / 2.0 - float(ri)
                        ry = float(span[1]) / 2.0 - float(ri)
                        if min(rx, ry) < 1.0:
                            return None
                        cx = float(low[0] + high[0]) / 2.0 + float(cs)
                        cy = float(low[1] + high[1]) / 2.0 + float(cs)
                        return (
                            f"M{core._fmt(cx-rx)} {core._fmt(cy)}"
                            f"A{core._fmt(rx)} {core._fmt(ry)} 0 1 0 {core._fmt(cx+rx)} {core._fmt(cy)}"
                            f"A{core._fmt(rx)} {core._fmt(ry)} 0 1 0 {core._fmt(cx-rx)} {core._fmt(cy)}Z"
                        )

                    core._ellipse_from_cycle = candidate
                    semantic = core.build_semantic_region_elements(labels, colors)
                    output = root / f"small-{center_shift}-{radius_inset}.svg"
                    output.write_text(_svg_document(256, 256, list(semantic["elements"])), encoding="utf-8")
                    multi = measure_multiscale_svg_from_base(output, _draw_small_details)
                    levels = multi.get("levels") or []
                    if not levels:
                        continue
                    results.append({
                        "center_shift": float(center_shift),
                        "radius_inset": float(radius_inset),
                        "min_iou": min(float(level.get("min_component_iou") or 0.0) for level in levels),
                        "min_recall": min(float(level.get("source_component_recall") or 0.0) for level in levels),
                        "min_precision": min(float(level.get("render_component_precision") or 0.0) for level in levels),
                        "min_small": min(float(level.get("small_component_recall") or 0.0) for level in levels),
                        "max_visible": max(float(level.get("visible_error_ratio") or 0.0) for level in levels),
                        "max_de": max(float(level.get("de00_p95") or 0.0) for level in levels),
                        "max_boundary": max(float(level.get("normalized_boundary_excess_px") or 0.0) for level in levels),
                        "levels": [
                            (level.get("scale"), level.get("min_component_iou"), level.get("source_component_recall"), level.get("render_component_precision"), level.get("small_component_recall"), level.get("visible_error_ratio"), level.get("de00_p95"), level.get("normalized_boundary_excess_px"))
                            for level in levels
                        ],
                    })
        finally:
            core._ellipse_from_cycle = original
    results.sort(key=lambda item: (item["min_recall"], item["min_precision"], item["min_small"], item["min_iou"], -item["max_visible"], -item["max_de"], -item["max_boundary"]), reverse=True)
    print("\nISSUE152_DIRECT_REAL_ELLIPSE_SWEEP_TOP=" + repr(results[:10]))
