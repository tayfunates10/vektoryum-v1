"""Render-feedback scale-topology repair around the native-good semantic core.

The native semantic fitter is preserved byte-for-byte in
``semantic_region_geometry_core``.  This wrapper is fixture-agnostic: it renders
the source-derived semantic SVG at the five acceptance scales, detects only
same-palette component merges, and repaints the minimum articulation pixel with
the source palette class at that location using a one-device-pixel vector
anchor.  Every repair is re-rendered immediately; the caller still runs the
unchanged native fidelity/component/boundary transaction and all five lineage
checks, so unsuccessful or unsafe repairs fail closed to the exact vector
fallback.  No raster data is embedded and no shared budget/threshold is raised.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import cv2
import numpy as np

from app import semantic_region_geometry_core as _core

SemanticRegionFitError = _core.SemanticRegionFitError
_REPAIR_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
_MAX_REPAIR_ANCHORS_PER_SCALE = 8


def _fmt(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" shape-rendering="crispEdges">'
        + "".join(elements)
        + "</svg>"
    )


def _nearest_palette_labels(rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int32)
    palette = np.asarray(colors, dtype=np.int32)
    diff = arr[:, :, None, :] - palette[None, None, :, :]
    distance = np.sum(diff * diff, axis=3, dtype=np.int64)
    return np.argmin(distance, axis=2).astype(np.int16)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    return max(0, int(count) - 1)


def _component_counts(labels: np.ndarray, color_count: int) -> list[int]:
    return [_component_count(labels == index) for index in range(color_count)]


def _render_labels(
    width: int,
    height: int,
    elements: list[str],
    colors: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray | None:
    try:
        from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    except Exception:
        return None
    handle = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    path = handle.name
    try:
        handle.write(_svg_document(width, height, elements).encode("utf-8"))
        handle.close()
        rendered = render_svg_to_rgb(path, int(target_width), int(target_height))
        if rendered is None:
            return None
        return _nearest_palette_labels(rendered, colors)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass


def _source_target_labels(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(labels, dtype=np.int16),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int16)


def _articulation_repair(
    render_labels: np.ndarray,
    reference_labels: np.ndarray,
    class_index: int,
) -> tuple[int, int, int] | None:
    """Find a false-positive class pixel whose removal restores a component."""
    mask = render_labels == int(class_index)
    before = _component_count(mask)
    candidates = np.argwhere(mask & (reference_labels != int(class_index)))
    for y, x in candidates.tolist():
        trial = mask.copy()
        trial[int(y), int(x)] = False
        if _component_count(trial) > before:
            replacement = int(reference_labels[int(y), int(x)])
            return int(x), int(y), replacement
    return None


def _device_anchor(
    x: int,
    y: int,
    target_width: int,
    target_height: int,
    source_width: int,
    source_height: int,
    fill: str,
) -> str:
    cx = (float(x) + 0.5) * float(source_width) / float(target_width)
    cy = (float(y) + 0.5) * float(source_height) / float(target_height)
    return (
        f'<path d="M{_fmt(cx)} {_fmt(cy)}h0.001" fill="none" stroke="{fill}" '
        'stroke-width="1" vector-effect="non-scaling-stroke" stroke-linecap="square"/>'
    )


def _repair_component_merges(
    labels: np.ndarray,
    colors: np.ndarray,
    elements: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    source_height, source_width = labels.shape
    source_counts = _component_counts(labels, len(colors))
    repaired = list(elements)
    reports: list[dict[str, Any]] = []

    for factor in _REPAIR_FACTORS:
        target_width = max(16, int(round(source_width * factor)))
        target_height = max(16, int(round(source_height * factor)))
        reference = _source_target_labels(labels, target_width, target_height)
        anchors: list[dict[str, Any]] = []
        for _attempt in range(_MAX_REPAIR_ANCHORS_PER_SCALE):
            rendered = _render_labels(
                source_width,
                source_height,
                repaired,
                colors,
                target_width,
                target_height,
            )
            if rendered is None:
                break
            counts = _component_counts(rendered, len(colors))
            merged_classes = [
                index
                for index, (expected, actual) in enumerate(zip(source_counts, counts, strict=True))
                if actual < expected
            ]
            if not merged_classes:
                break
            applied = False
            for class_index in merged_classes:
                repair = _articulation_repair(rendered, reference, class_index)
                if repair is None:
                    continue
                x, y, replacement = repair
                red, green, blue = (
                    int(value) for value in np.asarray(colors, dtype=np.uint8)[replacement]
                )
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                repaired.append(
                    _device_anchor(
                        x,
                        y,
                        target_width,
                        target_height,
                        source_width,
                        source_height,
                        fill,
                    )
                )
                anchors.append(
                    {
                        "class_index": int(class_index),
                        "replacement_class_index": int(replacement),
                        "target_pixel": [int(x), int(y)],
                    }
                )
                applied = True
                break
            if not applied:
                break
        final_labels = _render_labels(
            source_width,
            source_height,
            repaired,
            colors,
            target_width,
            target_height,
        )
        final_counts = (
            _component_counts(final_labels, len(colors)) if final_labels is not None else None
        )
        reports.append(
            {
                "scale": f"{factor:g}x",
                "size": [target_width, target_height],
                "anchors": anchors,
                "final_component_counts": final_counts,
            }
        )
    return repaired, reports


def build_semantic_region_elements(labels: np.ndarray, colors: np.ndarray) -> dict[str, Any]:
    report = _core.build_semantic_region_elements(labels, colors)
    base_elements = list(report.get("elements") or [])
    repaired, repair_report = _repair_component_merges(
        np.asarray(labels), np.asarray(colors, dtype=np.uint8), base_elements
    )
    added = len(repaired) - len(base_elements)
    if added:
        strategies = dict(report.get("strategy_counts") or {})
        strategies["render_feedback_component_anchor"] = added
        report["strategy_counts"] = dict(sorted(strategies.items()))
        report["path_count"] = int(report.get("path_count") or 0) + added
        report["node_count"] = int(report.get("node_count") or 0) + 2 * added
    report["elements"] = repaired
    report["scale_topology_repair"] = repair_report
    report["scale_topology_anchor_count"] = added
    return report


__all__ = ["SemanticRegionFitError", "build_semantic_region_elements"]
