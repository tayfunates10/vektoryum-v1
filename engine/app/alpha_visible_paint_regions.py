"""Boundary-stable vector paint regions for Issue #145 visible-alpha repair."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _viewbox
from app.alpha_mask_contour import trace_cell_contours

_SVG_NS = "http://www.w3.org/2000/svg"
_ALPHA_BANDS_PER_VISIBLE_CLASS = 4


def _source_white(rgba: np.ndarray) -> np.ndarray:
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3:4].astype(np.float64) / 255.0
    return np.clip(
        array[:, :, :3].astype(np.float64) * alpha
        + 255.0 * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def _loop_path_data(loops: list[list[tuple[int, int]]]) -> str:
    parts: list[str] = []
    for corners in loops:
        if len(corners) < 3:
            continue
        x0, y0 = corners[0]
        previous_x, previous_y = x0, y0
        commands = [f"M{x0} {y0}"]
        for x, y in corners[1:]:
            if y == previous_y:
                commands.append(f"H{x}")
            elif x == previous_x:
                commands.append(f"V{y}")
            else:
                commands.append(f"L{x} {y}")
            previous_x, previous_y = x, y
        commands.append("Z")
        parts.append("".join(commands))
    return "".join(parts)


def _alpha_band_ids(alpha: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Split one visible class by deterministic source-alpha quantiles."""
    values = np.asarray(alpha[selected], dtype=np.uint8)
    result = np.full(alpha.shape, -1, dtype=np.int16)
    if values.size == 0:
        return result
    unique = np.unique(values)
    if unique.size <= 1:
        result[selected] = 0
        return result

    quantiles = np.quantile(
        values.astype(np.float64),
        np.linspace(0.0, 1.0, _ALPHA_BANDS_PER_VISIBLE_CLASS + 1)[1:-1],
        method="nearest",
    ).astype(np.int16)
    thresholds = np.unique(quantiles)
    result[selected] = np.digitize(
        alpha[selected].astype(np.int16),
        thresholds,
        right=True,
    ).astype(np.int16)
    return result


def _straight_representative(
    source_rgba: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Least-squares straight RGB for the selected source-alpha pixels.

    White-composited visible RGB obeys V=a*S+255*(1-a). Minimizing visible
    squared error for a constant straight paint therefore weights source straight
    RGB by a^2. Alpha is still supplied solely by the existing source-alpha mask.
    """
    source = np.asarray(source_rgba, dtype=np.uint8)
    alpha = source[:, :, 3].astype(np.float64) / 255.0
    weights = np.square(alpha[selected])
    rgb = source[:, :, :3][selected].astype(np.float64)
    denominator = float(weights.sum())
    if denominator <= 1e-12:
        return np.zeros(3, dtype=np.uint8)
    value = (rgb * weights[:, None]).sum(axis=0) / denominator
    return np.rint(value).clip(0, 255).astype(np.uint8)


def build_boundary_stable_source_paint_group(
    root: ET.Element,
    source_rgba: np.ndarray,
    *,
    grid_width: int,
    grid_height: int,
) -> tuple[ET.Element, dict[str, Any]]:
    """Vectorize source-visible classes, refined only by source alpha bands."""
    from app.graph_source import canonical_segmentation  # noqa: PLC0415

    source = np.asarray(source_rgba, dtype=np.uint8)
    visible = _source_white(source)
    unique_visible = int(np.unique(visible.reshape(-1, 3), axis=0).shape[0])
    palette_size = max(1, min(8, unique_visible))
    labels, visible_palette = canonical_segmentation(visible, k=palette_size)
    alpha = source[:, :, 3]
    alpha_positive = alpha > 0

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    view_x, view_y, view_width, view_height = _viewbox(root)
    group = ET.Element(
        qname("g"),
        {
            "data-vektoryum-visible-alpha-paint": "source-visible-alpha-bands-v2",
            "transform": (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / float(grid_width):.12g} "
                f"{view_height / float(grid_height):.12g})"
            ),
        },
    )

    path_count = 0
    loop_count = 0
    used_class_count = 0
    used_band_count = 0
    for label in range(len(visible_palette)):
        class_mask = (labels == label) & alpha_positive
        if not bool(np.any(class_mask)):
            continue
        band_ids = _alpha_band_ids(alpha, class_mask)
        used_class_count += 1
        for band in sorted(int(value) for value in np.unique(band_ids[class_mask])):
            selected = class_mask & (band_ids == band)
            if not bool(np.any(selected)):
                continue
            loops = trace_cell_contours(selected)
            path_data = _loop_path_data(loops)
            if not path_data:
                continue
            color = _straight_representative(source, selected)
            ET.SubElement(
                group,
                qname("path"),
                {
                    "d": path_data,
                    "fill": "#{:02x}{:02x}{:02x}".format(
                        int(color[0]), int(color[1]), int(color[2])
                    ),
                    "fill-rule": "evenodd",
                    "data-vektoryum-visible-class": str(label),
                    "data-vektoryum-alpha-band": str(band),
                },
            )
            path_count += 1
            loop_count += len(loops)
            used_band_count += 1

    if path_count <= 0:
        raise RuntimeError("source_alpha_visible_paint_empty")
    return group, {
        "visible_segmentation_palette_size": int(len(visible_palette)),
        "visible_paint_path_count": int(path_count),
        "visible_paint_loop_count": int(loop_count),
        "visible_paint_used_label_count": int(used_class_count),
        "visible_paint_used_alpha_band_count": int(used_band_count),
        "visible_paint_alpha_bands_per_class": _ALPHA_BANDS_PER_VISIBLE_CLASS,
    }
