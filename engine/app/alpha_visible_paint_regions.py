"""Boundary-stable vector paint regions for Issue #145 visible-alpha repair.

The geometry is exactly one path per source-visible canonical class. Straight RGB
is selected by a deterministic classification search so the unchanged source alpha
mask composites each region back into the same visible class without adding path
commands or relaxing any budget.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _viewbox
from app.alpha_mask_contour import trace_cell_contours
from app.palette_ops import classify_rgb

_SVG_NS = "http://www.w3.org/2000/svg"
_MAX_COLOR_CANDIDATES = 64


def _source_white(rgba: np.ndarray) -> np.ndarray:
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3:4].astype(np.float64) / 255.0
    return np.clip(
        np.rint(
            array[:, :, :3].astype(np.float64) * alpha
            + 255.0 * (1.0 - alpha)
        ),
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


def _candidate_colors(source_rgb: np.ndarray, alpha: np.ndarray) -> list[np.ndarray]:
    rgb = np.asarray(source_rgb, dtype=np.uint8).reshape(-1, 3)
    a = np.asarray(alpha, dtype=np.uint8).reshape(-1)
    if rgb.size == 0:
        return [np.zeros(3, dtype=np.uint8)]

    tuples = [tuple(int(channel) for channel in row) for row in rgb]
    counts = Counter(tuples)
    frequent = sorted(counts, key=lambda value: (-counts[value], value))[
        :_MAX_COLOR_CANDIDATES
    ]
    candidates = [np.asarray(value, dtype=np.uint8) for value in frequent]

    weights = np.square(a.astype(np.float64) / 255.0)
    if float(weights.sum()) > 1e-12:
        mean = np.rint(
            (rgb.astype(np.float64) * weights[:, None]).sum(axis=0)
            / float(weights.sum())
        ).clip(0, 255).astype(np.uint8)
        candidates.append(mean)
    candidates.append(np.median(rgb, axis=0).round().clip(0, 255).astype(np.uint8))

    deduplicated: dict[tuple[int, int, int], np.ndarray] = {}
    for candidate in candidates:
        key = tuple(int(channel) for channel in candidate)
        deduplicated.setdefault(key, candidate)
    return [deduplicated[key] for key in sorted(deduplicated)]


def _classification_stable_color(
    source_rgba: np.ndarray,
    visible_rgb: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    label: int,
) -> tuple[np.ndarray, float]:
    selected = labels == int(label)
    source = np.asarray(source_rgba, dtype=np.uint8)
    alpha = source[:, :, 3][selected]
    source_rgb = source[:, :, :3][selected]
    target_visible = np.asarray(visible_rgb, dtype=np.uint8)[selected]
    a = alpha.astype(np.float64)[:, None] / 255.0

    best: tuple[float, float, tuple[int, int, int], np.ndarray] | None = None
    for candidate in _candidate_colors(source_rgb, alpha):
        composed = np.clip(
            np.rint(candidate.astype(np.float64)[None, :] * a + 255.0 * (1.0 - a)),
            0.0,
            255.0,
        ).astype(np.uint8)
        predicted = classify_rgb(
            composed.reshape((-1, 1, 3)),
            np.asarray(palette, dtype=np.float32),
        ).reshape(-1)
        agreement = float(np.mean(predicted == int(label)))
        mse = float(
            np.mean(
                np.square(
                    composed.astype(np.float64) - target_visible.astype(np.float64)
                )
            )
        )
        key = tuple(int(channel) for channel in candidate)
        score = (-agreement, mse, key, candidate)
        if best is None or score[:3] < best[:3]:
            best = score

    assert best is not None
    return best[3], float(-best[0])


def serialize_compact_visible_alpha_svg(root: ET.Element) -> bytes:
    """Serialize a repaired SVG without optional XML declaration bytes."""
    ET.register_namespace("", _SVG_NS)
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )


def build_boundary_stable_source_paint_group(
    root: ET.Element,
    source_rgba: np.ndarray,
    *,
    grid_width: int,
    grid_height: int,
) -> tuple[ET.Element, dict[str, Any]]:
    """Vectorize one geometry per canonical visible class with stable paint."""
    from app.graph_source import canonical_segmentation  # noqa: PLC0415

    source = np.asarray(source_rgba, dtype=np.uint8)
    visible = _source_white(source)
    unique_visible = int(np.unique(visible.reshape(-1, 3), axis=0).shape[0])
    palette_size = max(1, min(8, unique_visible))
    labels, visible_palette = canonical_segmentation(visible, k=palette_size)
    alpha_positive = source[:, :, 3] > 0

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    view_x, view_y, view_width, view_height = _viewbox(root)
    group = ET.Element(qname("g"))
    if (
        view_x != 0.0
        or view_y != 0.0
        or view_width != float(grid_width)
        or view_height != float(grid_height)
    ):
        group.set(
            "transform",
            (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / float(grid_width):.12g} "
                f"{view_height / float(grid_height):.12g})"
            ),
        )

    path_count = 0
    loop_count = 0
    agreements: list[float] = []
    for label in range(len(visible_palette)):
        selected = (labels == label) & alpha_positive
        if not bool(np.any(selected)):
            continue
        loops = trace_cell_contours(selected)
        path_data = _loop_path_data(loops)
        if not path_data:
            continue
        color, agreement = _classification_stable_color(
            source,
            visible,
            labels,
            visible_palette,
            label,
        )
        attributes = {
            "d": path_data,
            "fill": "#{:02x}{:02x}{:02x}".format(
                int(color[0]), int(color[1]), int(color[2])
            ),
        }
        if len(loops) > 1:
            attributes["fill-rule"] = "evenodd"
        ET.SubElement(group, qname("path"), attributes)
        path_count += 1
        loop_count += len(loops)
        agreements.append(agreement)

    if path_count <= 0:
        raise RuntimeError("source_alpha_visible_paint_empty")
    return group, {
        "visible_segmentation_palette_size": int(len(visible_palette)),
        "visible_paint_path_count": int(path_count),
        "visible_paint_loop_count": int(loop_count),
        "visible_paint_used_label_count": int(len(agreements)),
        "visible_paint_min_source_class_agreement": float(min(agreements)),
        "visible_paint_mean_source_class_agreement": float(np.mean(agreements)),
        "visible_paint_color_selection": "max_class_agreement_then_visible_mse_v1",
    }


# ``alpha_visible_paint`` is loaded before this module by the production alpha
# factory. Bind only its repair serializer; direct alpha helpers do not call it.
from app import alpha_visible_paint as _alpha_visible_paint  # noqa: E402

_alpha_visible_paint._serialized = serialize_compact_visible_alpha_svg
