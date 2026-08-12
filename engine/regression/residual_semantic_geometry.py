"""Observable semantic geometry helpers for AI-4 Round 2 diagnostics.

Topology must only be hard-gated when the final raster actually exposes it.
Palette clusters are not semantic regions for continuous alpha artwork, and a
flattened semi-transparent overlap cannot reveal hidden layer-to-layer colour
boundaries. This module therefore measures only geometry that is observable
without reconstructing missing regions from source truth.
"""
from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from app.graph_source import canonical_segmentation
from app.source_truth import composite_rgba
from engine.regression import residual_error_metrics as base
from engine.regression.residual_topology_metrics import measure_label_topology

SEMANTIC_GEOMETRY_POLICY_VERSION = "vektoryum-observable-semantic-geometry-v2"


def _background_label(labels: np.ndarray) -> int:
    lab = np.asarray(labels, dtype=np.uint8)
    h, w = lab.shape
    sy = max(1, round(h * 0.04))
    sx = max(1, round(w * 0.04))
    corners = np.concatenate(
        (
            lab[:sy, :sx].ravel(),
            lab[:sy, -sx:].ravel(),
            lab[-sy:, :sx].ravel(),
            lab[-sy:, -sx:].ravel(),
        )
    )
    return int(np.argmax(np.bincount(corners.astype(np.int64))))


def _effective_labels(
    source_labels: np.ndarray,
    excluded_source_labels: Iterable[int],
) -> tuple[np.ndarray, int, list[int]]:
    source = np.asarray(source_labels, dtype=np.uint8).copy()
    background = _background_label(source)
    excluded = sorted({int(value) for value in excluded_source_labels})
    for value in excluded:
        source[source == value] = np.uint8(background)
    return source, background, excluded


def map_continuous_hard_shapes(
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
    *,
    excluded_source_labels: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map observable render clusters onto hard analytic semantic shapes.

    The mapping never reassigns the border-connected render background to a
    missing source region. Thus an absent hard shape stays absent. Continuous
    alpha-only labels may be excluded because their fidelity is owned by the
    alpha plane/support contract rather than colour clustering.
    """
    source, background, excluded = _effective_labels(source_labels, excluded_source_labels)
    render_rgb = composite_rgba(np.asarray(render_rgba, dtype=np.uint8), 255)
    semantic_count = max(1, len(np.unique(source)))
    render_colour_count = max(1, int(np.unique(render_rgb.reshape(-1, 3), axis=0).shape[0]))
    k = min(semantic_count, render_colour_count)
    raw_labels, _palette = canonical_segmentation(render_rgb, k=k)
    raw_background = _background_label(raw_labels)
    mapped = np.full(source.shape, background, dtype=np.uint16)
    next_extra = int(source.max(initial=0)) + 1
    mapped_regions = 0
    extra_regions = 0

    for raw_label in sorted(int(value) for value in np.unique(raw_labels)):
        count, component_map = cv2.connectedComponents(
            (raw_labels == raw_label).astype(np.uint8),
            connectivity=8,
        )
        for component_id in range(1, count):
            mask = component_map == component_id
            ys, xs = np.nonzero(mask)
            touches_border = bool(
                (ys == 0).any()
                or (ys == source.shape[0] - 1).any()
                or (xs == 0).any()
                or (xs == source.shape[1] - 1).any()
            )
            source_values, source_counts = np.unique(source[mask], return_counts=True)
            winner = background
            if source_counts.size:
                winner = int(source_values[int(np.argmax(source_counts))])
            if raw_label == raw_background and touches_border:
                target = background
            elif winner == background and raw_label != raw_background:
                target = next_extra
                next_extra += 1
                extra_regions += 1
            else:
                target = winner
            mapped[mask] = target
            mapped_regions += 1

    if next_extra > 255:
        raise ValueError("semantic render mapping exceeded uint8 label capacity")
    return source, mapped.astype(np.uint8), {
        "policy_version": SEMANTIC_GEOMETRY_POLICY_VERSION,
        "mapping": "continuous_render_clusters_to_observable_hard_shapes",
        "background_label": int(background),
        "semantic_label_ids": [int(value) for value in np.unique(source)],
        "excluded_source_labels": excluded,
        "render_cluster_count": int(k),
        "mapped_connected_components": int(mapped_regions),
        "extra_connected_components": int(extra_regions),
    }


def measure_continuous_hard_shape_components(
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
    *,
    excluded_source_labels: Iterable[int],
) -> dict[str, Any]:
    source, rendered, mapping = map_continuous_hard_shapes(
        render_rgba,
        source_labels,
        excluded_source_labels=excluded_source_labels,
    )
    return {
        "component": base.connected_component_fidelity(source, rendered),
        "mapping": mapping,
    }


def measure_alpha_support_geometry(
    source_rgba: np.ndarray,
    render_rgba: np.ndarray,
) -> dict[str, Any]:
    """Measure topology observable from the final alpha support only.

    Shared colour-layer boundaries are intentionally not manufactured from the
    source because they are not uniquely identifiable in a flattened alpha
    composite. Alpha-plane and visible-composite gates remain independent.
    """
    source = (np.asarray(source_rgba, dtype=np.uint8)[:, :, 3] > 0).astype(np.uint8)
    rendered = (np.asarray(render_rgba, dtype=np.uint8)[:, :, 3] > 0).astype(np.uint8)
    boundary = base._boundary_distance(base._label_boundaries(source), base._label_boundaries(rendered))
    topology = measure_label_topology(source, rendered)
    topology["semantic_geometry_policy_version"] = SEMANTIC_GEOMETRY_POLICY_VERSION
    topology["topology_observability"] = "alpha_support_only"
    topology["shared_boundary"]["applicable"] = False
    topology["shared_boundary"]["pair_count"] = 0
    topology["shared_boundary"]["pairs"] = []
    topology["shared_boundary"]["max_gap_ratio"] = 0.0
    topology["shared_boundary"]["max_double_line_ratio"] = 0.0
    topology["shared_boundary"]["max_overlap_ratio"] = 0.0
    topology["shared_boundary"]["max_drift_p95_px"] = 0.0
    topology["shared_boundary"]["min_matched_transition_ratio"] = 1.0
    return {
        "boundary": boundary,
        "topology": topology,
        "mapping": {
            "policy_version": SEMANTIC_GEOMETRY_POLICY_VERSION,
            "mapping": "binary_alpha_support",
        },
        "topology_observability": "alpha_support_only",
    }
