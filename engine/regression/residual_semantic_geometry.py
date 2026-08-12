"""Semantic geometry helpers for AI-4 Round 2 residual diagnostics.

Palette segmentation is useful for colour residuals, but it is not a reliable
source of topology when two semantic regions intentionally share or collapse to
the same rendered tone, or when alpha compositing creates colours that are not
semantic regions. This module keeps colour residuals untouched and provides a
separate geometry-only view from the verified analytic fixture labels.
"""
from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from app.graph_source import canonical_segmentation
from app.palette_ops import classify_rgb
from app.source_truth import composite_rgba
from engine.regression import residual_error_metrics as base
from engine.regression.residual_topology_metrics import measure_label_topology

SEMANTIC_GEOMETRY_POLICY_VERSION = "vektoryum-semantic-geometry-v1"


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
    excluded_source_labels: Iterable[int] = (),
) -> tuple[np.ndarray, int, list[int]]:
    source = np.asarray(source_labels, dtype=np.uint8).copy()
    background = _background_label(source)
    excluded = sorted({int(value) for value in excluded_source_labels})
    for value in excluded:
        source[source == value] = np.uint8(background)
    return source, background, excluded


def _semantic_palette(source_rgba: np.ndarray, source_labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    rgb = composite_rgba(np.asarray(source_rgba, dtype=np.uint8), 255)
    labels = sorted(int(value) for value in np.unique(source_labels))
    centers: list[np.ndarray] = []
    for label in labels:
        pixels = rgb[source_labels == label]
        if pixels.size == 0:
            centers.append(np.zeros(3, dtype=np.float32))
        else:
            centers.append(np.median(pixels.astype(np.float32), axis=0).astype(np.float32))
    return np.stack(centers, axis=0), labels


def _spatially_disambiguate_components(
    predicted_labels: np.ndarray,
    source_labels: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[np.ndarray, int]:
    """Resolve only high-confidence disconnected colour-collapses spatially.

    A light interior can legitimately render with exactly the same RGB as the
    exterior background. It is still a separate connected component, so >=95%
    overlap with one analytic semantic region recovers its geometry without
    inventing missing adjacent regions.
    """
    predicted = np.asarray(predicted_labels, dtype=np.uint8)
    source = np.asarray(source_labels, dtype=np.uint8)
    result = predicted.copy()
    reassigned = 0
    for predicted_label in sorted(int(value) for value in np.unique(predicted)):
        count, component_map = cv2.connectedComponents(
            (predicted == predicted_label).astype(np.uint8),
            connectivity=8,
        )
        for component_id in range(1, count):
            mask = component_map == component_id
            source_values, source_counts = np.unique(source[mask], return_counts=True)
            if source_counts.size == 0:
                continue
            winner_index = int(np.argmax(source_counts))
            winner = int(source_values[winner_index])
            ratio = float(source_counts[winner_index] / source_counts.sum())
            if winner != predicted_label and ratio >= float(confidence):
                result[mask] = np.uint8(winner)
                reassigned += 1
    return result, reassigned


def map_discrete_render_to_semantics(
    source_rgba: np.ndarray,
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
    *,
    excluded_source_labels: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source, background, excluded = _effective_labels(source_labels, excluded_source_labels)
    centers, semantic_ids = _semantic_palette(source_rgba, source)
    render_rgb = composite_rgba(np.asarray(render_rgba, dtype=np.uint8), 255)
    nearest = classify_rgb(render_rgb, centers).astype(np.int64)
    ids = np.asarray(semantic_ids, dtype=np.uint8)
    predicted = ids[nearest]
    mapped, reassigned = _spatially_disambiguate_components(predicted, source)
    return source, mapped, {
        "policy_version": SEMANTIC_GEOMETRY_POLICY_VERSION,
        "mapping": "analytic_semantic_palette_plus_spatial_components",
        "background_label": int(background),
        "semantic_label_ids": semantic_ids,
        "excluded_source_labels": excluded,
        "reassigned_connected_components": int(reassigned),
    }


def map_continuous_render_to_semantics(
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
    *,
    excluded_source_labels: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map continuous-tone render clusters onto analytic semantic regions.

    Internal gradient/alpha colour bands are not topology. Render clusters are
    split into connected components, then each component is assigned by maximum
    spatial overlap with one analytic semantic region. A border-touching render
    background component stays background so a missing region cannot be restored
    from source truth.
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
        "mapping": "continuous_render_clusters_to_analytic_semantics",
        "background_label": int(background),
        "semantic_label_ids": [int(value) for value in np.unique(source)],
        "excluded_source_labels": excluded,
        "render_cluster_count": int(k),
        "mapped_connected_components": int(mapped_regions),
        "extra_connected_components": int(extra_regions),
    }


def _geometry_metrics(source_labels: np.ndarray, render_labels: np.ndarray) -> dict[str, Any]:
    component = base.connected_component_fidelity(source_labels, render_labels)
    boundary = base._boundary_distance(
        base._label_boundaries(source_labels),
        base._label_boundaries(render_labels),
    )
    topology = measure_label_topology(source_labels, render_labels)
    topology["semantic_geometry_policy_version"] = SEMANTIC_GEOMETRY_POLICY_VERSION
    return {"component": component, "boundary": boundary, "topology": topology}


def measure_discrete_semantic_geometry(
    source_rgba: np.ndarray,
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
) -> dict[str, Any]:
    source, rendered, mapping = map_discrete_render_to_semantics(
        source_rgba,
        render_rgba,
        source_labels,
    )
    result = _geometry_metrics(source, rendered)
    result["mapping"] = mapping
    result["topology_observability"] = "semantic_regions_observable"
    return result


def measure_continuous_hard_shape_components(
    render_rgba: np.ndarray,
    source_labels: np.ndarray,
    *,
    excluded_source_labels: Iterable[int],
) -> dict[str, Any]:
    source, rendered, mapping = map_continuous_render_to_semantics(
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
