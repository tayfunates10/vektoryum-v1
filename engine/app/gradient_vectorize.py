"""Compatibility wrapper adding chromatic-edge segmentation to gradients.

The original gradient vectorizer remains byte-for-byte in
``app.gradient_vectorize_base``. This module re-exports its namespace and replaces
only region segmentation: luminance Canny edges are combined with RGB-channel
edges so equal-luminance hard color boundaries are not misclassified as smooth
gradients. Slow, genuinely continuous gradients remain below the per-pixel edge
threshold and keep the existing linear-gradient path.
"""
from __future__ import annotations

import cv2
import numpy as np

from app import gradient_vectorize_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy_edge_based_segments = _base._edge_based_segments


def _nearest_region_fill(labels: np.ndarray) -> np.ndarray:
    edge_pixels = labels == 0
    if edge_pixels.any() and (~edge_pixels).any():
        _distance, nearest = cv2.distanceTransformWithLabels(
            edge_pixels.astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        seed_to_region = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
        seed_to_region[nearest[~edge_pixels]] = labels[~edge_pixels]
        labels[edge_pixels] = seed_to_region[nearest[edge_pixels]]
    return labels


def _edge_based_segments(rgb: np.ndarray) -> np.ndarray:
    """Segment on luminance plus hard chromatic edges.

    Channel Canny catches blue/red and other isoluminant boundaries that vanish in
    grayscale. Thresholds are intentionally above the derivative of a broad smooth
    gradient, so continuous artwork still forms one region and is modelled by the
    existing gradient fitter.
    """
    smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=30, sigmaSpace=30)
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    channel_edges = np.zeros_like(edges)
    for channel in cv2.split(smooth):
        channel_edges = cv2.bitwise_or(
            channel_edges,
            cv2.Canny(channel, 28, 84),
        )

    # Require a meaningful local RGB jump as a second chromatic signal. It closes
    # weak Canny gaps without turning gradual per-pixel ramps into region borders.
    work = smooth.astype(np.int16)
    jump_x = np.max(np.abs(work[:, 1:] - work[:, :-1]), axis=2)
    jump_y = np.max(np.abs(work[1:, :] - work[:-1, :]), axis=2)
    hard_jump = np.zeros_like(edges)
    hard_jump[:, 1:][jump_x >= 18] = 255
    hard_jump[1:, :][jump_y >= 18] = 255

    edges = cv2.bitwise_or(edges, channel_edges)
    edges = cv2.bitwise_or(edges, hard_jump)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    non_edge = (edges == 0).astype(np.uint8)
    _count, labels = cv2.connectedComponents(non_edge, connectivity=4)
    return _nearest_region_fill(labels.astype(np.int32))


_base._edge_based_segments = _edge_based_segments
