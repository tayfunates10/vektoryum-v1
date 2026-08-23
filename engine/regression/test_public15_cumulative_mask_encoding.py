from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_painter import (
    _cumulative_compact_subpaths,
    _rectilinear_subpaths,
    _requantize_alpha,
    build_painter_reconstruction_tree,
)


def test_cumulative_mask_encoding_reports_budgeted_path_stats() -> None:
    """Cumulative mask geometry must remain visible to the existing path/node budget."""
    ns = "http://www.w3.org/2000/svg"
    root = ET.Element(
        f"{{{ns}}}svg",
        {"width": "4", "height": "4", "viewBox": "0 0 4 4"},
    )
    ET.SubElement(
        root,
        f"{{{ns}}}rect",
        {"x": "0", "y": "0", "width": "4", "height": "4", "fill": "#336699"},
    )
    quantized = np.asarray(
        [[1, 1, 2, 2], [1, 1, 2, 2], [0, 1, 2, 3], [0, 1, 3, 3]],
        dtype=np.int32,
    )
    opacity = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0}

    candidate, stats = build_painter_reconstruction_tree(
        root,
        None,
        quantized,
        opacity,
        1.0,
        mask_encoding="cumulative",
        transaction_id="test",
    )

    assert stats["reconstruction_mask_encoding"] == "cumulative"
    assert stats["cumulative_threshold_count"] == 3
    assert stats["cumulative_command_count"] == 3 * stats["cumulative_loop_count"]
    assert stats["contour_path_count"] == stats["cumulative_threshold_count"]
    assert stats["contour_command_count"] == stats["cumulative_command_count"]
    assert len([node for node in candidate.iter() if str(node.tag).endswith("}mask")]) == 1


def test_cumulative_compact_path_preserves_exact_loop_vertices() -> None:
    """Implicit lineto syntax must reduce commands without changing loop geometry."""
    loops = [
        [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
        [(4, 4), (5, 4), (5, 5), (4, 5), (4, 4)],
    ]

    compact, compact_commands = _cumulative_compact_subpaths(loops)
    _legacy, legacy_commands = _rectilinear_subpaths(loops)

    assert compact == "M0 0L2 0 2 2 0 2ZM4 4L5 4 5 5 4 5Z"
    assert compact_commands == 6
    assert compact_commands < legacy_commands


def test_cumulative_requantization_keeps_multiple_alpha_levels() -> None:
    """Every fallback, including coarse q4/q3, must stay multi-level, never silhouette."""
    alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for levels in (32, 16, 8, 4, 3):
        quantized, opacity = _requantize_alpha(
            alpha, levels, allow_silhouette_shortcut=False
        )
        positive_levels = [level for level in opacity if int(level) > 0]
        assert len(positive_levels) > 1
        assert len(np.unique(quantized[alpha > 0])) > 1
