from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_painter import build_painter_reconstruction_tree


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
    assert stats["cumulative_threshold_count"] > 0
    assert stats["cumulative_command_count"] > 0
    assert stats["contour_path_count"] == stats["cumulative_threshold_count"]
    assert stats["contour_command_count"] == stats["cumulative_command_count"]
    assert len([node for node in candidate.iter() if str(node.tag).endswith("}mask")]) == 1
