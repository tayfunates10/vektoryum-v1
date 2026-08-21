from __future__ import annotations

import numpy as np
import pytest

from engine.regression.public15_support_topology_probe_v3 import _bind_selected_capture


def _snapshot(pixel_count: int, marker: int) -> dict[str, object]:
    return {
        "shape": [1, 1],
        "palette": [[marker, marker, marker]],
        "stats": {"paint_deficit_pixel_count": pixel_count},
        "signature": {"components": marker, "holes": 0},
        "labels_sha256": f"{marker:064x}",
        "labels": np.asarray([[marker]], dtype=np.int32),
    }


def test_binding_uses_selected_candidate_index_not_latest_snapshot() -> None:
    selected = {
        "index": 0,
        "geometry": {"paint_deficit_pixel_count": 943372},
    }
    captures = {
        0: _snapshot(943372, 1),
        1: _snapshot(943310, 2),
    }

    bound = _bind_selected_capture(selected, captures)

    assert bound["stats"]["paint_deficit_pixel_count"] == 943372
    assert bound["signature"]["components"] == 1


def test_binding_fails_closed_on_geometry_label_mismatch() -> None:
    selected = {
        "index": 0,
        "geometry": {"paint_deficit_pixel_count": 943372},
    }
    captures = {0: _snapshot(943310, 1)}

    with pytest.raises(RuntimeError, match="selected-label binding mismatch"):
        _bind_selected_capture(selected, captures)
