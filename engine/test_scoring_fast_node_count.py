from __future__ import annotations

import pytest

from app.geometry_cleanup import extract_points_from_path_data
from app.scoring import _count_path_points


@pytest.mark.parametrize(
    "path_data",
    [
        "M0 0 L10 10 Z",
        "m1 2 3 4 l5 6 7 8 h9 10 v11 12 z",
        "M0 0 10 10 20 20 L30 30 40 40",
        "M0 0 H10 20 30 V40 50 Z",
        "M0 0 C1 2 3 4 5 6 S7 8 9 10 Q11 12 13 14 T15 16 A2 3 0 0 1 17 18 Z",
        "M0 0 L10",
        "M1e2 -2.5e-1 l.5 .75 h-3.2 v4.1",
        "M0,0L1,1M2,2h3v4z",
    ],
)
def test_fast_node_count_matches_legacy_point_expansion(path_data: str) -> None:
    legacy = sum(
        len(subpath.get("points", []))
        for subpath in extract_points_from_path_data(path_data)
    )
    assert _count_path_points(path_data) == legacy


def test_curves_do_not_add_legacy_scoring_nodes() -> None:
    path_data = "M0 0 C10 0 10 10 20 10 S30 20 40 20 Q50 30 60 20 T80 20 A5 5 0 0 1 90 30"
    assert _count_path_points(path_data) == 1
