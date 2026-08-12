"""Issue #153 post-refit topology and bounded alpha-parent regression contracts."""

from __future__ import annotations

import numpy as np

from app.component_quality import measure_component_integrity_arrays


def test_post_refit_strict_measurement_exposes_micro_fragment_hidden_by_legacy_floor() -> None:
    """Boundary mutations must not hide a newly invented two-pixel component."""
    source = np.full((128, 128, 3), 255, dtype=np.uint8)
    source[24:104, 24:64] = (220, 35, 45)
    source[24:104, 64:104] = (30, 90, 220)

    rendered = source.copy()
    rendered[70:72, 90] = (220, 35, 45)

    ordinary = measure_component_integrity_arrays(source, rendered)
    strict = measure_component_integrity_arrays(
        source,
        rendered,
        min_component_pixels=2,
        min_component_fraction=0.0,
    )

    assert ordinary["status"] == "pass"
    assert strict["status"] == "fail"
    assert strict["render_cc_precision"] < 1.0
    assert strict["min_component_area"] == 2


def test_source_rgb_palette_guard_detects_semantic_colour_island() -> None:
    from app.component_quality import measure_discrete_rgb_palette_component_integrity_arrays

    source = np.full((128, 128, 3), 255, dtype=np.uint8)
    source[24:104, 24:64] = (220, 35, 45)
    source[24:104, 64:104] = (30, 90, 220)
    source[92:104, 42:86] = (245, 195, 25)
    identity = measure_discrete_rgb_palette_component_integrity_arrays(source, source.copy())
    assert identity["status"] == "pass"
    assert identity["render_cc_precision"] == 1.0

    invented = source.copy()
    invented[70:72, 90] = (220, 35, 45)
    report = measure_discrete_rgb_palette_component_integrity_arrays(source, invented)
    assert report["status"] == "fail"
    assert report["source_cc_recall"] == 1.0
    assert report["render_cc_precision"] < 1.0
    assert report["measurement_space"] == "source_snapped_rgb_palette"


def test_alpha_parent_selection_has_no_source_level_count_ceiling() -> None:
    """Continuous alpha may probe parents, while the candidate trial count stays bounded."""
    module = __import__("app.alpha_parent_selection", fromlist=["_select"])
    source = __import__("inspect").getsource(module._select)
    assert "len(levels) >" not in source
    assert "len(levels) <= 1" in source
    assert module._MAX_TRIALS == 4
