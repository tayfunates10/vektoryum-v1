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
