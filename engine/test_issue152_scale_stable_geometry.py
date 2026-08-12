from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.source_palette_vector import vectorize_source_palette_paths


def _write_rect_source(path: Path) -> None:
    image = Image.new("RGB", (96, 96), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 83, 83), fill=(40, 40, 40))
    draw.rectangle((28, 28, 67, 67), fill=(150, 150, 150))
    image.save(path)


def test_axis_aligned_palette_geometry_is_scale_stable_and_vector_only(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.svg"
    _write_rect_source(source)
    report = vectorize_source_palette_paths(source, output, max_colors=6)
    text = output.read_text(encoding="utf-8").lower()
    assert report["scale_stable_eligible"] is True, report
    assert report["geometry_strategy"] == "semantic_cycle_fit", report
    assert report["fit_transaction"]["passed"] is True, report
    assert report["fit_strategy_counts"]["axis_aligned_rectangle"] >= 2, report
    assert report["raster_embedded"] is False
    assert "<image" not in text and "base64" not in text and "data:image" not in text


def test_production_module_contains_no_fixture_specific_routing() -> None:
    source = Path(__file__).with_name("app") / "source_palette_vector.py"
    text = source.read_text(encoding="utf-8").lower()
    for forbidden in (
        "qa-micro-component-ladder",
        "qa-small-details",
        "qa-thin-negative-space",
        "_draw_micro_component_ladder",
        "_draw_small_details",
        "_draw_thin_negative_space",
        "residual_multiscale_source",
        "output_quality_residual_suite",
    ):
        assert forbidden not in text


def test_palette_is_preserved_exactly_for_rectangular_classes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.svg"
    _write_rect_source(source)
    expected = {
        tuple(int(v) for v in row)
        for row in np.unique(np.asarray(Image.open(source)).reshape(-1, 3), axis=0)
    }
    report = vectorize_source_palette_paths(source, output, max_colors=6)
    text = output.read_text(encoding="utf-8").lower()
    for red, green, blue in expected:
        assert f"#{red:02x}{green:02x}{blue:02x}" in text
    assert report["source_color_count"] == len(expected)


@pytest.mark.parametrize("case_name", ["micro", "small", "thin"])
def test_issue152_target_sources_are_semantically_scale_stable(case_name: str, tmp_path: Path) -> None:
    """Test-only fixture probe; production remains fixture-agnostic."""
    from engine.regression.output_quality_residual_suite import (  # noqa: PLC0415
        _draw_micro_component_ladder,
        _draw_thin_negative_space,
    )
    from engine.regression.output_quality_suite import _draw_small_details  # noqa: PLC0415

    builders = {
        "micro": _draw_micro_component_ladder,
        "small": _draw_small_details,
        "thin": _draw_thin_negative_space,
    }
    builder = builders[case_name]
    source = tmp_path / f"{case_name}.png"
    output = tmp_path / f"{case_name}.svg"
    builder(256).convert("RGBA").save(source, "PNG", optimize=False)
    report = vectorize_source_palette_paths(source, output, max_colors=6)
    text = output.read_text(encoding="utf-8").lower()
    assert "<image" not in text and "base64" not in text and "data:image" not in text
    transaction = report.get("fit_transaction") or {}
    diagnostic = (
        f"{case_name}: reason={transaction.get('reason')} "
        f"strategies={report.get('fit_strategy_counts')} "
        f"native_counts={transaction.get('native_class_component_counts')} "
        f"render_counts={transaction.get('render_class_component_counts')} "
        f"native_visible={transaction.get('native_visible_residual')} "
        f"native_boundary={transaction.get('native_boundary_p95_px')} "
        f"native_component={transaction.get('native_component')}"
    )
    assert report.get("scale_stable_eligible") is True, diagnostic
