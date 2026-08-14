from __future__ import annotations

from pathlib import Path

import cv2
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


def _component_stats(labels: np.ndarray, color_count: int) -> list[list[dict[str, object]]]:
    result: list[list[dict[str, object]]] = []
    for color_index in range(color_count):
        count, _lab, stats, _centroids = cv2.connectedComponentsWithStats(
            (labels == color_index).astype(np.uint8), 8
        )
        items = []
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            items.append({"bbox": [x, y, w, h], "area": area})
        result.append(sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0])))
    return result


def _thin_core_64_diagnostic(builder, tmp_path: Path) -> dict[str, object]:
    from app import semantic_region_geometry_core as core  # noqa: PLC0415
    from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
    from app.source_palette_vector import _nearest_palette_labels, _svg_document  # noqa: PLC0415

    source256 = np.asarray(builder(256).convert("RGB"), dtype=np.uint8)
    colors, inverse = np.unique(source256.reshape(-1, 3), axis=0, return_inverse=True)
    labels = inverse.reshape(256, 256)
    semantic = core.build_semantic_region_elements(labels, colors)
    svg_path = tmp_path / "thin-native-good-core.svg"
    svg_path.write_text(_svg_document(256, 256, list(semantic["elements"])), encoding="utf-8")
    rendered64 = render_svg_to_rgb(svg_path, 64, 64)
    source64 = np.asarray(builder(64).convert("RGB"), dtype=np.uint8)
    if rendered64 is None:
        return {"render": "unavailable"}
    source_labels = _nearest_palette_labels(source64, colors)
    render_labels = _nearest_palette_labels(rendered64, colors)
    return {
        "colors": colors.tolist(),
        "source64": _component_stats(source_labels, len(colors)),
        "core_render64": _component_stats(render_labels, len(colors)),
        "core_strategies": semantic.get("strategy_counts"),
    }


def _calibration_diagnostic(builder) -> dict[str, object]:
    """Read cached production calibration reports; this is test-only evidence."""
    from app.semantic_region_geometry import build_semantic_region_elements  # noqa: PLC0415

    source256 = np.asarray(builder(256).convert("RGB"), dtype=np.uint8)
    colors, inverse = np.unique(source256.reshape(-1, 3), axis=0, return_inverse=True)
    labels = inverse.reshape(256, 256)
    semantic = build_semantic_region_elements(labels, colors)
    keys = (
        "five_scale_parent_calibration",
        "five_scale_source_primitive_calibration",
        "five_scale_contract_refinement",
        "five_scale_source_phase_refinement",
        "five_scale_final_bounded_refinement",
        "five_scale_nested_seam_stabilization",
        "scale_topology_repair",
    )
    return {key: semantic.get(key) for key in keys if semantic.get(key) is not None}


def _issue152_multiscale_failures(svg_path: Path, builder) -> tuple[list[str], dict[str, object]]:
    """Test-only mirror of the fail-closed Issue #152 analytic scale axes."""
    from engine.regression import residual_measurement_contract as contract  # noqa: PLC0415
    from engine.regression import residual_multiscale_source as multiscale  # noqa: PLC0415

    multiscale.measure_residual_error = contract.measure_residual_error
    evidence = multiscale.measure_multiscale_svg_from_base(svg_path, builder, base_size=256)
    failures: list[str] = []
    if not bool(evidence.get("all_scales_measured")):
        failures.append("all_scales_measured")
    if not bool(evidence.get("source_contract_verified")):
        failures.append("source_contract_verified")
    for level in evidence.get("levels") or []:
        scale = str(level.get("scale"))
        checks = (
            ("source_cc_recall", float(level.get("source_component_recall") or 0.0) == 1.0),
            ("render_cc_precision", float(level.get("render_component_precision") or 0.0) == 1.0),
            ("small_component_recall", float(level.get("small_component_recall") or 0.0) == 1.0),
            ("min_true_component_iou", float(level.get("min_component_iou") or 0.0) >= 0.95),
            ("normalized_boundary_excess", float(level.get("normalized_boundary_excess_px") or 0.0) <= 0.25),
            ("visible_residual", float(level.get("visible_error_ratio") or 0.0) <= 0.01),
            ("de00_p95", float(level.get("de00_p95") or 0.0) <= 2.3),
        )
        failures.extend(f"{scale}_{name}" for name, passed in checks if not passed)
    return failures, evidence


def test_axis_aligned_palette_geometry_is_scale_stable_and_vector_only(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.svg"
    _write_rect_source(source)
    report = vectorize_source_palette_paths(source, output, max_colors=6)
    text = output.read_text(encoding="utf-8").lower()
    assert report["scale_stable_eligible"] is True, report
    assert report["geometry_strategy"] == "semantic_region_fit", report
    assert report["fit_transaction"]["passed"] is True, report
    assert report["fit_strategy_counts"]["axis_aligned_rectangle"] >= 2, report
    assert report["raster_embedded"] is False
    assert "<image" not in text and "base64" not in text and "data:image" not in text


def test_production_modules_contain_no_fixture_specific_routing() -> None:
    app_dir = Path(__file__).with_name("app")
    text = "\n".join(
        (app_dir / name).read_text(encoding="utf-8").lower()
        for name in ("source_palette_vector.py", "semantic_region_geometry.py")
    )
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
    """Strict test-only Issue #152 gate; production remains fixture-agnostic."""
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
    core64 = _thin_core_64_diagnostic(builder, tmp_path) if case_name == "thin" else None
    calibration = _calibration_diagnostic(builder)
    diagnostic = (
        f"{case_name}: reason={transaction.get('reason')} "
        f"strategies={report.get('fit_strategy_counts')} "
        f"regions={report.get('semantic_regions')} "
        f"source_counts={transaction.get('source_class_component_counts')} "
        f"native_counts={transaction.get('native_class_component_counts')} "
        f"levels={transaction.get('levels')} "
        f"native_visible={transaction.get('native_visible_residual')} "
        f"native_component={transaction.get('native_component')} "
        f"calibration={calibration} core64={core64} selected_svg={text}"
    )
    assert report.get("scale_stable_eligible") is True, diagnostic
    failures, evidence = _issue152_multiscale_failures(output, builder)
    assert not failures, f"{diagnostic} issue152_failures={failures} evidence={evidence}"
