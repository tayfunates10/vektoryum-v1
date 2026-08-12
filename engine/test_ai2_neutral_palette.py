from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from ai2_acceptance_metrics import build_acceptance_report, validate_report  # noqa: E402
from app.analyzer import analyze_image_from_mem  # noqa: E402
from app.neutral_palette import (  # noqa: E402
    detect_neutral_luminance_bands,
    geometric_preserves_neutral_palette,
    restore_layered_neutral_svg_palette,
    route_auto_layered_neutral_mode,
)


def _bands(values: list[tuple[int, int, int]], width: int = 48, height: int = 48) -> Image.Image:
    arr = np.zeros((height, width * len(values), 3), dtype=np.uint8)
    for idx, value in enumerate(values):
        arr[:, idx * width:(idx + 1) * width] = value
    return Image.fromarray(arr, mode="RGB")


def _gray_counter_fixture(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 36, 232, 220), radius=38, fill=(150, 150, 150, 255))
    draw.rounded_rectangle((38, 50, 218, 206), radius=30, fill=(15, 15, 15, 255))
    draw.ellipse((86, 78, 170, 174), fill=(255, 255, 255, 255))
    draw.rectangle((122, 168, 144, 220), fill=(255, 255, 255, 255))
    return image


def test_layered_neutral_detector_requires_three_real_plateaus() -> None:
    report = detect_neutral_luminance_bands(
        _bands([(18, 18, 18), (104, 104, 104), (196, 196, 196), (250, 250, 250)])
    )

    assert report["layered_neutral"] is True
    assert report["band_count"] >= 3
    assert len(report["band_centers"]) == report["band_count"]


def test_black_white_antialias_ramp_is_not_layered_neutral() -> None:
    # A continuous B/W ramp has many grey values but no three flat luminance
    # plateaus; AA pixels must not manufacture a third source band.
    ramp = np.linspace(0, 255, 192, dtype=np.uint8)
    arr = np.repeat(ramp[None, :, None], 48, axis=0)
    arr = np.repeat(arr, 3, axis=2)
    report = detect_neutral_luminance_bands(Image.fromarray(arr, mode="RGB"))

    assert report["layered_neutral"] is False


def test_chromatic_flat_bands_are_not_neutral_bands() -> None:
    report = detect_neutral_luminance_bands(
        _bands([(230, 20, 20), (20, 230, 20), (20, 20, 230), (250, 250, 250)])
    )

    assert report["layered_neutral"] is False


def test_geometric_neutral_canonicalization_guard_is_narrow() -> None:
    layered = {"layered_neutral": True}
    not_layered = {"layered_neutral": False}

    assert geometric_preserves_neutral_palette("geometric_logo", layered) is True
    assert geometric_preserves_neutral_palette("geometric_logo", not_layered) is False
    assert geometric_preserves_neutral_palette("single_color", layered) is False
    assert geometric_preserves_neutral_palette("logo_color", layered) is False


def test_single_color_layered_neutral_auto_route_is_narrow() -> None:
    layered = {"layered_neutral": True}
    not_layered = {"layered_neutral": False}

    assert route_auto_layered_neutral_mode("single_color", layered) == "geometric_logo"
    assert route_auto_layered_neutral_mode("single_color", not_layered) == "single_color"
    assert route_auto_layered_neutral_mode("geometric_logo", layered) == "geometric_logo"
    assert route_auto_layered_neutral_mode("logo_color", layered) == "logo_color"
    assert route_auto_layered_neutral_mode("photo_poster", layered) == "photo_poster"


def test_gray_border_counter_is_not_left_on_binary_single_color_route() -> None:
    fixture = _gray_counter_fixture()
    neutral = detect_neutral_luminance_bands(fixture)
    analysis = analyze_image_from_mem(fixture)

    assert neutral["layered_neutral"] is True
    assert analysis["neutral_luminance_report"] is not None
    assert analysis["neutral_luminance_report"]["layered_neutral"] is True
    assert analysis["recommended_mode"] == "geometric_logo"
    assert analysis["detected_type"] == "geometric_logo"


def test_source_neutral_palette_restore_maps_only_neutral_fills(tmp_path: Path) -> None:
    source = _gray_counter_fixture()
    svg_path = tmp_path / "candidate.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
        '<path fill="#000000" d="M0 0h80v80H0z"/>'
        '<path fill="#959595" d="M80 0h80v80H80z"/>'
        '<path fill="#ffffff" d="M160 0h96v80h-96z"/>'
        '<path fill="#e02020" d="M0 80h80v80H0z"/>'
        '</svg>',
        encoding="utf-8",
    )

    report = restore_layered_neutral_svg_palette(
        svg_path, source, mode="geometric_logo"
    )
    text = svg_path.read_text(encoding="utf-8").lower()

    assert report["applied"] is True
    assert '#0f0f0f' in text
    assert '#969696' in text
    assert '#ffffff' in text
    assert '#e02020' in text


def test_source_neutral_palette_restore_is_noop_outside_geometric(tmp_path: Path) -> None:
    source = _gray_counter_fixture()
    svg_path = tmp_path / "candidate.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><path fill="#000000" d="M0 0h10v10z"/></svg>',
        encoding="utf-8",
    )

    report = restore_layered_neutral_svg_palette(svg_path, source, mode="logo_color")

    assert report["applied"] is False
    assert '#000000' in svg_path.read_text(encoding="utf-8").lower()


def test_neutral_tone_steps_fixture_is_deterministic() -> None:
    fixture = _bands(
        [(12, 12, 12), (72, 72, 72), (136, 136, 136), (204, 204, 204), (248, 248, 248)],
        width=52,
        height=64,
    )
    first = detect_neutral_luminance_bands(fixture)
    second = detect_neutral_luminance_bands(fixture.copy())

    assert first == second
    assert first["layered_neutral"] is True
    assert first["band_count"] >= 3


def test_issue_137_native_256_named_reproducers_meet_all_acceptance_axes(tmp_path: Path) -> None:
    report = build_acceptance_report(tmp_path)
    # Pytest displays warnings even on success; this makes the exact before/after
    # evidence auditable in the focused GitHub Actions log without changing the
    # workflow or depending on a mutable artifact upload step.
    warnings.warn(
        "AI2_ACCEPTANCE_METRICS=" + json.dumps(report, sort_keys=True),
        UserWarning,
        stacklevel=1,
    )

    assert validate_report(report) == []
    assert set(report["cases"]) == {
        "qa-lowres-badge",
        "qa-gray-border-counter",
        "neutral-tone-steps",
    }
    for case in report["cases"].values():
        assert case["fixture_kind"] == "canonical_reproducer_not_historical_asset"
        assert case["native_size"] == [256, 256]
        assert case["after"]["source_cc_recall"] == 1.0
        assert case["after"]["render_cc_precision"] == 1.0
        assert case["after"]["min_true_cc_iou"] >= 0.95
        assert case["after"]["visible_residual"] <= 0.01
        assert case["after"]["de00_p95"] <= 1.0
        assert case["after"]["boundary_p95_px"] <= 0.75


def test_issue_137_native_256_metrics_are_deterministic_across_two_loops(tmp_path: Path) -> None:
    first = build_acceptance_report(tmp_path)
    second = build_acceptance_report(tmp_path)

    assert first == second
