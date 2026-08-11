from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from app.neutral_palette import (  # noqa: E402
    detect_neutral_luminance_bands,
    geometric_preserves_neutral_palette,
)


def _bands(values: list[tuple[int, int, int]], width: int = 48, height: int = 48) -> Image.Image:
    arr = np.zeros((height, width * len(values), 3), dtype=np.uint8)
    for idx, value in enumerate(values):
        arr[:, idx * width:(idx + 1) * width] = value
    return Image.fromarray(arr, mode="RGB")


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
