"""FAZ 3 source truth: straight/premultiplied alpha and background composition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def test_straight_premultiplied_roundtrip_visible_pixels() -> None:
    from app.source_truth import premultiply_rgba, unpremultiply_rgba

    rgba = np.array([
        [[227, 0, 11, 255], [20, 160, 240, 128]],
        [[80, 40, 200, 64], [99, 77, 55, 0]],
    ], dtype=np.uint8)
    restored = unpremultiply_rgba(premultiply_rgba(rgba))
    visible = rgba[:, :, 3] > 0
    error = np.abs(restored[:, :, :3].astype(int) - rgba[:, :, :3].astype(int))
    assert int(error[visible].max()) <= 2
    assert np.array_equal(restored[:, :, 3], rgba[:, :, 3])
    assert np.array_equal(restored[1, 1], np.array([0, 0, 0, 0], dtype=np.uint8))


def test_white_composite_and_alpha_recover_straight_color() -> None:
    from app.source_truth import composite_rgba, source_rgba_from_white_composite

    source = np.zeros((8, 8, 4), dtype=np.uint8)
    source[:, :, :3] = (227, 0, 11)
    source[:, :, 3] = 128
    white = composite_rgba(source, 255)
    restored = source_rgba_from_white_composite(white, source[:, :, 3])
    assert np.max(np.abs(restored[:, :, :3].astype(int) - source[:, :, :3].astype(int))) <= 1
    assert np.array_equal(restored[:, :, 3], source[:, :, 3])


def test_transparent_rgb_noise_does_not_change_appearance() -> None:
    from app.source_truth import composite_rgba

    clean = np.zeros((16, 16, 4), dtype=np.uint8)
    noisy = clean.copy()
    rng = np.random.default_rng(42)
    noisy[:, :, :3] = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    for background in (0, 255):
        assert np.array_equal(composite_rgba(clean, background), composite_rgba(noisy, background))


def test_checker_background_is_deterministic() -> None:
    from app.source_truth import checker_background

    first = checker_background(33, 47, cell=5)
    second = checker_background(33, 47, cell=5)
    assert np.array_equal(first, second)
    assert len(np.unique(first.reshape(-1, 3), axis=0)) == 2
