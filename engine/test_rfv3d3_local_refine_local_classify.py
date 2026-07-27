"""Bit-identical local palette classification for critical-component refinement."""
from __future__ import annotations

import numpy as np

from app.local_refine_local_classify import (
    _LocalRefineClassificationCache,
    _wrap_refine_critical_components,
)
from app.palette_ops import classify_rgb


class _RecordingDelegate:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def classify(self, rgb: np.ndarray, fills_rgb: np.ndarray) -> np.ndarray:
        self.shapes.append(tuple(rgb.shape))
        return classify_rgb(rgb, fills_rgb)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260727)
    rgb = rng.integers(0, 256, size=(31, 43, 3), dtype=np.uint8)
    fills = rng.integers(0, 256, size=(17, 3), dtype=np.uint8).astype(np.float32)
    return rgb, fills


def test_render_crop_is_bit_identical_and_only_crop_is_classified() -> None:
    rgb, fills = _fixture()
    delegate = _RecordingDelegate()
    cache = _LocalRefineClassificationCache(delegate, rgb)
    crop = (slice(4, 19), slice(7, 29))

    actual = cache.classify(rgb, fills)[crop]
    expected = classify_rgb(rgb, fills)[crop]

    np.testing.assert_array_equal(actual, expected)
    assert delegate.shapes == [(15, 22, 3)]


def test_source_crop_is_bit_identical_and_memoized_per_crop() -> None:
    rgb, fills = _fixture()
    delegate = _RecordingDelegate()
    cache = _LocalRefineClassificationCache(delegate, rgb)
    labels = cache.classify_source(fills)
    crop = (slice(3, 14), slice(5, 21))

    first = labels[crop]
    second = labels[crop]
    expected = classify_rgb(rgb, fills)[crop]

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert delegate.shapes == [(11, 16, 3)]


def test_wrapper_preserves_arguments_and_adapts_cache() -> None:
    rgb, fills = _fixture()
    delegate = _RecordingDelegate()
    seen = {}

    def original(svg_path, source_rgb, width, height, render_fn, cache=None):
        seen.update({
            "svg_path": svg_path,
            "source_is_same": source_rgb is rgb,
            "width": width,
            "height": height,
            "render_fn": render_fn,
            "cache": cache,
        })
        crop = (slice(1, 6), slice(2, 9))
        labels = cache.classify_source(fills)[crop]
        np.testing.assert_array_equal(labels, classify_rgb(rgb, fills)[crop])
        return {"status": "no_change"}

    render_fn = object()
    wrapped = _wrap_refine_critical_components(original)
    result = wrapped("candidate.svg", rgb, 43, 31, render_fn, cache=delegate)

    assert result == {"status": "no_change"}
    assert seen["svg_path"] == "candidate.svg"
    assert seen["source_is_same"] is True
    assert seen["width"] == 43 and seen["height"] == 31
    assert seen["render_fn"] is render_fn
    assert isinstance(seen["cache"], _LocalRefineClassificationCache)
    assert delegate.shapes == [(5, 7, 3)]
