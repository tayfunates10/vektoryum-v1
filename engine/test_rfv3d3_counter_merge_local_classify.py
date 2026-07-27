"""RFV-3D3 counter-merge yerel sınıflandırma sözleşmesi."""
from __future__ import annotations

import numpy as np

from app.counter_merge_local_classify import (
    _LocalClassificationCache,
    _wrap_merge_counters,
)
from app.palette_ops import classify_rgb


def test_local_crop_is_bit_identical_to_full_map_slice() -> None:
    rng = np.random.default_rng(1234567)
    rgb = rng.integers(0, 256, size=(87, 113, 3), dtype=np.uint8)
    fills = rng.integers(0, 256, size=(47, 3), dtype=np.uint8).astype(np.float32)
    expected = classify_rgb(rgb, fills)[11:72, 19:101]
    lazy = _LocalClassificationCache().classify(rgb, fills)
    actual = lazy[11:72, 19:101]
    assert np.array_equal(actual, expected)


def test_only_requested_crop_reaches_delegate() -> None:
    class RecordingDelegate:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def classify(self, rgb: np.ndarray, fills: np.ndarray) -> np.ndarray:
            self.shapes.append(rgb.shape)
            return classify_rgb(rgb, fills)

    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, size=(1200, 1600, 3), dtype=np.uint8)
    fills = rng.integers(0, 256, size=(128, 3), dtype=np.uint8).astype(np.float32)
    delegate = RecordingDelegate()
    lazy = _LocalClassificationCache(delegate).classify(rgb, fills)

    first = lazy[200:260, 400:490]
    second = lazy[200:260, 400:490]

    assert first.shape == (60, 90)
    assert second is first
    assert delegate.shapes == [(60, 90, 3)]
    assert (1200, 1600, 3) not in delegate.shapes


def test_wrapper_preserves_arguments_and_supplies_lazy_cache() -> None:
    seen: dict[str, object] = {}
    rgb = np.zeros((40, 50, 3), dtype=np.uint8)
    fills = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32)

    def original(
        svg_path,
        width,
        height,
        render_fn,
        source_rgb=None,
        max_merges=8,
        cache=None,
    ):
        seen.update({
            "svg_path": svg_path,
            "width": width,
            "height": height,
            "render_fn": render_fn,
            "source_rgb": source_rgb,
            "max_merges": max_merges,
            "cache": cache,
        })
        labels = cache.classify(source_rgb, fills)
        return {"crop": labels[4:12, 7:19]}

    render_fn = object()
    wrapped = _wrap_merge_counters(original)
    result = wrapped(
        "sample.svg",
        50,
        40,
        render_fn,
        source_rgb=rgb,
        max_merges=3,
        cache=None,
    )

    assert result["crop"].shape == (8, 12)
    assert seen["svg_path"] == "sample.svg"
    assert seen["width"] == 50
    assert seen["height"] == 40
    assert seen["render_fn"] is render_fn
    assert seen["source_rgb"] is rgb
    assert seen["max_merges"] == 3
    assert getattr(seen["cache"], "__vektoryum_local_counter_classify__", False)


def test_invalid_non_rectangular_access_fails_closed() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    fills = np.array([[0, 0, 0]], dtype=np.float32)
    lazy = _LocalClassificationCache().classify(rgb, fills)
    try:
        _ = lazy[3]
    except TypeError as exc:
        assert "iki boyutlu dilim" in str(exc)
    else:
        raise AssertionError("tek indeks erişimi fail-closed reddedilmeliydi")
