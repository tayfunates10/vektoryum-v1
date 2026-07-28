"""Contracts for the fail-closed compact alpha encoding ladder."""
from __future__ import annotations

import numpy as np

from app.alpha_candidate_knockout import (
    _quantize_alpha_step,
    budget_resilient_alpha_encodings,
)


def test_compact_ladder_preserves_legacy_order_and_descends() -> None:
    alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
    encodings = list(budget_resilient_alpha_encodings(alpha))
    names = [name for name, _array, _opacity in encodings]

    assert names[:2] == ["exact", "quantized_128"]
    assert names[-4:] == [
        "quantized_64",
        "quantized_32",
        "quantized_16",
        "quantized_8",
    ]

    unique_counts = [len(np.unique(array)) for _name, array, _opacity in encodings]
    assert unique_counts[-4] >= unique_counts[-3] >= unique_counts[-2] >= unique_counts[-1]


def test_quantization_preserves_transparent_opaque_and_support_fringe() -> None:
    alpha = np.array([[0, 1, 2, 127, 253, 254, 255]], dtype=np.uint8)

    for step in (4, 8, 16, 32):
        quantized, opacity = _quantize_alpha_step(alpha, step=step)
        assert int(quantized[0, 0]) == 0
        assert int(quantized[0, 1]) == 1
        assert int(quantized[0, 5]) == 254
        assert int(quantized[0, 6]) == 255
        assert 0 not in opacity
        assert opacity[1] == 1 / 255.0
        assert opacity[254] == 254 / 255.0
        assert opacity[255] == 1.0


def test_ordinary_alpha_error_is_bounded_by_half_step() -> None:
    alpha = np.arange(2, 254, dtype=np.uint8).reshape(14, 18)

    for step in (4, 8, 16, 32):
        quantized, _opacity = _quantize_alpha_step(alpha, step=step)
        delta = np.abs(quantized.astype(np.int16) - alpha.astype(np.int16))
        assert int(delta.max()) <= step // 2


def test_duplicate_compact_rasters_are_suppressed() -> None:
    alpha = np.array([[0, 1, 254, 255]], dtype=np.uint8)
    encodings = list(budget_resilient_alpha_encodings(alpha))
    payloads = [array.tobytes() for _name, array, _opacity in encodings]

    assert len(payloads) == len(set(payloads))
