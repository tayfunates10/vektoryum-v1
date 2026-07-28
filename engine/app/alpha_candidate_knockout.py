"""Compatibility wrapper binding compact candidate-knockout serialization.

The established implementation is retained byte-for-byte in
``app.alpha_candidate_knockout_base``. The wrapper replaces only two bounded
extension points:

* reconstruction-tree serialization uses the established compact ``<use>`` form;
* when the legacy exact/128-level source alpha exceeds the unchanged byte budget,
  progressively smaller 64/32/16/8-level candidates are tried in that order and
  each must pass every existing alpha, evaluator, structure and byte gate.

No byte, alpha, structure, evaluator or journal threshold is changed.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np

from app import alpha_candidate_knockout_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# Keep immutable test/diagnostic references before patching the base module.
_legacy_build_reconstruction_tree = _base._build_reconstruction_tree
_legacy_alpha_encodings = _base._alpha_encodings

from app.alpha_candidate_knockout_compact import (
    build_compact_knockout_reconstruction_tree,
)


def _quantize_alpha_step(
    alpha: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Quantize alpha without deleting the 1/254 support fringe.

    Exact transparent, exact opaque and one-unit support pixels remain exact.
    Ordinary samples move by at most half a quantization step, so every fallback
    remains subject to the unchanged direct-alpha and FinalArtifactEvaluator gates.
    """
    if step < 2:
        raise ValueError("alpha quantization step must be >= 2")
    source = np.asarray(alpha, dtype=np.uint8)
    wide = source.astype(np.uint16)
    quantized = (((wide + step // 2) // step) * step).clip(0, 255).astype(np.uint8)

    quantized[source == 0] = 0
    quantized[source == 255] = 255
    quantized[(source > 0) & (quantized == 0)] = 1
    quantized[(source < 255) & (quantized == 255)] = 254

    opacity_by_level = {
        int(value): int(value) / 255.0
        for value in np.unique(quantized)
        if int(value) > 0
    }
    return quantized, opacity_by_level


def _encoding_identity(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def budget_resilient_alpha_encodings(
    alpha: np.ndarray,
) -> Iterator[tuple[str, np.ndarray, dict[int, float]]]:
    """Yield authoritative legacy encodings, then bounded compact fallbacks.

    Legacy candidates retain their order and bytes. Compact candidates are only
    reached after earlier candidates fail a later gate, normally the unchanged
    byte budget. Duplicate rasters are suppressed deterministically.
    """
    seen: set[str] = set()
    for name, quantized, opacity_by_level in _legacy_alpha_encodings(alpha):
        identity = _encoding_identity(quantized)
        if identity in seen:
            continue
        seen.add(identity)
        yield name, quantized, opacity_by_level

    for name, step in (
        ("quantized_64", 4),
        ("quantized_32", 8),
        ("quantized_16", 16),
        ("quantized_8", 32),
    ):
        quantized, opacity_by_level = _quantize_alpha_step(alpha, step=step)
        identity = _encoding_identity(quantized)
        if identity in seen:
            continue
        seen.add(identity)
        yield name, quantized, opacity_by_level


_base._build_reconstruction_tree = build_compact_knockout_reconstruction_tree
_base._alpha_encodings = budget_resilient_alpha_encodings
_build_reconstruction_tree = build_compact_knockout_reconstruction_tree
_alpha_encodings = budget_resilient_alpha_encodings
