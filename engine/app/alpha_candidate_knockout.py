"""Compatibility wrapper binding compact candidate-knockout serialization.

The established implementation is retained byte-for-byte in
``app.alpha_candidate_knockout_base``. Only the reconstruction-tree serializer is
replaced; all alpha, structure, evaluator, journal and byte-budget gates remain
inside the original implementation and execute unchanged.
"""
from __future__ import annotations

from app import alpha_candidate_knockout_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

# Keep an immutable test/diagnostic reference before patching the base module.
_legacy_build_reconstruction_tree = _base._build_reconstruction_tree

from app.alpha_candidate_knockout_compact import (
    build_compact_knockout_reconstruction_tree,
)

_base._build_reconstruction_tree = build_compact_knockout_reconstruction_tree
_build_reconstruction_tree = build_compact_knockout_reconstruction_tree
