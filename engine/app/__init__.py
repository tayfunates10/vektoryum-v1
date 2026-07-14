"""Application package runtime bindings."""
from __future__ import annotations

from functools import wraps

# ``app.pipeline`` imports ``app.vector_engines`` immediately after package
# initialization, so loading that existing module here adds no new dependency.
# Centerline graph, scoring and quality modules remain completely lazy for every
# non-centerline request and benchmark category.
from app import vector_engines as _vector_engines


def _lazy_graph_centerline(*args, **kwargs):
    from app.centerline_svg import vectorize_skeleton_graph_to_svg  # noqa: PLC0415

    return vectorize_skeleton_graph_to_svg(*args, **kwargs)


_vector_engines.vectorize_skeleton_to_svg = _lazy_graph_centerline

# AA-2 binds versioned confidence metadata at the package boundary instead of
# rewriting the mature analyzer decision tree. Every normal import of
# ``app.analyzer`` passes through package initialization first, so pipeline, API,
# tests and direct analyzer callers all receive the same contract. The wrapper
# preserves the existing result dictionary and never changes ``recommended_mode``.
from app import analyzer as _analyzer

if not getattr(_analyzer.analyze_image_from_mem, "__vektoryum_contract_wrapped__", False):
    _original_analyze_image_from_mem = _analyzer.analyze_image_from_mem

    @wraps(_original_analyze_image_from_mem)
    def _analyze_image_from_mem_with_contract(image):
        report = _original_analyze_image_from_mem(image)
        from app.analyzer_contracts import attach_analyzer_contract  # noqa: PLC0415

        return attach_analyzer_contract(report, image)

    _analyze_image_from_mem_with_contract.__vektoryum_contract_wrapped__ = True
    _analyzer.analyze_image_from_mem = _analyze_image_from_mem_with_contract


del _analyzer
del _vector_engines
