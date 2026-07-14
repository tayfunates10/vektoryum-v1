"""Application package runtime bindings."""
from __future__ import annotations

# ``app.pipeline`` imports ``app.vector_engines`` immediately after package
# initialization, so loading that existing module here adds no new dependency.
# Centerline graph, scoring and quality modules remain completely lazy for every
# non-centerline request and benchmark category.
from app import vector_engines as _vector_engines


def _lazy_graph_centerline(*args, **kwargs):
    from app.centerline_svg import vectorize_skeleton_graph_to_svg  # noqa: PLC0415

    return vectorize_skeleton_graph_to_svg(*args, **kwargs)


_vector_engines.vectorize_skeleton_to_svg = _lazy_graph_centerline

del _vector_engines
