"""Application package runtime bindings."""
from __future__ import annotations

# The candidate dispatcher resolves this module global at call time. Binding the
# measured graph implementation here keeps the public engine name stable and
# also applies in spawned worker processes, where the package initializer runs
# before ``app.pipeline`` is imported.
from app import vector_engines as _vector_engines
from app.centerline_contracts import install_centerline_contracts as _install_contracts
from app.centerline_svg import vectorize_skeleton_graph_to_svg as _graph_centerline

_vector_engines.vectorize_skeleton_to_svg = _graph_centerline
_install_contracts()

del _graph_centerline, _install_contracts, _vector_engines
