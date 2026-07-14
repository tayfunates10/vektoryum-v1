from __future__ import annotations

import numpy as np

from app.canonical_curve_fit import CanonicalFitConfig, fit_canonical_curves
from app.half_edge_graph import build_half_edge_graph


def _two_region_graph():
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:, 4:] = 1
    return build_half_edge_graph(labels, ["#000000", "#ffffff"])


def test_hg3_fits_each_canonical_curve_once_and_twins_share_it():
    graph = _two_region_graph()
    report = fit_canonical_curves(graph)

    assert report.valid, report.errors
    assert report.fitted_curves > 0
    for edge in graph.half_edges.values():
        twin = graph.half_edges[edge.twin_id]
        assert twin.curve_id == edge.curve_id
        assert graph.curves[twin.curve_id] is graph.curves[edge.curve_id]


def test_hg3_locks_fitted_endpoints_to_graph_vertices():
    graph = _two_region_graph()
    report = fit_canonical_curves(graph)

    assert report.valid, report.errors
    for curve in graph.curves.values():
        if not curve.fitted_segments:
            continue
        assert curve.fitted_segments[0][0] == graph.vertices[curve.start_vertex_id].point
        assert curve.fitted_segments[-1][3] == graph.vertices[curve.end_vertex_id].point


def test_hg3_falls_back_when_error_budget_is_impossible():
    labels = np.indices((10, 10)).sum(axis=0).astype(np.uint8) % 2
    graph = build_half_edge_graph(labels, ["#000000", "#ffffff"])
    report = fit_canonical_curves(
        graph,
        CanonicalFitConfig(tolerance_px=5.0, max_error_px=0.0, linearity_epsilon_px=0.0),
    )

    assert report.valid, report.errors
    assert report.fallback_curves >= 0
    assert all(np.isfinite(c.fit_error_max) for c in graph.curves.values())


def test_hg3_is_deterministic():
    graph_a = _two_region_graph()
    graph_b = _two_region_graph()
    report_a = fit_canonical_curves(graph_a)
    report_b = fit_canonical_curves(graph_b)

    assert report_a == report_b
    assert {
        k: (v.fitted_segments, v.fit_error_max, v.fit_error_p95, v.fit_fallback)
        for k, v in graph_a.curves.items()
    } == {
        k: (v.fitted_segments, v.fit_error_max, v.fit_error_p95, v.fit_fallback)
        for k, v in graph_b.curves.items()
    }
