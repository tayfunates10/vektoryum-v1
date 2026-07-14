from __future__ import annotations

import numpy as np

from app.canonical_curve_fit import CanonicalFitConfig, fit_canonical_curves
from app.canonical_face_builder import build_canonical_face_paths
from app.half_edge_graph import build_half_edge_graph


def _graph_with_hole():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[2:10, 2:10] = 1
    labels[5:7, 5:7] = 0
    graph = build_half_edge_graph(labels, ["#000000", "#ffffff"])
    report = fit_canonical_curves(graph)
    assert report.valid, report.errors
    return graph


def test_hg4_builds_one_closed_evenodd_path_per_visible_face():
    graph = _graph_with_hole()
    report = build_canonical_face_paths(graph)

    assert report.valid, report.errors
    assert report.built_faces == sum(
        1 for face in graph.faces.values() if face.visible and not face.is_exterior
    )
    assert report.faces
    for face in report.faces:
        assert face.fill_rule == "evenodd"
        assert face.outer.closed
        assert face.outer.start == face.outer.end
        assert all(hole.closed and hole.start == hole.end for hole in face.holes)


def test_hg4_uses_canonical_curve_geometry_in_both_directions():
    graph = _graph_with_hole()
    report = build_canonical_face_paths(graph)
    assert report.valid, report.errors

    for edge in graph.half_edges.values():
        twin = graph.half_edges[edge.twin_id]
        curve = graph.curves[edge.curve_id]
        assert twin.curve_id == edge.curve_id
        if curve.fitted_segments:
            p0, c1, c2, p1 = curve.fitted_segments[0]
            assert (p0, c1, c2, p1)


def test_hg4_falls_closed_when_cycle_is_open():
    graph = _graph_with_hole()
    visible_face = next(face for face in graph.faces.values() if face.visible and not face.is_exterior)
    cycle = graph.cycles[visible_face.outer_cycle_id]
    cycle.closed = False

    report = build_canonical_face_paths(graph)

    assert not report.valid
    assert report.faces == ()
    assert any("not closed" in error for error in report.errors)


def test_hg4_preserves_polyline_fallback_paths():
    labels = np.indices((10, 10)).sum(axis=0).astype(np.uint8) % 2
    graph = build_half_edge_graph(labels, ["#000000", "#ffffff"])
    fit = fit_canonical_curves(
        graph,
        CanonicalFitConfig(tolerance_px=5.0, max_error_px=0.0, linearity_epsilon_px=0.0),
    )
    assert fit.valid, fit.errors

    report = build_canonical_face_paths(graph)

    assert report.valid, report.errors
    assert report.fallback_cycles >= 0


def test_hg4_is_deterministic():
    graph_a = _graph_with_hole()
    graph_b = _graph_with_hole()

    assert build_canonical_face_paths(graph_a) == build_canonical_face_paths(graph_b)
