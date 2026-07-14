from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.canonical_curve_fit import fit_canonical_curves
from app.canonical_face_builder import build_canonical_face_paths
from app.half_edge_graph import build_half_edge_graph
from app.shadow_svg_serializer import serialize_shadow_svg_paths


def _built_graph():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[2:10, 2:10] = 1
    labels[5:7, 5:7] = 0
    graph = build_half_edge_graph(labels, ["#000000", "#ffffff"])
    fit = fit_canonical_curves(graph)
    assert fit.valid, fit.errors
    build = build_canonical_face_paths(graph)
    assert build.valid, build.errors
    return graph, build


def test_hg5_serializes_complete_evenodd_shadow_payloads():
    graph, build = _built_graph()

    report = serialize_shadow_svg_paths(graph, build)

    assert report.valid, report.errors
    assert report.serialized_faces == build.built_faces
    assert report.serialized_cycles == build.built_cycles
    assert report.faces
    for face in report.faces:
        assert face.fill_rule == "evenodd"
        assert face.path_data.startswith("M ")
        assert face.path_data.endswith("Z")
        assert len(face.digest_sha256) == 64
        assert "-0" not in face.path_data


def test_hg5_serializes_holes_as_multiple_closed_subpaths():
    graph, build = _built_graph()

    report = serialize_shadow_svg_paths(graph, build)

    assert report.valid, report.errors
    face_with_hole = next(face for face in build.faces if face.holes)
    serialized = next(face for face in report.faces if face.face_id == face_with_hole.face_id)
    assert serialized.path_data.count("M ") == 1 + len(face_with_hole.holes)
    assert serialized.path_data.count(" Z") == 1 + len(face_with_hole.holes)


def test_hg5_is_deterministic_and_precision_bounded():
    graph_a, build_a = _built_graph()
    graph_b, build_b = _built_graph()

    report_a = serialize_shadow_svg_paths(graph_a, build_a, precision=3)
    report_b = serialize_shadow_svg_paths(graph_b, build_b, precision=3)

    assert report_a == report_b
    assert all(".000" not in face.path_data for face in report_a.faces)


def test_hg5_fails_closed_on_invalid_build_report():
    graph, build = _built_graph()
    invalid = replace(build, valid=False, errors=("forced invalid build",))

    report = serialize_shadow_svg_paths(graph, invalid)

    assert not report.valid
    assert report.faces == ()
    assert report.serialized_faces == 0
    assert "forced invalid build" in report.errors


def test_hg5_fails_closed_on_visible_face_coverage_mismatch():
    graph, build = _built_graph()
    incomplete = replace(build, faces=build.faces[:-1], built_faces=max(0, build.built_faces - 1))

    report = serialize_shadow_svg_paths(graph, incomplete)

    assert not report.valid
    assert report.faces == ()
    assert any("coverage mismatch" in error for error in report.errors)


def test_hg5_fails_closed_on_invalid_fill_color():
    graph, build = _built_graph()
    visible = next(face for face in graph.faces.values() if face.visible and not face.is_exterior)
    visible.fill_color = "not-a-color"

    report = serialize_shadow_svg_paths(graph, build)

    assert not report.valid
    assert report.faces == ()
    assert any("invalid fill color" in error for error in report.errors)
