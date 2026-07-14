from __future__ import annotations

from hashlib import sha256
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from app.centerline_graph import trace_skeleton_graph, validate_centerline_report
from app.centerline_svg import read_centerline_report, vectorize_skeleton_graph_to_svg
from app.quality import basic_svg_quality_check
import app.scoring as scoring
import app.vector_engines as vector_engines


def _mask(height: int = 20, width: int = 20) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _assert_valid(result: dict) -> dict:
    report = result["report"]
    valid, errors = validate_centerline_report(report)
    assert valid is True, errors
    assert report["valid"] is True
    assert report["topology"]["edge_count"] == report["topology"]["serialized_edge_count"]
    return report["topology"]


def test_line_traces_as_one_open_path() -> None:
    mask = _mask()
    mask[7, 2:15] = 255

    result = trace_skeleton_graph(mask, min_branch=4)
    topology = _assert_valid(result)

    assert topology["path_count"] == 1
    assert topology["endpoint_count"] == 2
    assert topology["junction_count"] == 0
    assert result["paths"][0]["closed"] is False


def test_polyline_preserves_one_connected_open_path() -> None:
    mask = _mask()
    mask[4, 3:12] = 255
    mask[4:14, 11] = 255

    result = trace_skeleton_graph(mask, min_branch=3)
    topology = _assert_valid(result)

    assert topology["path_count"] == 1
    assert topology["endpoint_count"] == 2
    assert len(result["paths"][0]["points"]) >= 3


def test_t_junction_keeps_three_connected_branches() -> None:
    mask = _mask()
    mask[5, 3:14] = 255
    mask[5:15, 8] = 255

    topology = _assert_valid(trace_skeleton_graph(mask, min_branch=3))

    assert topology["path_count"] == 3
    assert topology["endpoint_count"] == 3
    assert topology["junction_count"] == 1


def test_cross_junction_keeps_four_connected_branches() -> None:
    mask = _mask()
    mask[9, 2:17] = 255
    mask[2:17, 9] = 255

    topology = _assert_valid(trace_skeleton_graph(mask, min_branch=3))

    assert topology["path_count"] == 4
    assert topology["endpoint_count"] == 4
    assert topology["junction_count"] == 1


def test_loop_is_serialized_once_without_outline_contouring() -> None:
    mask = _mask()
    mask[4, 4:15] = 255
    mask[14, 4:15] = 255
    mask[4:15, 4] = 255
    mask[4:15, 14] = 255

    result = trace_skeleton_graph(mask, min_branch=3)
    topology = _assert_valid(result)

    assert topology["path_count"] == 1
    assert topology["loop_count"] == 1
    assert topology["endpoint_count"] == 0
    assert result["paths"][0]["closed"] is True


def test_short_endpoint_spur_is_pruned_but_main_line_survives() -> None:
    mask = _mask()
    mask[10, 2:17] = 255
    mask[7:11, 9] = 255

    result = trace_skeleton_graph(mask, min_branch=4)
    topology = _assert_valid(result)

    assert topology["pruned_spur_count"] == 1
    assert topology["path_count"] == 1
    assert topology["endpoint_count"] == 2
    assert topology["junction_count"] == 0


def _source_image() -> np.ndarray:
    image = np.full((48, 64), 255, dtype=np.uint8)
    cv2.line(image, (6, 24), (58, 24), 0, 5)
    cv2.line(image, (32, 24), (32, 42), 0, 5)
    return image


def test_svg_output_is_deterministic_open_and_measured(tmp_path) -> None:
    source = tmp_path / "source.png"
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    cv2.imwrite(str(source), _source_image())

    report = vectorize_skeleton_graph_to_svg(source, first, {"min_branch": 4})
    vectorize_skeleton_graph_to_svg(source, second, {"min_branch": 4})

    assert sha256(first.read_bytes()).hexdigest() == sha256(second.read_bytes()).hexdigest()
    embedded = read_centerline_report(first)
    assert embedded == report
    assert embedded["backend"] == "opencv_skeleton_graph"
    assert embedded["measurement_available"] is True
    root = ET.parse(first).getroot()
    paths = [element for element in root.iter() if element.tag.split("}")[-1] == "path"]
    assert paths
    assert all("Z" not in (element.get("d") or "").upper() for element in paths)
    assert all((element.get("fill") or "").lower() == "none" for element in paths)


def test_empty_source_fails_closed_without_svg(tmp_path) -> None:
    source = tmp_path / "empty.png"
    output = tmp_path / "output.svg"
    cv2.imwrite(str(source), np.full((24, 24), 255, dtype=np.uint8))

    with pytest.raises(RuntimeError, match="centerline topology contract failed"):
        vectorize_skeleton_graph_to_svg(source, output)

    assert output.exists() is False


def test_candidate_score_exposes_backend_topology_and_confidence(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.svg"
    cv2.imwrite(str(source), _source_image())
    vectorize_skeleton_graph_to_svg(source, output, {"min_branch": 4})

    score = scoring.score_vector_candidate(
        original_path=source,
        svg_path=output,
        analysis_report={"estimated_color_count": 1},
        mode="centerline",
    )
    details = score["score_details"]

    assert details["centerline_backend"] == "opencv_skeleton_graph"
    assert details["centerline_measurement_available"] is True
    assert details["centerline_valid"] is True
    assert 0.0 <= details["centerline_confidence"] <= 1.0
    assert details["centerline_topology"]["edge_count"] > 0


def test_invalid_or_unmeasured_fallback_cannot_be_production_ready() -> None:
    invalid_report = {
        "schema_version": "centerline-graph-v1",
        "backend": "opencv_skeleton_graph",
        "measurement_available": False,
        "valid": False,
        "confidence": 1.0,
        "topology": {},
    }
    quality = basic_svg_quality_check(
        score_details={
            "path_count": 10,
            "node_count": 30,
            "unique_colors": 1,
            "has_bitmap": False,
            "centerline_backend": "opencv_skeleton_graph",
            "centerline_report": invalid_report,
            "centerline_measurement_available": False,
        },
        mode="centerline",
        total_score=100,
        fidelity_score=99,
    )

    assert quality["status"] == "needs_review"
    assert quality["metrics"]["centerline_valid"] is False
    assert any("production-ready" in warning for warning in quality["warnings"])


def test_public_candidate_dispatcher_uses_graph_fallback() -> None:
    assert vector_engines.vectorize_skeleton_to_svg is vectorize_skeleton_graph_to_svg
