from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from app.component_quality import (  # noqa: E402
    _hungarian_minimize,
    component_gate_applicability,
    gate_candidate_scores,
    has_open_required_cycle,
    measure_component_integrity_arrays,
)
from app.pipeline import select_best  # noqa: E402
from app.scoring import score_vector_candidate  # noqa: E402


def _micro_source(size: int = 256) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (28, 52), (212, 188), (10, 10, 10), thickness=-1)
    cv2.circle(image, (226, 30), 6, (10, 10, 10), thickness=-1)
    return image


def test_source_component_loss_is_detected_even_when_global_area_is_tiny() -> None:
    source = _micro_source()
    render = source.copy()
    cv2.circle(render, (226, 30), 8, (255, 255, 255), thickness=-1)
    report = measure_component_integrity_arrays(source, render, k=2)
    assert report["measured"] is True
    assert report["status"] == "fail"
    assert report["source_cc_recall"] < 1.0
    assert report["render_cc_precision"] == 1.0
    assert report["min_true_cc_iou"] == 0.0


def test_fake_render_component_reduces_precision() -> None:
    source = _micro_source()
    render = source.copy()
    cv2.circle(render, (24, 226), 6, (10, 10, 10), thickness=-1)
    report = measure_component_integrity_arrays(source, render, k=2)
    assert report["measured"] is True
    assert report["status"] == "fail"
    assert report["source_cc_recall"] == 1.0
    assert report["render_cc_precision"] < 1.0


def test_fragmented_component_cannot_hide_behind_global_similarity() -> None:
    source = _micro_source()
    render = source.copy()
    render[52:189, 118:124] = 255
    report = measure_component_integrity_arrays(source, render, k=2)
    assert report["measured"] is True
    assert report["status"] == "fail"
    assert report["render_cc_precision"] < 1.0
    assert report["min_true_cc_iou"] < 0.95


def test_exact_component_preservation_passes_all_three_axes() -> None:
    source = _micro_source()
    report = measure_component_integrity_arrays(source, source.copy(), k=2)
    assert report["status"] == "pass"
    assert report["source_cc_recall"] == 1.0
    assert report["render_cc_precision"] == 1.0
    assert report["min_true_cc_iou"] == 1.0


def _optimal_pairs(weights: list[list[float]]) -> list[tuple[int, int]]:
    max_weight = max(max(row) for row in weights)
    return _hungarian_minimize([[max_weight - value for value in row] for row in weights])


def test_optimal_assignment_beats_greedy_adversarial_2x2() -> None:
    weights = [[0.90, 0.80], [0.85, 0.10]]
    pairs = _optimal_pairs(weights)
    assert set(pairs) == {(0, 1), (1, 0)}
    assert sum(weights[i][j] for i, j in pairs) == 1.65


def test_optimal_assignment_beats_greedy_adversarial_3x3() -> None:
    weights = [[0.90, 0.80, 0.10], [0.80, 0.10, 0.10], [0.10, 0.70, 0.60]]
    pairs = _optimal_pairs(weights)
    assert set(pairs) == {(0, 1), (1, 0), (2, 2)}
    assert round(sum(weights[i][j] for i, j in pairs), 2) == 2.20


def test_component_gate_keeps_measured_scores_numerically_pure() -> None:
    failed = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "source_cc_recall": 0.5,
        "render_cc_precision": 0.5,
        "min_true_cc_iou": 0.5,
    }
    gated = gate_candidate_scores(93.94, 62.12, failed)
    assert gated["selection_disqualified"] is True
    assert gated["selection_safe"] is False
    assert gated["total_score"] == 93.94
    assert gated["fidelity_score"] == 62.12


def test_safe_candidate_pool_blocks_pareto_dominated_unsafe_winner() -> None:
    unsafe = {
        "name": "unsafe_high",
        "total_score": 99.5,
        "fidelity_score": 99.7,
        "rendered_ok": True,
        "selection_safe": False,
        "selection_disqualified": True,
        "score_details": {"path_count": 10, "edge_f1": 1.0, "has_bitmap": False},
    }
    safe = {
        "name": "safe_lower",
        "total_score": 96.0,
        "fidelity_score": 96.2,
        "rendered_ok": True,
        "selection_safe": True,
        "selection_disqualified": False,
        "score_details": {"path_count": 10, "edge_f1": 1.0, "has_bitmap": False},
    }
    chosen, _raw, _reason = select_best([unsafe, safe], "single_color")
    assert chosen["name"] == "safe_lower"


def test_all_failed_candidates_preserve_legacy_order_without_safe_alternative() -> None:
    failed = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "source_cc_recall": 0.5,
        "render_cc_precision": 0.5,
        "min_true_cc_iou": 0.5,
    }
    hi_gate = gate_candidate_scores(93.94, 62.12, failed)
    lo_gate = gate_candidate_scores(85.05, 33.28, failed)
    hi = {
        "name": "legacy_hi", "total_score": hi_gate["total_score"],
        "fidelity_score": hi_gate["fidelity_score"], "rendered_ok": True,
        **{k: hi_gate[k] for k in ("selection_safe", "selection_disqualified")},
        "score_details": {"path_count": 10, "edge_f1": 0.8, "has_bitmap": False},
    }
    lo = {
        "name": "legacy_lo", "total_score": lo_gate["total_score"],
        "fidelity_score": lo_gate["fidelity_score"], "rendered_ok": True,
        **{k: lo_gate[k] for k in ("selection_safe", "selection_disqualified")},
        "score_details": {"path_count": 10, "edge_f1": 0.8, "has_bitmap": False},
    }
    # Scores remain unchanged for diagnostics. With no safe alternative the
    # established wrapper keeps legacy availability; a safe alternative, when
    # present, is tested above and always excludes this unsafe high scorer.
    chosen, _raw, _reason = select_best([lo, hi], "single_color")
    assert chosen["name"] == "legacy_hi"
    assert round(hi["total_score"] - lo["total_score"], 2) == 8.89


def test_missing_applicable_measurement_is_fail_closed_without_score_corruption() -> None:
    missing = {
        "applicable": True,
        "measured": False,
        "status": "needs_review",
        "reason": "render_unavailable",
    }
    gated = gate_candidate_scores(100.0, 100.0, missing)
    assert gated["selection_disqualified"] is True
    assert gated["selection_safe"] is False
    assert gated["component_quality_status"] == "needs_review"
    assert gated["total_score"] == 100.0
    assert gated["fidelity_score"] == 100.0


def test_photo_and_gradient_applicability_is_explicit_and_safe() -> None:
    photo = component_gate_applicability("photo_poster", {"has_gradient": False})
    gradient = component_gate_applicability("logo_color", {"has_gradient": True})
    flat_logo = component_gate_applicability(
        "geometric_logo", {"has_gradient": False, "likely_photo_or_complex": False}
    )
    assert photo == {"applicable": False, "reason": "photo_or_complex_input"}
    assert gradient == {"applicable": False, "reason": "gradient_input"}
    assert flat_logo == {"applicable": True, "reason": "palette_like_input"}


def test_component_metrics_are_deterministic_across_two_loops() -> None:
    source = _micro_source()
    render = source.copy()
    cv2.circle(render, (24, 226), 6, (10, 10, 10), thickness=-1)
    assert measure_component_integrity_arrays(source, render, k=2) == measure_component_integrity_arrays(
        source.copy(), render.copy(), k=2
    )


def test_open_filled_cycle_is_diagnostic_but_open_stroke_is_allowed(tmp_path: Path) -> None:
    open_fill = tmp_path / "open-fill.svg"
    open_fill.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14" fill="#000000"/></svg>', encoding="utf-8"
    )
    open_stroke = tmp_path / "open-stroke.svg"
    open_stroke.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14" fill="none" stroke="#000000"/></svg>', encoding="utf-8"
    )
    closed_fill = tmp_path / "closed-fill.svg"
    closed_fill.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14 Z" fill="#000000"/></svg>', encoding="utf-8"
    )
    assert has_open_required_cycle(open_fill) is True
    assert has_open_required_cycle(open_stroke) is False
    assert has_open_required_cycle(closed_fill) is False


def test_filled_cycle_parser_accepts_implicit_curve_and_relative_endpoint_closure(tmp_path: Path) -> None:
    implicit_curve = tmp_path / "implicit-curve.svg"
    implicit_curve.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path d="M 2 2 C 8 0 14 0 18 2 C 18 10 10 18 2 2" fill="#000"/></svg>', encoding="utf-8"
    )
    relative_closed = tmp_path / "relative-closed.svg"
    relative_closed.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path d="m 2 2 l 12 0 l -6 12 l -6 -12" fill="#000"/></svg>', encoding="utf-8"
    )
    relative_open = tmp_path / "relative-open.svg"
    relative_open.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path d="m 2 2 l 12 0 l -6 12" fill="#000"/></svg>', encoding="utf-8"
    )
    assert has_open_required_cycle(implicit_curve) is False
    assert has_open_required_cycle(relative_closed) is False
    assert has_open_required_cycle(relative_open) is True


def test_score_vector_candidate_does_not_mutate_svg_bytes(tmp_path: Path) -> None:
    source = np.zeros((256, 256, 3), dtype=np.uint8)
    source[:, :85] = (15, 15, 15)
    source[:, 85:170] = (150, 150, 150)
    source[:, 170:] = (255, 255, 255)
    source_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_path)
    svg_path = tmp_path / "candidate.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
        '<path fill="#000000" d="M0 0H85V256H0Z"/>'
        '<path fill="#969696" d="M85 0H170V256H85Z"/>'
        '<path fill="#ffffff" d="M170 0H256V256H170Z"/>'
        '</svg>', encoding="utf-8"
    )
    before = hashlib.sha256(svg_path.read_bytes()).hexdigest()
    score_vector_candidate(
        source_path,
        svg_path,
        {"estimated_color_count": 3, "has_gradient": False, "likely_photo_or_complex": False},
        "geometric_logo",
        geometry_report={"report": {
            "straight_edge_score": 1.0,
            "corner_cleanliness_score": 1.0,
            "axis_alignment_score": 1.0,
            "geometry_score": 1.0,
        }},
    )
    after = hashlib.sha256(svg_path.read_bytes()).hexdigest()
    assert after == before


def test_strict_post_refit_measurement_detects_two_pixel_invented_island() -> None:
    original = np.full((128, 128, 3), 255, dtype=np.uint8)
    original[24:104, 24:64] = (220, 35, 45)
    original[24:104, 64:104] = (30, 90, 220)
    rendered = original.copy()
    rendered[70:72, 90] = (220, 35, 45)

    legacy = measure_component_integrity_arrays(original, rendered)
    strict = measure_component_integrity_arrays(
        original, rendered, min_component_pixels=2, min_component_fraction=0.0,
    )
    assert legacy["status"] == "pass"
    assert strict["status"] == "fail"
    assert strict["render_cc_precision"] < 1.0
    assert strict["min_component_area"] == 2
