from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from app.component_quality import (  # noqa: E402
    component_gate_applicability,
    gate_candidate_scores,
    has_open_required_cycle,
    measure_component_integrity_arrays,
)


def _micro_source(size: int = 256) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (28, 52), (212, 188), (10, 10, 10), thickness=-1)
    # Independent micro component: large enough to clear the existing 48px
    # support floor, small enough that global SSIM remains dominated by the body.
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


def test_pareto_dominated_high_global_candidate_cannot_win() -> None:
    unsafe_report = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "source_cc_recall": 0.75,
        "render_cc_precision": 1.0,
        "min_true_cc_iou": 0.0,
    }
    safe_report = {
        "applicable": True,
        "measured": True,
        "status": "pass",
        "source_cc_recall": 1.0,
        "render_cc_precision": 1.0,
        "min_true_cc_iou": 0.97,
    }

    unsafe = gate_candidate_scores(99.5, 99.7, unsafe_report)
    safe = gate_candidate_scores(96.0, 96.2, safe_report)

    assert unsafe["selection_disqualified"] is True
    assert safe["selection_disqualified"] is False
    assert safe["total_score"] > unsafe["total_score"]
    assert safe["fidelity_score"] > unsafe["fidelity_score"]


def test_all_failed_candidates_keep_legacy_score_ordering() -> None:
    failed = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "source_cc_recall": 0.5,
        "render_cc_precision": 0.5,
        "min_true_cc_iou": 0.5,
    }
    legacy_winner = gate_candidate_scores(93.94, 62.12, failed)
    legacy_runner_up = gate_candidate_scores(85.05, 33.28, failed)

    assert legacy_winner["selection_disqualified"] is True
    assert legacy_runner_up["selection_disqualified"] is True
    assert legacy_winner["total_score"] > legacy_runner_up["total_score"]
    assert legacy_winner["fidelity_score"] > legacy_runner_up["fidelity_score"]
    assert round(legacy_winner["total_score"] - legacy_runner_up["total_score"], 2) == 8.89


def test_missing_applicable_measurement_is_fail_closed_needs_review() -> None:
    missing = {
        "applicable": True,
        "measured": False,
        "status": "needs_review",
        "reason": "render_unavailable",
    }
    gated = gate_candidate_scores(100.0, 100.0, missing)

    assert gated["selection_disqualified"] is True
    assert gated["component_quality_status"] == "needs_review"
    assert gated["total_score"] < 0.0
    assert gated["fidelity_score"] < 0.0


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

    first = measure_component_integrity_arrays(source, render, k=2)
    second = measure_component_integrity_arrays(source.copy(), render.copy(), k=2)

    assert first == second


def test_open_filled_cycle_is_rejected_but_open_stroke_is_allowed(tmp_path: Path) -> None:
    open_fill = tmp_path / "open-fill.svg"
    open_fill.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14" fill="#000000"/></svg>',
        encoding="utf-8",
    )
    open_stroke = tmp_path / "open-stroke.svg"
    open_stroke.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14" fill="none" stroke="#000000"/></svg>',
        encoding="utf-8",
    )
    closed_fill = tmp_path / "closed-fill.svg"
    closed_fill.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<path d="M 2 2 L 14 2 L 8 14 Z" fill="#000000"/></svg>',
        encoding="utf-8",
    )

    assert has_open_required_cycle(open_fill) is True
    assert has_open_required_cycle(open_stroke) is False
    assert has_open_required_cycle(closed_fill) is False


def test_release_safety_failure_uses_same_band_as_component_failure() -> None:
    # Regression from the explicit single_color release corpus: the new CC gate
    # briefly promoted a raster-exact but geometrically open filled candidate.
    # Structural-release failure and measured CC failure must be in the same
    # disqualified band so, when no fully eligible candidate exists, the old
    # score ordering is preserved rather than promoting the open candidate.
    open_exact = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "reason": "open_required_cycle",
        "source_cc_recall": 1.0,
        "render_cc_precision": 1.0,
        "min_true_cc_iou": 1.0,
        "open_required_cycle": True,
    }
    closed_component_review = {
        "applicable": True,
        "measured": True,
        "status": "fail",
        "reason": "component_integrity_fail",
        "source_cc_recall": 1.0,
        "render_cc_precision": 1.0,
        "min_true_cc_iou": 0.9369,
        "open_required_cycle": False,
    }

    open_gated = gate_candidate_scores(81.26, 100.0, open_exact)
    closed_gated = gate_candidate_scores(86.90, 94.53, closed_component_review)

    assert open_gated["selection_disqualified"] is True
    assert closed_gated["selection_disqualified"] is True
    assert closed_gated["total_score"] > open_gated["total_score"]
    assert closed_gated["fidelity_score"] > open_gated["fidelity_score"]
