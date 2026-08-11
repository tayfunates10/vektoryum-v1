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
