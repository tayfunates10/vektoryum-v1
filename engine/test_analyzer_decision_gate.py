from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PIL import Image

import app
import app.analyzer as analyzer
import app.analyzer_decision_gate as gate
import app.pipeline as pipeline
from app.analyzer_decision_gate import (
    REVIEW_FALLBACK_MODE,
    apply_auto_decision_to_final_artifact,
    bind_precomputed_analysis,
    consume_precomputed_analysis,
    decide_trace_mode,
    reset_precomputed_analysis,
    verify_stored_contract,
)
from app.analyzer_runtime import (
    clear_job_auto_decision,
    register_job_auto_decision,
    take_final_svg_auto_decision,
)


def _geometric_logo() -> Image.Image:
    arr = np.full((420, 640, 3), 255, dtype=np.uint8)
    arr[35:385, 35:65] = 0
    arr[35:385, 575:605] = 0
    arr[35:65, 35:605] = 0
    arr[355:385, 35:605] = 0
    arr[110:280, 130:310] = (230, 25, 35)
    arr[100:320, 365:520] = 0
    return Image.fromarray(arr, "RGB")


def _analysis(monkeypatch) -> tuple[Image.Image, dict]:
    monkeypatch.setattr(analyzer, "calculate_semantic_edge_stats", lambda _image: None)
    image = _geometric_logo()
    report = analyzer.analyze_image_from_mem(image)
    assert report["analyzer_contract"]["status"] == "valid"
    return image, report


def test_verified_recommendation_is_accepted(monkeypatch) -> None:
    image, report = _analysis(monkeypatch)
    rebuilt, errors = verify_stored_contract(report, image)
    assert not errors
    assert rebuilt is not None
    monkeypatch.setattr(gate, "MIN_AUTO_CONFIDENCE", 0.0)
    monkeypatch.setattr(gate, "MIN_AUTO_MARGIN", -1.0)
    decision = decide_trace_mode(report, image, "auto")
    assert decision["status"] == "accepted"
    assert decision["execution_mode"] == report["recommended_mode"]
    assert decision["fallback_applied"] is False
    assert decision["reason_codes"] == ["verified_recommendation"]


def test_review_keeps_verified_recommendation(monkeypatch) -> None:
    image, report = _analysis(monkeypatch)
    monkeypatch.setattr(gate, "MIN_AUTO_CONFIDENCE", 1.01)
    decision = decide_trace_mode(report, image, "auto")
    assert decision["status"] == "needs_review"
    assert decision["execution_mode"] == report["recommended_mode"]
    assert decision["fallback_applied"] is False
    assert "confidence_below_minimum" in decision["reason_codes"]


def test_unverified_report_uses_review_fallback(monkeypatch) -> None:
    image, base = _analysis(monkeypatch)
    cases = []
    first = copy.deepcopy(base)
    first["analyzer_contract"]["source_pixel_sha256"] = "0" * 64
    cases.append((first, "digest_mismatch:source_pixel_sha256"))
    second = copy.deepcopy(base)
    second["analyzer_contract"]["feature_digest"] = "1" * 64
    cases.append((second, "digest_mismatch:feature_digest"))
    third = copy.deepcopy(base)
    third["recommendation_digest"] = "2" * 64
    cases.append((third, "top_level_mismatch:digest"))
    fourth = copy.deepcopy(base)
    fourth["analyzer_contract"]["confidence"] = None
    fourth["recommendation_confidence"] = None
    cases.append((fourth, "metadata_mismatch:confidence"))
    fifth = copy.deepcopy(base)
    fifth["recommended_mode"] = "centerline"
    fifth["detected_type"] = "centerline"
    cases.append((fifth, "recommendation_unsupported"))

    for report, expected in cases:
        decision = decide_trace_mode(report, image, "auto")
        assert decision["status"] == "needs_review"
        assert decision["execution_mode"] == REVIEW_FALLBACK_MODE
        assert decision["fallback_applied"] is True
        assert expected in decision["reason_codes"]


def test_manual_mode_is_unchanged(monkeypatch) -> None:
    image, report = _analysis(monkeypatch)
    report.pop("analyzer_contract")
    decision = decide_trace_mode(report, image, "centerline")
    assert decision["status"] == "manual"
    assert decision["execution_mode"] == "centerline"
    assert decision["reason_codes"] == ["manual_mode_bypass"]


def test_precomputed_report_is_single_use(monkeypatch) -> None:
    _image, report = _analysis(monkeypatch)
    token = bind_precomputed_analysis(report)
    try:
        consumed = consume_precomputed_analysis()
        assert consumed == report
        assert consumed is not report
        assert consume_precomputed_analysis() is None
    finally:
        reset_precomputed_analysis(token)


def test_job_record_matches_only_final_export(tmp_path) -> None:
    job_dir = tmp_path / "job123"
    job_dir.mkdir()
    decision = {"status": "needs_review", "execution_mode": "geometric_logo"}
    register_job_auto_decision(job_dir, decision)
    try:
        assert take_final_svg_auto_decision(job_dir / "candidate.svg") is None
        assert take_final_svg_auto_decision(job_dir / "job123.svg") == decision
        assert take_final_svg_auto_decision(job_dir / "job123.svg") is None
    finally:
        clear_job_auto_decision(job_dir)


def test_final_result_is_review_when_auto_decision_requires_review() -> None:
    analysis = {
        "auto_decision": {
            "status": "needs_review",
            "execution_mode": "geometric_logo",
            "reason_codes": ["margin_below_minimum"],
        }
    }
    final = {
        "verdict": "production_ready",
        "quality_verdict": "production_ready",
        "soft_warnings": [],
        "soft_warning_codes": [],
    }
    updated = apply_auto_decision_to_final_artifact(final, analysis, "auto")
    assert updated["verdict"] == "needs_review"
    assert updated["quality_verdict"] == "needs_review"
    assert "analyzer_auto_review" in updated["soft_warning_codes"]


def test_pipeline_uses_verified_mode_for_review(monkeypatch, tmp_path) -> None:
    image, prepared = _analysis(monkeypatch)
    observed_modes: list[str] = []

    def fake_core(image, original_path, trace_mode, job_dir, refine=True, edge_cleanup=True):
        observed_modes.append(trace_mode)
        return {
            "analysis": analyzer.analyze_image_from_mem(image),
            "mode_used": trace_mode,
            "mode_warning": None,
            "preprocess_report": {},
            "results": [],
            "scored": [],
            "best": None,
            "raw_best": None,
            "selection_reason": "test",
            "refine_info": {},
            "refit_info": {},
            "transform_journal": None,
            "structure_report": None,
        }

    monkeypatch.setattr(app, "_original_run_pipeline", fake_core)
    monkeypatch.setattr(app, "_analysis_entry", lambda _image: copy.deepcopy(prepared))
    monkeypatch.setattr(gate, "MIN_AUTO_CONFIDENCE", 0.0)
    monkeypatch.setattr(gate, "MIN_AUTO_MARGIN", -1.0)

    accepted = pipeline.run_pipeline(image, Path("source.png"), "auto", tmp_path / "accepted")
    assert observed_modes[-1] == prepared["recommended_mode"]
    assert accepted["auto_decision"]["status"] == "accepted"

    monkeypatch.setattr(gate, "MIN_AUTO_CONFIDENCE", 1.01)
    review_dir = tmp_path / "review-job"
    review = pipeline.run_pipeline(image, Path("source.png"), "auto", review_dir)
    assert observed_modes[-1] == prepared["recommended_mode"]
    assert review["mode_used"] == prepared["recommended_mode"]
    assert review["auto_decision"]["status"] == "needs_review"
    assert review["auto_decision"]["fallback_applied"] is False
    clear_job_auto_decision(review_dir)

    manual = pipeline.run_pipeline(image, Path("source.png"), "centerline", tmp_path / "manual")
    assert observed_modes[-1] == "centerline"
    assert manual["mode_used"] == "centerline"
    assert manual["auto_decision"]["status"] == "manual"
