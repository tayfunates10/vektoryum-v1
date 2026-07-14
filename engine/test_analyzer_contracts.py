from __future__ import annotations

import copy

import numpy as np
from PIL import Image

import app.analyzer as analyzer
from app.analyzer_contracts import (
    AUTO_RECOMMENDATION_MODES,
    CALIBRATION_VERSION,
    CONTRACT_SCHEMA_VERSION,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    SUPPORT_MODEL_VERSION,
    build_analyzer_contract,
    calibration_summary,
    mode_support_scores,
    validate_feature_snapshot,
)


def _geometric_logo() -> Image.Image:
    width, height = 800, 600
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    arr[60:540, 60:90] = 0
    arr[60:540, 710:740] = 0
    arr[60:90, 60:740] = 0
    arr[510:540, 60:740] = 0
    arr[160:360, 180:420] = (255, 0, 0)
    arr[160:440, 470:660] = 0
    return Image.fromarray(arr, "RGB")


def _raw_analysis(monkeypatch) -> tuple[Image.Image, dict]:
    monkeypatch.setattr(analyzer, "calculate_semantic_edge_stats", lambda _image: None)
    image = _geometric_logo()
    report = analyzer.analyze_image_from_mem(image)
    return image, report


def test_feature_schema_is_versioned_machine_readable_and_bounded() -> None:
    assert FEATURE_SCHEMA_VERSION == "analyzer-features-v1"
    assert SUPPORT_MODEL_VERSION == "analyzer-mode-support-v1"
    assert CALIBRATION_VERSION == "analyzer-confidence-calibration-v1"
    assert CONTRACT_SCHEMA_VERSION == "analyzer-decision-contract-v1"
    assert len(FEATURE_SCHEMA) >= 18
    for name, spec in FEATURE_SCHEMA.items():
        assert name
        assert spec["kind"] in {"integer", "number", "boolean"}
        assert isinstance(spec["unit"], str) and spec["unit"]
        if spec["kind"] != "boolean":
            assert spec["minimum"] <= spec["maximum"]


def test_calibration_is_derived_from_committed_labeled_evidence() -> None:
    summary = calibration_summary()
    assert summary["schema_version"] == CALIBRATION_VERSION
    assert summary["method"] == "laplace-smoothed-margin-bin-v1"
    assert summary["evidence_case_count"] == 18
    assert len(summary["evidence_sha256"]) == 64
    assert sum(item["total"] for item in summary["bins"]) == 18
    assert all(0.0 < item["confidence"] < 1.0 for item in summary["bins"])


def test_repeated_decoded_pixels_produce_identical_contract(monkeypatch) -> None:
    image, first = _raw_analysis(monkeypatch)
    second = analyzer.analyze_image_from_mem(image.copy())

    assert first["recommended_mode"] == "geometric_logo"
    assert second["recommended_mode"] == first["recommended_mode"]
    assert first["analyzer_contract"] == second["analyzer_contract"]
    contract = first["analyzer_contract"]
    assert contract["status"] == "valid"
    assert contract["optional_signals"] == {"hed": "unavailable"}
    assert 0.0 <= contract["confidence"] <= 0.85
    assert -1.0 <= contract["runner_up_margin"] <= 1.0
    assert contract["runner_up_mode"] in AUTO_RECOMMENDATION_MODES
    assert len(contract["source_pixel_sha256"]) == 64
    assert len(contract["feature_digest"]) == 64
    assert len(contract["recommendation_digest"]) == 64
    assert first["recommendation_confidence"] == contract["confidence"]
    assert first["recommendation_margin"] == contract["runner_up_margin"]
    assert first["recommendation_digest"] == contract["recommendation_digest"]


def test_pixel_or_recommendation_change_changes_digest(monkeypatch) -> None:
    image, report = _raw_analysis(monkeypatch)
    original = report["analyzer_contract"]

    changed = image.copy()
    changed.putpixel((0, 0), (254, 255, 255))
    changed_report = analyzer.analyze_image_from_mem(changed)
    assert changed_report["analyzer_contract"]["source_pixel_sha256"] != original["source_pixel_sha256"]
    assert changed_report["analyzer_contract"]["recommendation_digest"] != original["recommendation_digest"]

    forged = copy.deepcopy(report)
    forged.pop("analyzer_contract", None)
    forged["recommended_mode"] = "logo_color"
    forged["detected_type"] = "logo_color"
    forged_contract = build_analyzer_contract(forged, image)
    assert forged_contract["recommendation_digest"] != original["recommendation_digest"]


def test_missing_nonfinite_and_partial_optional_features_fail_closed(monkeypatch) -> None:
    image, report = _raw_analysis(monkeypatch)
    base = {key: value for key, value in report.items() if key not in {
        "analyzer_contract",
        "recommendation_confidence",
        "recommendation_margin",
        "recommendation_digest",
    }}

    missing = copy.deepcopy(base)
    missing.pop("edge_density")
    missing_contract = build_analyzer_contract(missing, image)
    assert missing_contract["status"] == "invalid"
    assert missing_contract["confidence"] is None
    assert missing_contract["recommendation_digest"] is None
    assert "missing_feature:edge_density" in missing_contract["errors"]

    nonfinite = copy.deepcopy(base)
    nonfinite["quality_score"] = float("nan")
    nonfinite_contract = build_analyzer_contract(nonfinite, image)
    assert nonfinite_contract["status"] == "invalid"
    assert nonfinite_contract["confidence"] is None
    assert "nonfinite_or_invalid:quality_score" in nonfinite_contract["errors"]

    partial = copy.deepcopy(base)
    partial["semantic_edge_density"] = 0.1
    partial["edge_coherence"] = None
    partial_contract = build_analyzer_contract(partial, image)
    assert partial_contract["status"] == "invalid"
    assert partial_contract["optional_signals"] == {"hed": "invalid"}
    assert "partial_optional_signal:hed" in partial_contract["errors"]


def test_support_scores_and_snapshot_validation_are_bounded(monkeypatch) -> None:
    _image, report = _raw_analysis(monkeypatch)
    snapshot, errors, signals = validate_feature_snapshot(report)
    assert not errors
    assert snapshot is not None
    assert signals == {"hed": "unavailable"}
    scores = mode_support_scores(snapshot)
    assert set(scores) == set(AUTO_RECOMMENDATION_MODES)
    assert all(0.0 <= score <= 1.0 for score in scores.values())
