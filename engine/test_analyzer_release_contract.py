from __future__ import annotations

import copy
import hashlib

from analyzer_release_contract import (
    AUTO_MODES,
    CASE_KINDS,
    REPEAT_COUNT,
    SCHEMA_VERSION,
    THRESHOLDS,
    compute_release_metrics,
    validate_release_report,
)


def _digest(mode: str, name: str) -> str:
    return hashlib.sha256(f"{mode}:{name}".encode("utf-8")).hexdigest()


def _sample(mode: str, repeat_index: int) -> dict:
    return {
        "repeat_index": repeat_index,
        "status": "success",
        "contract_status": "valid",
        "source_pixel_sha256": _digest(mode, "source"),
        "feature_digest": _digest(mode, "feature"),
        "recommendation_digest": _digest(mode, "recommendation"),
        "recommended_mode": mode,
        "decision_status": "accepted",
        "execution_mode": mode,
        "fallback_applied": False,
        "confidence": 0.8,
        "runner_up_mode": "logo_color" if mode != "logo_color" else "minimal_ai",
        "runner_up_margin": 0.2,
        "reason_codes": ["verified_recommendation"],
        "hed_status": "unavailable",
    }


def _report() -> dict:
    cases = []
    for mode in AUTO_MODES:
        for kind in CASE_KINDS:
            samples = [_sample(mode, index) for index in range(1, REPEAT_COUNT + 1)]
            cases.append(
                {
                    "case_id": f"{mode}-{kind}",
                    "label": mode,
                    "kind": kind,
                    "environment": "no_hed",
                    "source_sha256": _digest(mode, kind),
                    "deterministic": True,
                    "samples": samples,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "repeat_count": REPEAT_COUNT,
        "environment": "no_hed",
        "thresholds": THRESHOLDS,
        "metrics": compute_release_metrics(cases),
        "verdict": "release_ready",
        "errors": [],
        "cases": cases,
    }


def test_complete_release_report_passes() -> None:
    report = _report()
    assert validate_release_report(report) == []
    metrics = report["metrics"]
    assert metrics["accepted_wrong_mode_count"] == 0
    assert metrics["determinism_failure_count"] == 0
    assert all(value == 1.0 for value in metrics["recommendation_precision_by_mode"].values())
    assert all(value == 1.0 for value in metrics["recommendation_recall_by_label"].values())
    assert all(value == 1.0 for value in metrics["accepted_precision_by_mode"].values())


def test_missing_mode_kind_and_repeat_count_fail() -> None:
    missing = _report()
    missing["cases"].pop()
    missing["metrics"] = compute_release_metrics(missing["cases"])
    missing["verdict"] = "failed"
    assert "mode_kind_coverage" in validate_release_report(missing)

    repeats = _report()
    repeats["cases"][0]["samples"].pop()
    repeats["metrics"] = compute_release_metrics(repeats["cases"])
    repeats["verdict"] = "failed"
    assert any(item.startswith("sample_count:") for item in validate_release_report(repeats))


def test_repeat_digest_difference_fails() -> None:
    report = _report()
    report["cases"][0]["samples"][2]["feature_digest"] = "f" * 64
    report["cases"][0]["deterministic"] = False
    report["metrics"] = compute_release_metrics(report["cases"])
    report["verdict"] = "failed"
    errors = validate_release_report(report)
    assert any(item.startswith("determinism:") for item in errors)
    assert "determinism_failures" in errors


def test_invalid_contract_hed_or_digest_fails() -> None:
    report = _report()
    sample = report["cases"][0]["samples"][0]
    sample["contract_status"] = "invalid"
    sample["hed_status"] = "measured"
    sample["source_pixel_sha256"] = "not-a-digest"
    report["metrics"] = compute_release_metrics(report["cases"])
    report["verdict"] = "failed"
    errors = validate_release_report(report)
    assert any(item.startswith("contract_status:") for item in errors)
    assert any(item.startswith("hed_status:") for item in errors)
    assert any(item.startswith("digest:") for item in errors)
    assert "invalid_contracts" in errors


def test_accepted_wrong_mode_fails() -> None:
    report = _report()
    case = next(item for item in report["cases"] if item["label"] == "lineart" and item["kind"] == "boundary")
    for sample in case["samples"]:
        sample["recommended_mode"] = "single_color"
        sample["execution_mode"] = "single_color"
    report["metrics"] = compute_release_metrics(report["cases"])
    report["verdict"] = "failed"
    errors = validate_release_report(report)
    assert "accepted_wrong_mode" in errors
    assert any(item.startswith("boundary_false_accept:") for item in errors)


def test_correct_review_is_allowed_but_in_domain_prediction_mismatch_is_not() -> None:
    reviewed = _report()
    case = next(item for item in reviewed["cases"] if item["label"] == "lineart" and item["kind"] == "in_domain")
    for sample in case["samples"]:
        sample["decision_status"] = "needs_review"
        sample["confidence"] = 0.25
        sample["reason_codes"] = ["margin_below_minimum"]
    reviewed["metrics"] = compute_release_metrics(reviewed["cases"])
    assert validate_release_report(reviewed) == []
    assert reviewed["metrics"]["review_count"] == 1
    assert reviewed["metrics"]["accepted_precision_by_mode"]["lineart"] == 1.0

    mismatched = copy.deepcopy(reviewed)
    case = next(item for item in mismatched["cases"] if item["label"] == "minimal_ai" and item["kind"] == "in_domain")
    for sample in case["samples"]:
        sample["recommended_mode"] = "logo_color"
        sample["execution_mode"] = "logo_color"
        sample["decision_status"] = "needs_review"
        sample["confidence"] = 0.25
        sample["reason_codes"] = ["support_contradiction"]
    mismatched["metrics"] = compute_release_metrics(mismatched["cases"])
    mismatched["verdict"] = "failed"
    errors = validate_release_report(mismatched)
    assert any(item.startswith("in_domain_prediction_mismatch:") for item in errors)
    assert "correct_recommendation_coverage:minimal_ai" not in errors


def test_review_aware_calibration_uses_safe_acceptance_outcome() -> None:
    report = _report()
    case = next(item for item in report["cases"] if item["label"] == "photo_poster" and item["kind"] == "boundary")
    for sample in case["samples"]:
        sample["decision_status"] = "needs_review"
        sample["confidence"] = 0.25
        sample["reason_codes"] = ["confidence_below_minimum"]
    report["metrics"] = compute_release_metrics(report["cases"])
    assert validate_release_report(report) == []
    assert report["metrics"]["brier_score"] <= THRESHOLDS["brier_score_max"]
    assert report["metrics"]["expected_calibration_error"] <= THRESHOLDS["expected_calibration_error_max"]


def test_metric_snapshot_must_match_samples() -> None:
    report = _report()
    report["metrics"]["accepted_wrong_mode_count"] = 99
    report["verdict"] = "failed"
    assert "metrics_mismatch" in validate_release_report(report)
