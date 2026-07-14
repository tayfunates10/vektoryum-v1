from __future__ import annotations

import copy

from analyzer_release_contract import (
    AUTO_MODES,
    CASE_KINDS,
    REPEAT_COUNT,
    SCHEMA_VERSION,
    THRESHOLDS,
    compute_release_metrics,
    validate_release_report,
)


def _sample(mode: str, repeat_index: int) -> dict:
    token = mode.replace("_", "")[:8].ljust(8, "0")
    digest = (token * 8)[:64]
    return {
        "repeat_index": repeat_index,
        "status": "success",
        "contract_status": "valid",
        "source_pixel_sha256": digest,
        "feature_digest": digest[::-1],
        "recommendation_digest": (digest[1:] + digest[:1]),
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
                    "source_sha256": "a" * 64,
                    "deterministic": True,
                    "samples": samples,
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "repeat_count": REPEAT_COUNT,
        "environment": "no_hed",
        "thresholds": THRESHOLDS,
        "metrics": compute_release_metrics(cases),
        "verdict": "release_ready",
        "errors": [],
        "cases": cases,
    }
    return report


def test_complete_release_report_passes() -> None:
    report = _report()
    assert validate_release_report(report) == []
    assert report["metrics"]["accepted_wrong_mode_count"] == 0
    assert report["metrics"]["determinism_failure_count"] == 0
    assert all(value == 1.0 for value in report["metrics"]["accepted_precision_by_mode"].values())


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


def test_invalid_contract_or_hed_status_fails() -> None:
    report = _report()
    report["cases"][0]["samples"][0]["contract_status"] = "invalid"
    report["cases"][0]["samples"][0]["hed_status"] = "measured"
    report["metrics"] = compute_release_metrics(report["cases"])
    report["verdict"] = "failed"
    errors = validate_release_report(report)
    assert any(item.startswith("contract_status:") for item in errors)
    assert any(item.startswith("hed_status:") for item in errors)
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


def test_boundary_review_is_allowed_but_in_domain_review_is_not() -> None:
    boundary = _report()
    case = next(item for item in boundary["cases"] if item["label"] == "lineart" and item["kind"] == "boundary")
    for sample in case["samples"]:
        sample["decision_status"] = "needs_review"
        sample["reason_codes"] = ["margin_below_minimum"]
    boundary["metrics"] = compute_release_metrics(boundary["cases"])
    assert validate_release_report(boundary) == []

    in_domain = copy.deepcopy(boundary)
    case = next(item for item in in_domain["cases"] if item["label"] == "lineart" and item["kind"] == "in_domain")
    for sample in case["samples"]:
        sample["decision_status"] = "needs_review"
        sample["reason_codes"] = ["margin_below_minimum"]
    in_domain["metrics"] = compute_release_metrics(in_domain["cases"])
    in_domain["verdict"] = "failed"
    assert any(item.startswith("in_domain_not_accepted:") for item in validate_release_report(in_domain))


def test_metric_snapshot_must_match_samples() -> None:
    report = _report()
    report["metrics"]["accepted_wrong_mode_count"] = 99
    report["verdict"] = "failed"
    assert "metrics_mismatch" in validate_release_report(report)
