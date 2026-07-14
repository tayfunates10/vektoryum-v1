"""Fail-closed release contract for the labeled AI Analyzer corpus."""
from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "analyzer-release-report-v1"
REPEAT_COUNT = 3
AUTO_MODES = (
    "geometric_logo",
    "minimal_ai",
    "logo_color",
    "single_color",
    "lineart",
    "photo_poster",
)
CASE_KINDS = ("in_domain", "boundary")
REQUIRED_ENVIRONMENTS = ("no_hed",)

THRESHOLDS = {
    "accepted_wrong_mode_max": 0,
    "determinism_failures_max": 0,
    "invalid_contracts_max": 0,
    "per_mode_accepted_precision_min": 1.0,
    "per_mode_correct_accepts_min": 1,
    "brier_score_max": 0.35,
    "expected_calibration_error_max": 0.35,
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_release_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if report.get("repeat_count") != REPEAT_COUNT:
        errors.append("repeat_count")
    if report.get("thresholds") != THRESHOLDS:
        errors.append("thresholds")

    cases = report.get("cases")
    if not isinstance(cases, list):
        return errors + ["cases_missing"]

    expected_pairs = {(mode, kind) for mode in AUTO_MODES for kind in CASE_KINDS}
    observed_pairs: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    accepted_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_accepts_by_label = {mode: 0 for mode in AUTO_MODES}
    accepted_wrong = 0
    invalid_contracts = 0
    determinism_failures = 0
    calibration_rows: list[tuple[float, float]] = []

    for case in cases:
        if not isinstance(case, dict):
            errors.append("case_invalid")
            continue
        case_id = str(case.get("case_id") or "")
        label = case.get("label")
        kind = case.get("kind")
        environment = case.get("environment")
        samples = case.get("samples")
        if not case_id or case_id in case_ids:
            errors.append("case_id")
        case_ids.add(case_id)
        if label not in AUTO_MODES or kind not in CASE_KINDS:
            errors.append(f"case_scope:{case_id}")
            continue
        if environment not in REQUIRED_ENVIRONMENTS:
            errors.append(f"environment:{case_id}")
        observed_pairs.add((label, kind))
        if not isinstance(samples, list) or len(samples) != REPEAT_COUNT:
            errors.append(f"sample_count:{case_id}")
            continue

        signatures: set[tuple[Any, ...]] = set()
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("status") != "success":
                errors.append(f"sample_status:{case_id}")
                continue
            signature = (
                sample.get("source_pixel_sha256"),
                sample.get("feature_digest"),
                sample.get("recommendation_digest"),
                sample.get("recommended_mode"),
                sample.get("decision_status"),
                sample.get("execution_mode"),
                sample.get("confidence"),
                sample.get("runner_up_margin"),
                tuple(sample.get("reason_codes") or []),
                sample.get("hed_status"),
            )
            signatures.add(signature)
            if sample.get("contract_status") != "valid":
                invalid_contracts += 1
            if sample.get("hed_status") != "unavailable":
                errors.append(f"hed_status:{case_id}")

        deterministic = len(signatures) == 1
        if bool(case.get("deterministic")) != deterministic or not deterministic:
            determinism_failures += 1
        representative = samples[0]
        prediction = representative.get("recommended_mode")
        decision_status = representative.get("decision_status")
        confidence = _finite(representative.get("confidence"))
        if prediction not in AUTO_MODES:
            errors.append(f"prediction:{case_id}")
            continue
        accepted = decision_status == "accepted"
        if decision_status not in {"accepted", "needs_review"}:
            errors.append(f"decision_status:{case_id}")
        correct = prediction == label
        if accepted:
            accepted_by_prediction[prediction] += 1
            if correct:
                correct_by_prediction[prediction] += 1
                correct_accepts_by_label[label] += 1
            else:
                accepted_wrong += 1
        if confidence is None or not 0.0 <= confidence <= 1.0:
            errors.append(f"confidence:{case_id}")
        else:
            calibration_rows.append((confidence, 1.0 if correct else 0.0))
        if kind == "in_domain" and not (accepted and correct):
            errors.append(f"in_domain_not_accepted:{case_id}")
        if kind == "boundary" and not (correct or decision_status == "needs_review"):
            errors.append(f"boundary_false_accept:{case_id}")

    if observed_pairs != expected_pairs:
        errors.append("mode_kind_coverage")

    precision: dict[str, float | None] = {}
    for mode in AUTO_MODES:
        denominator = accepted_by_prediction[mode]
        precision[mode] = None if denominator == 0 else correct_by_prediction[mode] / denominator
        if correct_accepts_by_label[mode] < THRESHOLDS["per_mode_correct_accepts_min"]:
            errors.append(f"correct_accept_coverage:{mode}")
        if precision[mode] is None or precision[mode] < THRESHOLDS["per_mode_accepted_precision_min"]:
            errors.append(f"accepted_precision:{mode}")

    brier = None
    ece = None
    if calibration_rows:
        brier = sum((confidence - outcome) ** 2 for confidence, outcome in calibration_rows) / len(calibration_rows)
        bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        weighted = 0.0
        for lower, upper in bins:
            rows = [
                row for row in calibration_rows
                if lower <= row[0] <= upper if lower == 0.0 else lower < row[0] <= upper
            ]
            if rows:
                avg_conf = sum(row[0] for row in rows) / len(rows)
                avg_acc = sum(row[1] for row in rows) / len(rows)
                weighted += len(rows) / len(calibration_rows) * abs(avg_conf - avg_acc)
        ece = weighted
    if brier is None or brier > THRESHOLDS["brier_score_max"]:
        errors.append("brier_score")
    if ece is None or ece > THRESHOLDS["expected_calibration_error_max"]:
        errors.append("expected_calibration_error")
    if accepted_wrong > THRESHOLDS["accepted_wrong_mode_max"]:
        errors.append("accepted_wrong_mode")
    if invalid_contracts > THRESHOLDS["invalid_contracts_max"]:
        errors.append("invalid_contracts")
    if determinism_failures > THRESHOLDS["determinism_failures_max"]:
        errors.append("determinism_failures")

    metrics = report.get("metrics")
    expected_metrics = {
        "accepted_wrong_mode_count": accepted_wrong,
        "invalid_contract_count": invalid_contracts,
        "determinism_failure_count": determinism_failures,
        "accepted_precision_by_mode": precision,
        "correct_accepts_by_label": correct_accepts_by_label,
        "brier_score": None if brier is None else round(brier, 6),
        "expected_calibration_error": None if ece is None else round(ece, 6),
    }
    if metrics != expected_metrics:
        errors.append("metrics_mismatch")
    expected_verdict = "release_ready" if not errors else "failed"
    if report.get("verdict") != expected_verdict:
        errors.append("verdict")
    return sorted(set(errors))


__all__ = [
    "AUTO_MODES",
    "CASE_KINDS",
    "REPEAT_COUNT",
    "REQUIRED_ENVIRONMENTS",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "validate_release_report",
]
