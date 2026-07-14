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
    "per_mode_recommendation_precision_min": 1.0,
    "per_mode_correct_recommendations_min": 1,
    "accepted_precision_min": 1.0,
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


def _in_bin(confidence: float, lower: float, upper: float) -> bool:
    return lower <= confidence <= upper if lower == 0.0 else lower < confidence <= upper


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def compute_release_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    recommended_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_recommended_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_recommendations_by_label = {mode: 0 for mode in AUTO_MODES}
    accepted_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_accepted_by_prediction = {mode: 0 for mode in AUTO_MODES}
    correct_accepts_by_label = {mode: 0 for mode in AUTO_MODES}
    confusion = {
        label: {prediction: 0 for prediction in AUTO_MODES}
        for label in AUTO_MODES
    }
    accepted_wrong = 0
    invalid_contracts = 0
    determinism_failures = 0
    accepted_count = 0
    accepted_correct_count = 0
    review_count = 0
    calibration_rows: list[tuple[float, float]] = []

    for case in cases:
        samples = case.get("samples") or []
        if not samples:
            determinism_failures += 1
            continue
        signatures = {
            (
                sample.get("source_pixel_sha256"),
                sample.get("feature_digest"),
                sample.get("recommendation_digest"),
                sample.get("recommended_mode"),
                sample.get("decision_status"),
                sample.get("execution_mode"),
                sample.get("fallback_applied"),
                sample.get("confidence"),
                sample.get("runner_up_margin"),
                tuple(sample.get("reason_codes") or []),
                sample.get("hed_status"),
            )
            for sample in samples
        }
        if len(signatures) != 1:
            determinism_failures += 1
        invalid_contracts += sum(
            sample.get("contract_status") != "valid" for sample in samples
        )

        representative = samples[0]
        label = case.get("label")
        prediction = representative.get("recommended_mode")
        accepted = representative.get("decision_status") == "accepted"
        correct = prediction == label

        if label in confusion and prediction in AUTO_MODES:
            confusion[label][prediction] += 1
            recommended_by_prediction[prediction] += 1
            if correct:
                correct_recommended_by_prediction[prediction] += 1
                correct_recommendations_by_label[label] += 1

        if accepted:
            accepted_count += 1
            if prediction in accepted_by_prediction:
                accepted_by_prediction[prediction] += 1
            if correct:
                accepted_correct_count += 1
                if prediction in correct_accepted_by_prediction:
                    correct_accepted_by_prediction[prediction] += 1
                if label in correct_accepts_by_label:
                    correct_accepts_by_label[label] += 1
            else:
                accepted_wrong += 1
        else:
            review_count += 1

        confidence = _finite(representative.get("confidence"))
        if confidence is not None and 0.0 <= confidence <= 1.0:
            # AA-3 confidence governs safe auto-acceptance. A correct recommendation
            # that is intentionally routed to review is therefore a negative outcome
            # for acceptance calibration, not a classification failure.
            calibration_rows.append((confidence, 1.0 if accepted and correct else 0.0))

    recommendation_precision: dict[str, float | None] = {}
    accepted_precision: dict[str, float | None] = {}
    recommendation_recall: dict[str, float] = {}
    label_totals = {mode: 0 for mode in AUTO_MODES}
    for case in cases:
        label = case.get("label")
        if label in label_totals:
            label_totals[label] += 1

    for mode in AUTO_MODES:
        recommendation_total = recommended_by_prediction[mode]
        recommendation_precision[mode] = (
            None
            if recommendation_total == 0
            else correct_recommended_by_prediction[mode] / recommendation_total
        )
        accepted_total = accepted_by_prediction[mode]
        accepted_precision[mode] = (
            None
            if accepted_total == 0
            else correct_accepted_by_prediction[mode] / accepted_total
        )
        recommendation_recall[mode] = (
            0.0
            if label_totals[mode] == 0
            else correct_recommendations_by_label[mode] / label_totals[mode]
        )

    brier = None
    ece = None
    if calibration_rows:
        brier = sum((confidence - outcome) ** 2 for confidence, outcome in calibration_rows) / len(calibration_rows)
        weighted = 0.0
        for lower, upper in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)):
            rows = [row for row in calibration_rows if _in_bin(row[0], lower, upper)]
            if rows:
                average_confidence = sum(row[0] for row in rows) / len(rows)
                average_accuracy = sum(row[1] for row in rows) / len(rows)
                weighted += len(rows) / len(calibration_rows) * abs(
                    average_confidence - average_accuracy
                )
        ece = weighted

    case_count = len(cases)
    return {
        "confusion_matrix": confusion,
        "recommendation_precision_by_mode": recommendation_precision,
        "recommendation_recall_by_label": recommendation_recall,
        "accepted_precision_by_mode": accepted_precision,
        "correct_recommendations_by_label": correct_recommendations_by_label,
        "correct_accepts_by_label": correct_accepts_by_label,
        "accepted_wrong_mode_count": accepted_wrong,
        "invalid_contract_count": invalid_contracts,
        "determinism_failure_count": determinism_failures,
        "accepted_count": accepted_count,
        "accepted_correct_count": accepted_correct_count,
        "review_count": review_count,
        "review_rate": 0.0 if case_count == 0 else round(review_count / case_count, 6),
        "brier_score": None if brier is None else round(brier, 6),
        "expected_calibration_error": None if ece is None else round(ece, 6),
    }


def validate_release_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if report.get("repeat_count") != REPEAT_COUNT:
        errors.append("repeat_count")
    if report.get("environment") not in REQUIRED_ENVIRONMENTS:
        errors.append("report_environment")
    if report.get("thresholds") != THRESHOLDS:
        errors.append("thresholds")

    cases = report.get("cases")
    if not isinstance(cases, list):
        return errors + ["cases_missing"]

    expected_pairs = {(mode, kind) for mode in AUTO_MODES for kind in CASE_KINDS}
    observed_pairs: set[tuple[str, str]] = set()
    case_ids: set[str] = set()

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
        if not _valid_digest(case.get("source_sha256")):
            errors.append(f"source_sha256:{case_id}")
        if label not in AUTO_MODES or kind not in CASE_KINDS:
            errors.append(f"case_scope:{case_id}")
            continue
        observed_pairs.add((label, kind))
        if environment not in REQUIRED_ENVIRONMENTS:
            errors.append(f"environment:{case_id}")
        if not isinstance(samples, list) or len(samples) != REPEAT_COUNT:
            errors.append(f"sample_count:{case_id}")
            continue

        signatures: set[tuple[Any, ...]] = set()
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("status") != "success":
                errors.append(f"sample_status:{case_id}")
                continue
            signatures.add(
                (
                    sample.get("source_pixel_sha256"),
                    sample.get("feature_digest"),
                    sample.get("recommendation_digest"),
                    sample.get("recommended_mode"),
                    sample.get("decision_status"),
                    sample.get("execution_mode"),
                    sample.get("fallback_applied"),
                    sample.get("confidence"),
                    sample.get("runner_up_margin"),
                    tuple(sample.get("reason_codes") or []),
                    sample.get("hed_status"),
                )
            )
            for name in (
                "source_pixel_sha256",
                "feature_digest",
                "recommendation_digest",
            ):
                if not _valid_digest(sample.get(name)):
                    errors.append(f"digest:{case_id}:{name}")
            if sample.get("contract_status") != "valid":
                errors.append(f"contract_status:{case_id}")
            if sample.get("hed_status") != "unavailable":
                errors.append(f"hed_status:{case_id}")
            confidence = _finite(sample.get("confidence"))
            if confidence is None or not 0.0 <= confidence <= 1.0:
                errors.append(f"confidence:{case_id}")
            if sample.get("runner_up_mode") not in AUTO_MODES:
                errors.append(f"runner_up_mode:{case_id}")
            if not isinstance(sample.get("reason_codes"), list):
                errors.append(f"reason_codes:{case_id}")
            if not isinstance(sample.get("fallback_applied"), bool):
                errors.append(f"fallback_applied:{case_id}")
        deterministic = len(signatures) == 1
        if bool(case.get("deterministic")) != deterministic or not deterministic:
            errors.append(f"determinism:{case_id}")

        representative = samples[0]
        prediction = representative.get("recommended_mode")
        decision_status = representative.get("decision_status")
        correct = prediction == label
        if prediction not in AUTO_MODES:
            errors.append(f"prediction:{case_id}")
        if decision_status not in {"accepted", "needs_review"}:
            errors.append(f"decision_status:{case_id}")
        if kind == "in_domain" and not correct:
            errors.append(f"in_domain_prediction_mismatch:{case_id}")
        if kind == "boundary" and not (correct or decision_status == "needs_review"):
            errors.append(f"boundary_false_accept:{case_id}")

    if observed_pairs != expected_pairs:
        errors.append("mode_kind_coverage")

    metrics = compute_release_metrics(cases)
    if report.get("metrics") != metrics:
        errors.append("metrics_mismatch")
    if metrics["accepted_wrong_mode_count"] > THRESHOLDS["accepted_wrong_mode_max"]:
        errors.append("accepted_wrong_mode")
    if metrics["invalid_contract_count"] > THRESHOLDS["invalid_contracts_max"]:
        errors.append("invalid_contracts")
    if metrics["determinism_failure_count"] > THRESHOLDS["determinism_failures_max"]:
        errors.append("determinism_failures")
    for mode in AUTO_MODES:
        if metrics["correct_recommendations_by_label"][mode] < THRESHOLDS["per_mode_correct_recommendations_min"]:
            errors.append(f"correct_recommendation_coverage:{mode}")
        precision = metrics["recommendation_precision_by_mode"][mode]
        if precision is None or precision < THRESHOLDS["per_mode_recommendation_precision_min"]:
            errors.append(f"recommendation_precision:{mode}")
        accepted_precision = metrics["accepted_precision_by_mode"][mode]
        if accepted_precision is not None and accepted_precision < THRESHOLDS["accepted_precision_min"]:
            errors.append(f"accepted_precision:{mode}")
    brier = metrics["brier_score"]
    ece = metrics["expected_calibration_error"]
    if brier is None or brier > THRESHOLDS["brier_score_max"]:
        errors.append("brier_score")
    if ece is None or ece > THRESHOLDS["expected_calibration_error_max"]:
        errors.append("expected_calibration_error")

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
    "compute_release_metrics",
    "validate_release_report",
]
