"""Fail-closed CVE-4 all-mode artifact and corpus release contract."""
from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "core-vector-engine-release-v1"
VALIDATION_SCHEMA_VERSION = "core-vector-engine-release-validation-v1"
REPEAT_COUNT = 3
PRODUCTION_MODES = (
    "geometric_logo",
    "minimal_ai",
    "logo_color",
    "flat_logo",
    "single_color",
    "lineart",
    "centerline",
    "photo_poster",
)
IN_DOMAIN_MODES = frozenset(PRODUCTION_MODES) - {"photo_poster"}
REQUIRED_WORKFLOWS = (
    "Exact final SVG contract",
    "Core all-mode release contract",
    "Benchmark v1 seed corpus",
)
ARTIFACT_LIMITS = {
    "ink_recall_min": 0.995,
    "ink_precision_min": 0.975,
    "component_delta": 0,
    "seam_ratio_max": 0.002,
    "halo_ratio_max": 0.02,
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _add(reasons: list[str], code: str) -> None:
    if code not in reasons:
        reasons.append(code)


def _metric_failures(mode: str, sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return [f"{mode}:metrics_missing"]

    required = ("ink_recall", "ink_precision", "component_delta", "seam_ratio", "halo_ratio")
    for name in required:
        if not _finite_number(metrics.get(name)):
            _add(reasons, f"{mode}:{name}_unmeasured")
    if reasons:
        return reasons

    if float(metrics["ink_recall"]) < ARTIFACT_LIMITS["ink_recall_min"]:
        _add(reasons, f"{mode}:ink_recall_below_min")
    if float(metrics["ink_precision"]) < ARTIFACT_LIMITS["ink_precision_min"]:
        _add(reasons, f"{mode}:ink_precision_below_min")
    if int(metrics["component_delta"]) != ARTIFACT_LIMITS["component_delta"]:
        _add(reasons, f"{mode}:component_delta_nonzero")
    if float(metrics["seam_ratio"]) > ARTIFACT_LIMITS["seam_ratio_max"]:
        _add(reasons, f"{mode}:seam_ratio_above_max")
    if float(metrics["halo_ratio"]) > ARTIFACT_LIMITS["halo_ratio_max"]:
        _add(reasons, f"{mode}:halo_ratio_above_max")
    return reasons


def _completed_sample_failures(mode: str, sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    digest = str(sample.get("artifact_sha256") or "")
    evaluator_digest = str(sample.get("evaluator_sha256") or "")
    if len(digest) != 64:
        _add(reasons, f"{mode}:artifact_digest_missing")
    if digest != evaluator_digest:
        _add(reasons, f"{mode}:evaluator_digest_mismatch")
    if sample.get("output_digest_match") is not True:
        _add(reasons, f"{mode}:output_digest_mismatch")
    if sample.get("score_snapshot_match") is not True:
        _add(reasons, f"{mode}:stale_score_snapshot")

    structure = sample.get("structure")
    if not isinstance(structure, dict):
        _add(reasons, f"{mode}:structure_missing")
    else:
        if structure.get("structural_safe") is not True:
            _add(reasons, f"{mode}:structural_unsafe")
        if structure.get("has_bitmap") is not False:
            _add(reasons, f"{mode}:embedded_bitmap")
        if structure.get("nonfinite") is not False:
            _add(reasons, f"{mode}:nonfinite_geometry")
        if structure.get("open_required_cycle") is not False:
            _add(reasons, f"{mode}:open_required_cycle")
        if int(structure.get("path_count") or 0) <= 0:
            _add(reasons, f"{mode}:empty_vector_document")

    verdict = sample.get("verdict")
    if verdict not in {"production_ready", "needs_review"}:
        _add(reasons, f"{mode}:invalid_completed_verdict")
    if mode == "photo_poster":
        if verdict != "needs_review":
            _add(reasons, "photo_poster:false_production_ready")
    else:
        reasons.extend(code for code in _metric_failures(mode, sample) if code not in reasons)
    return reasons


def validate_release_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete three-run report without inferring missing evidence."""
    reasons: list[str] = []
    mode_results: dict[str, dict[str, Any]] = {}

    if payload.get("schema_version") != SCHEMA_VERSION:
        _add(reasons, "unsupported_release_schema")
    if payload.get("repeat_count") != REPEAT_COUNT:
        _add(reasons, "repeat_count_mismatch")

    workflows = payload.get("required_workflows")
    if workflows != list(REQUIRED_WORKFLOWS):
        _add(reasons, "required_workflow_contract_mismatch")

    modes = payload.get("modes")
    if not isinstance(modes, list):
        modes = []
        _add(reasons, "mode_results_missing")

    mode_names = [str(item.get("mode", "")) for item in modes if isinstance(item, dict)]
    if len(mode_names) != len(set(mode_names)):
        _add(reasons, "duplicate_mode_result")
    if set(mode_names) != set(PRODUCTION_MODES):
        _add(reasons, "production_mode_coverage_mismatch")

    for item in modes:
        if not isinstance(item, dict):
            _add(reasons, "invalid_mode_record")
            continue
        mode = str(item.get("mode", ""))
        if mode not in PRODUCTION_MODES:
            continue
        mode_reasons: list[str] = []
        samples = item.get("samples")
        if not isinstance(samples, list) or len(samples) != REPEAT_COUNT:
            _add(mode_reasons, f"{mode}:repeat_sample_count_mismatch")
            samples = samples if isinstance(samples, list) else []
        indices = [sample.get("repeat_index") for sample in samples if isinstance(sample, dict)]
        if indices != list(range(1, REPEAT_COUNT + 1)):
            _add(mode_reasons, f"{mode}:repeat_indices_invalid")

        statuses = [sample.get("status") for sample in samples if isinstance(sample, dict)]
        if any(status not in {"completed", "unavailable", "failed"} for status in statuses):
            _add(mode_reasons, f"{mode}:invalid_sample_status")
        if "failed" in statuses:
            _add(mode_reasons, f"{mode}:sample_failed")

        if statuses and set(statuses) == {"unavailable"}:
            for sample in samples:
                codes = sample.get("reason_codes") if isinstance(sample, dict) else None
                if not isinstance(codes, list) or not codes:
                    _add(mode_reasons, f"{mode}:unavailable_without_reason")
                if sample.get("artifact_sha256") not in {None, ""}:
                    _add(mode_reasons, f"{mode}:unavailable_has_artifact")
            mode_status = "unavailable"
        elif statuses and set(statuses) == {"completed"}:
            digests: set[str] = set()
            verdicts: set[str] = set()
            for sample in samples:
                sample_reasons = _completed_sample_failures(mode, sample)
                for code in sample_reasons:
                    _add(mode_reasons, code)
                digests.add(str(sample.get("artifact_sha256") or ""))
                verdicts.add(str(sample.get("verdict") or ""))
            if len(digests) != 1:
                _add(mode_reasons, f"{mode}:non_deterministic_artifact_digest")
            if mode == "photo_poster":
                mode_codes = item.get("reason_codes") or []
                if "accepted_photo_product_limit" not in mode_codes:
                    _add(mode_reasons, "photo_poster:product_limit_reason_missing")
                mode_status = "needs_review"
            else:
                mode_status = "production_ready" if verdicts == {"production_ready"} else "needs_review"
                if mode_status == "needs_review" and not item.get("reason_codes"):
                    _add(mode_reasons, f"{mode}:needs_review_without_reason")
        else:
            _add(mode_reasons, f"{mode}:mixed_repeat_status")
            mode_status = "failed"

        mode_results[mode] = {
            "status": "fail" if mode_reasons else "pass",
            "release_status": mode_status,
            "reason_codes": mode_reasons,
        }
        for code in mode_reasons:
            _add(reasons, code)

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "pass" if not reasons else "fail",
        "reason_codes": reasons,
        "mode_results": {mode: mode_results.get(mode) for mode in PRODUCTION_MODES},
    }


__all__ = [
    "ARTIFACT_LIMITS",
    "IN_DOMAIN_MODES",
    "PRODUCTION_MODES",
    "REPEAT_COUNT",
    "REQUIRED_WORKFLOWS",
    "SCHEMA_VERSION",
    "validate_release_report",
]
