from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "engine" / "regression" / "rfv3_quality_decision_policy.json"
QUALIFICATION_MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
EVIDENCE_DIR = ROOT / "docs" / "real_world_fidelity" / "evidence"
PIPELINE_RESULTS_PATH = EVIDENCE_DIR / "rfv3_pipeline_results.json"
RETRY_AUDIT_PATH = EVIDENCE_DIR / "rfv3_retry_audit.json"
MEASUREMENT_ENVELOPE_PATH = EVIDENCE_DIR / "rfv3_measurement_envelope.json"
PUBLICATION_ENVELOPE_PATH = EVIDENCE_DIR / "rfv3_publication_envelope.json"
DECISION_PATH = EVIDENCE_DIR / "rfv3_quality_decision.json"
SHA256_LEN = 64


class DecisionError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise DecisionError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"measured_count": 0, "min": None, "median": None, "p95": None, "max": None}
    normalized = [float(value) for value in values]
    return {
        "measured_count": len(normalized),
        "min": min(normalized),
        "median": statistics.median(normalized),
        "p95": _percentile(normalized, 95.0),
        "max": max(normalized),
    }


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema") != "vektoryum-rfv3-quality-decision-policy-v1":
        raise DecisionError("quality decision policy schema drift")
    if policy.get("expected_case_count") != 24 or policy.get("expected_repeat_sample_count") != 72:
        raise DecisionError("finite evidence count contract drift")
    if policy.get("expected_repeat_count") != 3:
        raise DecisionError("repeat count contract drift")
    if policy.get("max_transient_retries_per_repeat") != 1:
        raise DecisionError("retry budget contract drift")
    if policy.get("raw_assets_in_repository") is not False:
        raise DecisionError("raw asset repository boundary drift")
    if policy.get("rfv4_requires_release_go") is not True:
        raise DecisionError("RFV-4 release boundary drift")
    thresholds = policy.get("quality_thresholds")
    if not isinstance(thresholds, dict) or thresholds.get("fidelity_percent_min") != 99.0:
        raise DecisionError("fidelity threshold drift")
    component = thresholds.get("component_score_min")
    if component != {"ssim": 0.98, "edge_f1": 0.98, "alpha_iou": 0.98}:
        raise DecisionError("component threshold drift")
    return policy


def _verify_file_digests(policy: dict[str, Any]) -> None:
    actual = {
        "pipeline_results": sha256_file(PIPELINE_RESULTS_PATH),
        "retry_audit": sha256_file(RETRY_AUDIT_PATH),
        "measurement_envelope": sha256_file(MEASUREMENT_ENVELOPE_PATH),
    }
    if actual != policy["committed_file_sha256"]:
        raise DecisionError("committed RFV-3 evidence digest mismatch")


def _validate_envelopes(
    policy: dict[str, Any],
    measurement: dict[str, Any],
    publication: dict[str, Any],
) -> None:
    if measurement.get("schema") != "vektoryum-rfv3-live-measurement-envelope-v1":
        raise DecisionError("measurement envelope schema mismatch")
    if publication.get("schema") != "vektoryum-rfv3-actions-publication-envelope-v1":
        raise DecisionError("publication envelope schema mismatch")
    expected_common = {
        "case_count": policy["expected_case_count"],
        "cases_sha256": policy["expected_cases_sha256"],
        "pipeline_results_sha256": policy["artifact_file_sha256"]["pipeline_results"],
        "retry_audit_sha256": policy["artifact_file_sha256"]["retry_audit"],
    }
    for key, expected in expected_common.items():
        if measurement.get(key) != expected or publication.get(key) != expected:
            raise DecisionError(f"envelope mismatch: {key}")
    if measurement.get("repeat_sample_count") != policy["expected_repeat_sample_count"]:
        raise DecisionError("measurement repeat count mismatch")
    if publication.get("repeat_sample_count") != policy["expected_repeat_sample_count"]:
        raise DecisionError("publication repeat count mismatch")
    if measurement.get("engine_version") != policy["evidence_engine_version"]:
        raise DecisionError("measurement engine version mismatch")
    if publication.get("head_sha") != policy["evidence_engine_version"]:
        raise DecisionError("publication head SHA mismatch")
    if publication.get("artifact_id") != policy["evidence_artifact_id"]:
        raise DecisionError("publication artifact identity mismatch")
    if publication.get("artifact_digest") != policy["evidence_artifact_digest"]:
        raise DecisionError("publication artifact digest mismatch")
    if measurement.get("raw_assets_in_repository") is not False:
        raise DecisionError("measurement envelope exposes raw assets")
    if publication.get("raw_assets_in_repository") is not False:
        raise DecisionError("publication envelope exposes raw assets")


def _validate_retry_audit(policy: dict[str, Any], retry: dict[str, Any], expected_ids: set[str]) -> dict[str, Any]:
    if retry.get("schema") != "vektoryum-rfv3-live-retry-audit-v1":
        raise DecisionError("retry audit schema mismatch")
    if retry.get("expected_case_count") != 24 or retry.get("completed_case_count") != 24:
        raise DecisionError("retry audit case count mismatch")
    if retry.get("repeat_count") != policy["expected_repeat_count"]:
        raise DecisionError("retry audit repeat count mismatch")
    samples = retry.get("samples")
    if not isinstance(samples, list) or len(samples) != policy["expected_repeat_sample_count"]:
        raise DecisionError("retry audit sample count mismatch")
    seen: set[tuple[str, int]] = set()
    retried_count = 0
    max_attempt_count = 0
    allowed = set(policy["allowed_retry_classes"])
    for sample in samples:
        if not isinstance(sample, dict):
            raise DecisionError("invalid retry sample")
        case_id = sample.get("case_id")
        repeat_index = sample.get("repeat_index")
        key = (case_id, repeat_index)
        if case_id not in expected_ids or repeat_index not in (1, 2, 3) or key in seen:
            raise DecisionError("invalid or duplicate retry sample identity")
        seen.add(key)
        attempt_count = sample.get("attempt_count")
        retried = sample.get("retried")
        attempts = sample.get("attempts")
        if not isinstance(attempt_count, int) or isinstance(attempt_count, bool) or not 1 <= attempt_count <= 2:
            raise DecisionError("retry attempt budget exceeded")
        if not isinstance(attempts, list) or len(attempts) != attempt_count:
            raise DecisionError("retry attempts do not match attempt_count")
        if retried is not (attempt_count == 2):
            raise DecisionError("retry flag mismatch")
        if sample.get("status") != "success":
            raise DecisionError("incomplete retry sample")
        for attempt in attempts:
            if attempt.get("status") not in {"success", "failed"}:
                raise DecisionError("invalid retry attempt status")
            retry_class = attempt.get("retry_class")
            if retry_class is not None and retry_class not in allowed:
                raise DecisionError("non-allowlisted retry class")
        retried_count += int(retried)
        max_attempt_count = max(max_attempt_count, attempt_count)
    if len(seen) != 72:
        raise DecisionError("retry audit identity shrinkage")
    return {
        "status": "passed",
        "sample_count": len(samples),
        "retried_sample_count": retried_count,
        "max_attempt_count": max_attempt_count,
        "allowed_retry_budget": policy["max_transient_retries_per_repeat"],
    }


def evaluate_quality_decision(
    policy: dict[str, Any],
    qualification: dict[str, Any],
    pipeline: dict[str, Any],
    retry: dict[str, Any],
    measurement: dict[str, Any],
    publication: dict[str, Any],
) -> dict[str, Any]:
    policy = validate_policy(policy)
    _validate_envelopes(policy, measurement, publication)
    cases = qualification.get("cases")
    if qualification.get("status") != "qualified" or not isinstance(cases, list) or len(cases) != 24:
        raise DecisionError("qualification corpus is not the exact reviewed 24-case set")
    if qualification.get("cases_sha256") != policy["expected_cases_sha256"]:
        raise DecisionError("qualification case-set digest mismatch")
    expected_ids = {case.get("case_id") for case in cases if isinstance(case, dict)}
    if len(expected_ids) != 24 or None in expected_ids:
        raise DecisionError("invalid qualification case identities")

    if pipeline.get("schema_version") != "benchmark-results-v1":
        raise DecisionError("pipeline result schema mismatch")
    results = pipeline.get("results")
    if pipeline.get("case_count") != 24 or not isinstance(results, list) or len(results) != 24:
        raise DecisionError("pipeline result count mismatch")
    method = pipeline.get("measurement_method")
    if not isinstance(method, dict):
        raise DecisionError("missing measurement method")
    if method.get("cases_sha256") != policy["expected_cases_sha256"] or method.get("repeat_count") != 3:
        raise DecisionError("measurement method identity mismatch")
    if method.get("artifact_sha_policy") != "all_successful_repeats_must_match":
        raise DecisionError("artifact determinism contract mismatch")

    result_ids: set[str] = set()
    quality_rows: list[dict[str, Any]] = []
    thresholds = policy["quality_thresholds"]
    component_thresholds = thresholds["component_score_min"]
    fidelity_threshold = thresholds["fidelity_percent_min"]
    metric_values: dict[str, list[float]] = {name: [] for name in policy["required_quality_metrics"] + policy["observational_metrics"]}
    violation_case_ids: set[str] = set()
    missing_metric_cases: list[dict[str, Any]] = []
    component_violation_counts = {"fidelity": 0, "ssim": 0, "edge_f1": 0, "alpha_iou": 0}

    for result in results:
        if not isinstance(result, dict):
            raise DecisionError("invalid pipeline result")
        case_id = result.get("case_id")
        if case_id not in expected_ids or case_id in result_ids:
            raise DecisionError("unknown or duplicate pipeline case")
        result_ids.add(case_id)
        if result.get("engine_version") != policy["evidence_engine_version"]:
            raise DecisionError("mixed measurement engine versions")
        artifact_sha = result.get("artifact_sha256")
        if not isinstance(artifact_sha, str) or len(artifact_sha) != SHA256_LEN:
            raise DecisionError("invalid artifact SHA-256")
        if result.get("failure") is not None:
            raise DecisionError("pipeline result contains a failure")
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise DecisionError("pipeline metrics missing")
        violations: list[str] = []
        missing: list[str] = []
        fidelity = metrics.get("fidelity")
        if not _finite_number(fidelity):
            missing.append("fidelity")
        else:
            metric_values["fidelity"].append(float(fidelity))
            if float(fidelity) < fidelity_threshold:
                violations.append("fidelity")
                component_violation_counts["fidelity"] += 1
        for metric, threshold in component_thresholds.items():
            value = metrics.get(metric)
            if not _finite_number(value):
                missing.append(metric)
            else:
                metric_values[metric].append(float(value))
                if float(value) < threshold:
                    violations.append(metric)
                    component_violation_counts[metric] += 1
        for metric in policy["observational_metrics"]:
            value = metrics.get(metric)
            if value is not None:
                if not _finite_number(value):
                    raise DecisionError(f"non-finite observational metric: {metric}")
                metric_values[metric].append(float(value))
        if missing:
            missing_metric_cases.append({"case_id": case_id, "metrics": sorted(missing)})
        if missing or violations:
            violation_case_ids.add(case_id)
        quality_rows.append({
            "case_id": case_id,
            "fidelity": fidelity if _finite_number(fidelity) else None,
            "missing_required_metrics": sorted(missing),
            "threshold_violations": sorted(violations),
        })

    if result_ids != expected_ids:
        raise DecisionError("pipeline case-set shrinkage")
    retry_summary = _validate_retry_audit(policy, retry, expected_ids)
    quality_rows.sort(key=lambda row: row["case_id"])
    worst_cases = sorted(
        quality_rows,
        key=lambda row: float("inf") if row["fidelity"] is None else float(row["fidelity"]),
    )[:5]
    quality_passed = not violation_case_ids
    release_decision = "go" if quality_passed else "no_go"
    reasons: list[str] = []
    if component_violation_counts["fidelity"]:
        reasons.append("fidelity_threshold_not_met")
    if any(component_violation_counts[name] for name in ("ssim", "edge_f1", "alpha_iou")):
        reasons.append("component_threshold_not_met")
    if missing_metric_cases:
        reasons.append("missing_required_component_metrics")

    quality_summary = {}
    for metric in policy["required_quality_metrics"]:
        values = metric_values[metric]
        threshold = fidelity_threshold if metric == "fidelity" else component_thresholds[metric]
        quality_summary[metric] = {
            **_summary(values),
            "required_count": 24,
            "missing_count": 24 - len(values),
            "threshold": threshold,
            "pass_count": sum(value >= threshold for value in values),
            "violation_count": component_violation_counts[metric],
        }

    performance_summary = {
        metric: _summary(metric_values[metric])
        for metric in ("render_ms", "peak_rss_mb", "path_count", "svg_bytes", "delta_e00")
    }
    decision = {
        "schema": "vektoryum-rfv3-reviewed-quality-decision-v1",
        "measurement_head_sha": policy["evidence_engine_version"],
        "cases_sha256": policy["expected_cases_sha256"],
        "case_count": 24,
        "repeat_sample_count": 72,
        "completeness_gate": {"status": "passed", "unique_case_count": 24, "failed_case_count": 0},
        "retry_gate": retry_summary,
        "artifact_determinism_gate": {"status": "passed", "policy": method["artifact_sha_policy"]},
        "quality_gate": {
            "status": "passed" if quality_passed else "failed",
            "violation_case_count": len(violation_case_ids),
            "component_violation_counts": component_violation_counts,
            "missing_metric_cases": sorted(missing_metric_cases, key=lambda item: item["case_id"]),
            "metrics": quality_summary,
            "worst_fidelity_cases": worst_cases,
        },
        "performance_observation": performance_summary,
        "release_decision": release_decision,
        "release_block_reasons": reasons,
        "rfv3_phase_decision": "implemented_measurement_gate_release_blocked" if release_decision == "no_go" else "implemented_measurement_gate_release_go",
        "rfv4_allowed": release_decision == "go",
        "raw_assets_in_repository": False,
        "evidence": {
            "artifact_id": policy["evidence_artifact_id"],
            "artifact_digest": policy["evidence_artifact_digest"],
            "pipeline_results_sha256": policy["artifact_file_sha256"]["pipeline_results"],
            "retry_audit_sha256": policy["artifact_file_sha256"]["retry_audit"],
            "measurement_envelope_sha256": policy["artifact_file_sha256"]["measurement_envelope"],
            "committed_pipeline_results_sha256": policy["committed_file_sha256"]["pipeline_results"],
            "committed_retry_audit_sha256": policy["committed_file_sha256"]["retry_audit"],
            "committed_measurement_envelope_sha256": policy["committed_file_sha256"]["measurement_envelope"],
        },
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return decision


def evaluate_committed_evidence() -> dict[str, Any]:
    policy = validate_policy(load_json(POLICY_PATH))
    _verify_file_digests(policy)
    return evaluate_quality_decision(
        policy,
        load_json(QUALIFICATION_MANIFEST_PATH),
        load_json(PIPELINE_RESULTS_PATH),
        load_json(RETRY_AUDIT_PATH),
        load_json(MEASUREMENT_ENVELOPE_PATH),
        load_json(PUBLICATION_ENVELOPE_PATH),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the reviewed RFV-3 live evidence without fabricating quality success.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-committed", action="store_true")
    args = parser.parse_args()
    try:
        decision = evaluate_committed_evidence()
        if args.verify_committed:
            committed = load_json(DECISION_PATH)
            if committed != decision:
                raise DecisionError("committed quality decision is not reproducible")
        if args.output:
            args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except DecisionError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "accepted",
        "quality_gate": decision["quality_gate"]["status"],
        "release_decision": decision["release_decision"],
        "decision_sha256": decision["decision_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
