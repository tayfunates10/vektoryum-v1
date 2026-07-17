"""Sanitized RFV-3E diagnostics for the exact-metric fallback cluster.

The live PR #100 artifact proves that the selected SVG path and file existed and
that the exact evaluator was attempted for cases -10, -14 and -18.  It also
proves an ``exact_metrics_incomplete`` result and an explicit partial-report
fallback.  It does not contain repeat-level evaluator reports or hard-failure
codes, so the production cause remains unresolved and no production fix is
allowed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = ROOT / "docs/real_world_fidelity/evidence/rfv3e_exact_metric_path_fallback_diagnostics.json"
DEFAULT_PLAN = ROOT / "docs/real_world_fidelity/evidence/rfv3e_remediation_plan.json"
DEFAULT_DECISION = ROOT / "docs/real_world_fidelity/evidence/rfv3_quality_decision.json"
DEFAULT_ROADMAP = ROOT / "docs/real_world_fidelity_roadmap.json"

SCHEMA = "vektoryum-rfv3e-exact-metric-path-fallback-diagnostics-v1"
PROVENANCE_SCHEMA = "rfv3d2-metric-provenance-v1"
EXPECTED_CASES = (
    "qualification-public-10",
    "qualification-public-14",
    "qualification-public-18",
)
EXPECTED_GAPS = ("delta_e00", "edge_f1", "ssim")
EXPECTED_THRESHOLDS = {"alpha_iou": 0.98, "edge_f1": 0.98, "fidelity": 99.0, "ssim": 0.98}
EXPECTED_SOURCE = {
    "artifact_digest": "sha256:d747ce3d8b1eb5e403bea39ba2607a7b75d3cb8cff2f77f26a1b528d7a7dd037",
    "artifact_id": 8399777964,
    "artifact_name": "rfv3-live-measurement-0c09bcbad152c2661673e214dd159183e50a6525",
    "head_sha": "0c09bcbad152c2661673e214dd159183e50a6525",
    "measurement_envelope_file_sha256": "abaa39ec421a980b711c8b54b49b4768a28a55a966e4ea05d71480b6a051a988",
    "pipeline_results_file_sha256": "6e6c8f1458a1725153f7187fb3eefeef645b9565c56df4aed9d4c6fc35b7631a",
    "pull_request": 100,
    "repository": "tayfunates10/vektoryum-v1",
    "retry_audit_file_sha256": "c7eec85ced0f5504ec0bf52aa29961c943df436bd922814ba3f9f328cc6c347f",
    "workflow_name": "Real-world fidelity RFV-3B live production measurement",
    "workflow_run_attempt": 1,
    "workflow_run_id": 29555984755,
}
EXPECTED_BINDINGS = {
    "canonical_decision_file_sha256": "4fb585cc99c566c6a340ad8a2a982aaea4cea68077a197acfd20ea1db814ace6",
    "canonical_decision_sha256": "39f2c1a7ac81f2a0f338c0099c0c82de53f295fb4b56dc5b6ed3179004c6148c",
    "canonical_results_sha256": "663697ba9c021b49d42b5e8941c2d81794d3fb60c96b2273c61fe838b23f3e0a",
    "corpus_or_case_set_sha256": "5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89",
    "remediation_plan_git_blob_sha": "822db26d6ef2012f1065aecba689106c58a3ae44",
}
EXPECTED_ARTIFACT_SHAS = {
    "qualification-public-10": "c617bbf0533b4ddcf75106fd2a9500a49e6b68f6b0fd363ffee47fd61b6e1e74",
    "qualification-public-14": "f2ace52eab2ed5c75a53bbb3033900714930e07f61ce1dd3be3717a66346220a",
    "qualification-public-18": "621759af9e376c3d4cfbe38d2bc7b61ce1c0023730cf9e52aa6aa40479a3e43a",
}

_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:home|tmp|var|Users|private|opt|workspace|runner)(?:/[^\s\"']+)+")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/](?:[^\s\"']+[\\/])*[^\s\"']+")
_SECRET = re.compile(r"(?i)(authorization\s*:|bearer\s+[A-Za-z0-9._~+/-]{8,}|github_pat_|ghp_[A-Za-z0-9]{10,}|token[=:\s]+[A-Za-z0-9._~+/-]{8,})")
_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")


class DiagnosticsError(RuntimeError):
    """Fail-closed diagnostics contract error."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticsError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticsError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DiagnosticsError(f"unreadable evidence file: {path}") from exc


def git_blob_sha(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DiagnosticsError(f"unreadable binding file: {path}") from exc
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def reject_sensitive(value: Any) -> None:
    for text in _strings(value):
        if _POSIX_PATH.search(text):
            raise DiagnosticsError("sanitized evidence contains a POSIX absolute path")
        if _WINDOWS_PATH.search(text):
            raise DiagnosticsError("sanitized evidence contains a Windows absolute path")
        if _SECRET.search(text):
            raise DiagnosticsError("sanitized evidence contains a secret-like token")
        if _TRACEBACK.search(text):
            raise DiagnosticsError("sanitized evidence contains a full traceback")


def _index(rows: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise DiagnosticsError(f"{label} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise DiagnosticsError(f"invalid {label} row")
        identity = row[key]
        if identity in result:
            raise DiagnosticsError(f"duplicate {label}: {identity}")
        result[identity] = row
    return result


def _repeat_rows(samples: Any, case_id: str) -> list[dict[str, Any]]:
    rows = [row for row in samples if isinstance(row, dict) and row.get("case_id") == case_id]
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("repeat_index")
        if not isinstance(index, int) or index in by_index:
            raise DiagnosticsError(f"invalid or duplicate repeat for {case_id}")
        by_index[index] = row
    if sorted(by_index) != [1, 2, 3]:
        raise DiagnosticsError(f"repeat coverage drift for {case_id}")
    return [
        {
            "attempt_count": by_index[index].get("attempt_count"),
            "repeat_index": index,
            "retried": by_index[index].get("retried"),
            "status": by_index[index].get("status"),
        }
        for index in (1, 2, 3)
    ]


def build_evidence(
    pipeline_results: Path,
    retry_audit: Path,
    measurement_envelope: Path,
    publication_envelope: Path,
) -> dict[str, Any]:
    digests = {
        "measurement_envelope_file_sha256": sha256_file(measurement_envelope),
        "pipeline_results_file_sha256": sha256_file(pipeline_results),
        "retry_audit_file_sha256": sha256_file(retry_audit),
    }
    if any(digests[name] != EXPECTED_SOURCE[name] for name in digests):
        raise DiagnosticsError("source artifact file digest drift")

    pipeline = load_json(pipeline_results)
    retry = load_json(retry_audit)
    measurement = load_json(measurement_envelope)
    publication = load_json(publication_envelope)
    if pipeline.get("schema_version") != "benchmark-results-v1" or pipeline.get("case_count") != 24:
        raise DiagnosticsError("unexpected pipeline result contract")
    if retry.get("schema") != "vektoryum-rfv3-live-retry-audit-v1" or retry.get("sample_count") != 72:
        raise DiagnosticsError("unexpected retry audit contract")
    if measurement.get("engine_version") != EXPECTED_SOURCE["head_sha"]:
        raise DiagnosticsError("measurement head SHA drift")
    if measurement.get("cases_sha256") != EXPECTED_BINDINGS["corpus_or_case_set_sha256"]:
        raise DiagnosticsError("measurement corpus identity drift")
    publication_digest = str(publication.get("artifact_digest") or "")
    if publication_digest and not publication_digest.startswith("sha256:"):
        publication_digest = f"sha256:{publication_digest}"
    if publication.get("artifact_id") != EXPECTED_SOURCE["artifact_id"] or publication_digest != EXPECTED_SOURCE["artifact_digest"]:
        raise DiagnosticsError("publication artifact identity drift")
    if publication.get("head_sha") != EXPECTED_SOURCE["head_sha"]:
        raise DiagnosticsError("publication head SHA drift")

    rows = _index(pipeline.get("results"), "case_id", "pipeline case")
    samples = retry.get("samples")
    if not isinstance(samples, list):
        raise DiagnosticsError("retry samples missing")
    observations: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        row = rows.get(case_id)
        if row is None:
            raise DiagnosticsError(f"missing expected case: {case_id}")
        provenance = row.get("metric_provenance")
        metrics = row.get("metrics")
        if not isinstance(provenance, dict) or provenance.get("schema") != PROVENANCE_SCHEMA:
            raise DiagnosticsError(f"unknown metric provenance for {case_id}")
        if not isinstance(metrics, dict):
            raise DiagnosticsError(f"metrics missing for {case_id}")
        missing = tuple(sorted(name for name in EXPECTED_GAPS if metrics.get(name) is None))
        if missing != EXPECTED_GAPS:
            raise DiagnosticsError(f"unexpected exact component gap for {case_id}: {missing}")
        artifact = row.get("artifact_sha256")
        if artifact != EXPECTED_ARTIFACT_SHAS[case_id] or provenance.get("selected_svg_sha256") != artifact:
            raise DiagnosticsError(f"artifact identity drift for {case_id}")
        observations.append(
            {
                "artifact_sha256": artifact,
                "case_id": case_id,
                "exact_evaluator_attempted": provenance.get("exact_evaluator_attempted"),
                "exact_evaluator_completed": provenance.get("exact_evaluator_completed"),
                "exact_evaluator_failure_class": provenance.get("exact_evaluator_failure_class"),
                "fallback_source": provenance.get("fallback_source"),
                "fallback_used": provenance.get("fallback_used"),
                "metric_source": provenance.get("metric_source"),
                "missing_exact_component_metrics": list(missing),
                "repeat_audit": _repeat_rows(samples, case_id),
                "repeat_metric_provenance_available": False,
                "selected_svg_file_present": provenance.get("selected_svg_file_present"),
                "selected_svg_path_present": provenance.get("selected_svg_path_present"),
                "selected_svg_sha256": provenance.get("selected_svg_sha256"),
            }
        )

    payload = {
        "analysis": {
            "aggregate_case_provenance_consistent": True,
            "cluster_root_cause_class": None,
            "diagnostic_failure_class": "provenance_missing_repeat_level_evaluator_report_detail",
            "next_branch": "agent/rfv-3e-exact-metric-path-provenance-completion",
            "observed_failure_class": "exact_metrics_incomplete",
            "observed_failure_class_status": "proven_at_aggregate_case_level",
            "original_routing_hypothesis_status": "disproven",
            "production_fix_allowed": False,
            "production_fix_forbidden_scope": [
                "measurement-path routing change",
                "final-artifact evaluator behavior change",
                "winner selection or serializer change",
                "threshold, corpus, repeat-count or release-decision change",
            ],
            "production_fix_scope": [],
            "root_cause_status": "unresolved",
            "unresolved_reason": (
                "The published artifact proves that the winner path and file existed and the exact evaluator was attempted, "
                "but it does not publish per-repeat evaluator provenance, evaluator report status, hard-failure codes, or "
                "which evaluator metric group was absent. exact_metrics_incomplete is an observed result class, not the "
                "production cause of that result."
            ),
        },
        "bindings": dict(EXPECTED_BINDINGS),
        "observations": observations,
        "plan_claim_assessment": {
            "cluster_id": "exact-metric-path-fallback",
            "plan_claim": "selected winner SVG does not reach the exact final-artifact evaluator",
            "requires_plan_correction_before_production_fix": True,
            "status": "superseded_by_live_diagnostics",
        },
        "release_decision": "no_go",
        "rfv4_allowed": False,
        "schema": SCHEMA,
        "scope": {"case_ids": list(EXPECTED_CASES), "cluster_id": "exact-metric-path-fallback"},
        "source": dict(EXPECTED_SOURCE),
    }
    reject_sensitive(payload)
    return payload


def verify_evidence(
    evidence_path: Path = DEFAULT_EVIDENCE,
    plan_path: Path = DEFAULT_PLAN,
    decision_path: Path = DEFAULT_DECISION,
    roadmap_path: Path = DEFAULT_ROADMAP,
) -> dict[str, Any]:
    evidence = load_json(evidence_path)
    plan = load_json(plan_path)
    decision = load_json(decision_path)
    roadmap = load_json(roadmap_path)

    if evidence.get("schema") != SCHEMA:
        raise DiagnosticsError("unexpected diagnostics schema")
    if evidence.get("source") != EXPECTED_SOURCE or evidence.get("bindings") != EXPECTED_BINDINGS:
        raise DiagnosticsError("source or canonical binding drift")
    if git_blob_sha(plan_path) != EXPECTED_BINDINGS["remediation_plan_git_blob_sha"]:
        raise DiagnosticsError("remediation plan Git blob binding drift")
    if plan.get("source_results_sha256") != EXPECTED_BINDINGS["canonical_results_sha256"]:
        raise DiagnosticsError("plan results binding drift")
    if plan.get("source_decision_file_sha256") != EXPECTED_BINDINGS["canonical_decision_file_sha256"]:
        raise DiagnosticsError("plan decision file binding drift")
    if plan.get("source_decision_sha256") != EXPECTED_BINDINGS["canonical_decision_sha256"]:
        raise DiagnosticsError("plan decision identity drift")
    if plan.get("source_cases_sha256") != EXPECTED_BINDINGS["corpus_or_case_set_sha256"]:
        raise DiagnosticsError("plan corpus identity drift")
    if plan.get("thresholds") != EXPECTED_THRESHOLDS:
        raise DiagnosticsError("plan threshold drift")

    cluster = next((item for item in plan.get("clusters", []) if item.get("cluster_id") == "exact-metric-path-fallback"), None)
    if not isinstance(cluster, dict) or tuple(cluster.get("evidence_case_ids", [])) != EXPECTED_CASES:
        raise DiagnosticsError("priority-one cluster scope drift")

    if decision.get("release_decision") != "no_go" or decision.get("rfv4_allowed") is not False:
        raise DiagnosticsError("canonical release decision drift")
    if decision.get("decision_sha256") != EXPECTED_BINDINGS["canonical_decision_sha256"]:
        raise DiagnosticsError("canonical decision digest drift")
    if decision.get("cases_sha256") != EXPECTED_BINDINGS["corpus_or_case_set_sha256"]:
        raise DiagnosticsError("canonical case-set identity drift")
    quality_metrics = (decision.get("quality_gate") or {}).get("metrics") or {}
    for name, threshold in EXPECTED_THRESHOLDS.items():
        if (quality_metrics.get(name) or {}).get("threshold") != threshold:
            raise DiagnosticsError(f"canonical threshold drift: {name}")

    phase_status = {item.get("id"): item.get("status") for item in roadmap.get("phases", []) if isinstance(item, dict)}
    if phase_status.get("RFV-3") != "pending" or phase_status.get("RFV-4") != "pending":
        raise DiagnosticsError("roadmap phase status drift")

    scope = evidence.get("scope") or {}
    if tuple(scope.get("case_ids", [])) != EXPECTED_CASES or scope.get("cluster_id") != "exact-metric-path-fallback":
        raise DiagnosticsError("diagnostic scope drift")
    observations = evidence.get("observations")
    indexed = _index(observations, "case_id", "diagnostic case")
    if tuple(indexed) != EXPECTED_CASES:
        raise DiagnosticsError("diagnostic case order or identity drift")
    for case_id in EXPECTED_CASES:
        row = indexed[case_id]
        artifact = EXPECTED_ARTIFACT_SHAS[case_id]
        if row.get("artifact_sha256") != artifact or row.get("selected_svg_sha256") != artifact:
            raise DiagnosticsError(f"artifact binding drift for {case_id}")
        if row.get("selected_svg_path_present") is not True or row.get("selected_svg_file_present") is not True:
            raise DiagnosticsError(f"routing evidence drift for {case_id}")
        if row.get("exact_evaluator_attempted") is not True or row.get("exact_evaluator_completed") is not False:
            raise DiagnosticsError(f"evaluator state drift for {case_id}")
        if row.get("exact_evaluator_failure_class") != "exact_metrics_incomplete":
            raise DiagnosticsError(f"failure class drift for {case_id}")
        if row.get("fallback_used") is not True or row.get("fallback_source") != "partial_quality_report":
            raise DiagnosticsError(f"fallback provenance drift for {case_id}")
        if row.get("metric_source") != "partial_quality_report":
            raise DiagnosticsError(f"metric source drift for {case_id}")
        if tuple(row.get("missing_exact_component_metrics", [])) != EXPECTED_GAPS:
            raise DiagnosticsError(f"missing component drift for {case_id}")
        repeats = row.get("repeat_audit")
        if not isinstance(repeats, list) or [item.get("repeat_index") for item in repeats] != [1, 2, 3]:
            raise DiagnosticsError(f"repeat coverage drift for {case_id}")
        if any(item.get("status") != "success" or item.get("attempt_count") != 1 or item.get("retried") is not False for item in repeats):
            raise DiagnosticsError(f"repeat audit drift for {case_id}")
        if row.get("repeat_metric_provenance_available") is not False:
            raise DiagnosticsError(f"repeat provenance availability drift for {case_id}")

    analysis = evidence.get("analysis") or {}
    exact_analysis = {
        "aggregate_case_provenance_consistent": True,
        "cluster_root_cause_class": None,
        "diagnostic_failure_class": "provenance_missing_repeat_level_evaluator_report_detail",
        "next_branch": "agent/rfv-3e-exact-metric-path-provenance-completion",
        "observed_failure_class": "exact_metrics_incomplete",
        "observed_failure_class_status": "proven_at_aggregate_case_level",
        "original_routing_hypothesis_status": "disproven",
        "production_fix_allowed": False,
        "production_fix_scope": [],
        "root_cause_status": "unresolved",
    }
    for key, value in exact_analysis.items():
        if analysis.get(key) != value:
            raise DiagnosticsError(f"analysis drift: {key}")
    if not analysis.get("production_fix_forbidden_scope") or not analysis.get("unresolved_reason"):
        raise DiagnosticsError("fail-closed analysis detail missing")
    assessment = evidence.get("plan_claim_assessment") or {}
    if assessment.get("status") != "superseded_by_live_diagnostics" or assessment.get("requires_plan_correction_before_production_fix") is not True:
        raise DiagnosticsError("plan claim assessment drift")
    if evidence.get("release_decision") != "no_go" or evidence.get("rfv4_allowed") is not False:
        raise DiagnosticsError("diagnostic release gate drift")

    reject_sensitive(evidence)
    digest_a = canonical_sha256(evidence)
    digest_b = canonical_sha256(json.loads(canonical_bytes(evidence)))
    if digest_a != digest_b:
        raise DiagnosticsError("non-deterministic canonical serialization")
    return {
        "case_count": 3,
        "canonical_evidence_sha256": digest_a,
        "observed_failure_class": "exact_metrics_incomplete",
        "original_routing_hypothesis_status": "disproven",
        "production_fix_allowed": False,
        "release_decision": "no_go",
        "rfv4_allowed": False,
        "root_cause_status": "unresolved",
        "schema": "vektoryum-rfv3e-exact-metric-path-fallback-diagnostics-verification-v1",
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--pipeline-results", type=Path, required=True)
    build.add_argument("--retry-audit", type=Path, required=True)
    build.add_argument("--measurement-envelope", type=Path, required=True)
    build.add_argument("--publication-envelope", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    verify.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    verify.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    verify.add_argument("--roadmap", type=Path, default=DEFAULT_ROADMAP)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_evidence(args.pipeline_results, args.retry_audit, args.measurement_envelope, args.publication_envelope)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(payload))
        print(json.dumps({"output": str(args.output), "status": "built"}, sort_keys=True))
    else:
        print(json.dumps(verify_evidence(args.evidence, args.plan, args.decision, args.roadmap), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
