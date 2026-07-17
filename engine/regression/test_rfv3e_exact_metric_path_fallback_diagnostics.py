from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from engine.regression.rfv3e_exact_metric_path_fallback_diagnostics import (
    DEFAULT_DECISION,
    DEFAULT_EVIDENCE,
    DEFAULT_PLAN,
    DEFAULT_ROADMAP,
    DiagnosticsError,
    canonical_bytes,
    canonical_sha256,
    load_json,
    verify_evidence,
)


Mutator = Callable[[dict[str, Any]], None]


def _set(path: tuple[Any, ...], value: Any) -> Mutator:
    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    return mutate


def _delete(path: tuple[Any, ...]) -> Mutator:
    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for part in path[:-1]:
            target = target[part]
        del target[path[-1]]
    return mutate


def _append(path: tuple[Any, ...], value: Any) -> Mutator:
    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for part in path:
            target = target[part]
        target.append(value)
    return mutate


class RFV3EExactMetricPathDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = load_json(DEFAULT_EVIDENCE)
        self.plan = load_json(DEFAULT_PLAN)
        self.decision = load_json(DEFAULT_DECISION)
        self.roadmap = load_json(DEFAULT_ROADMAP)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_bytes(canonical_bytes(payload))
        return path

    def _reject_evidence(self, mutator: Mutator) -> None:
        payload = copy.deepcopy(self.evidence)
        mutator(payload)
        with self.assertRaises(DiagnosticsError):
            verify_evidence(self._write("evidence.json", payload))

    def _reject_with_bindings(
        self,
        *,
        plan: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        roadmap: dict[str, Any] | None = None,
    ) -> None:
        with self.assertRaises(DiagnosticsError):
            verify_evidence(
                DEFAULT_EVIDENCE,
                self._write("plan.json", plan or self.plan),
                self._write("decision.json", decision or self.decision),
                self._write("roadmap.json", roadmap or self.roadmap),
            )

    def test_committed_evidence_verifies(self) -> None:
        report = verify_evidence()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["root_cause_status"], "unresolved")
        self.assertFalse(report["production_fix_allowed"])

    def test_canonical_serialization_is_deterministic(self) -> None:
        a = canonical_sha256(self.evidence)
        b = canonical_sha256(json.loads(canonical_bytes(self.evidence)))
        self.assertEqual(a, b)

    def test_plan_threshold_drift_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["thresholds"]["ssim"] = 0.97
        self._reject_with_bindings(plan=plan)

    def test_plan_case_scope_drift_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        cluster = next(x for x in plan["clusters"] if x["cluster_id"] == "exact-metric-path-fallback")
        cluster["evidence_case_ids"] = cluster["evidence_case_ids"][:-1]
        self._reject_with_bindings(plan=plan)

    def test_plan_results_binding_drift_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["source_results_sha256"] = "0" * 64
        self._reject_with_bindings(plan=plan)

    def test_decision_release_drift_rejected(self) -> None:
        decision = copy.deepcopy(self.decision)
        decision["release_decision"] = "go"
        self._reject_with_bindings(decision=decision)

    def test_decision_threshold_drift_rejected(self) -> None:
        decision = copy.deepcopy(self.decision)
        decision["quality_gate"]["metrics"]["edge_f1"]["threshold"] = 0.97
        self._reject_with_bindings(decision=decision)

    def test_decision_corpus_drift_rejected(self) -> None:
        decision = copy.deepcopy(self.decision)
        decision["cases_sha256"] = "0" * 64
        self._reject_with_bindings(decision=decision)

    def test_roadmap_unlock_rejected(self) -> None:
        roadmap = copy.deepcopy(self.roadmap)
        next(x for x in roadmap["phases"] if x["id"] == "RFV-4")["status"] = "merged"
        self._reject_with_bindings(roadmap=roadmap)


EVIDENCE_MUTATIONS: dict[str, Mutator] = {
    "schema_drift": _set(("schema",), "unknown"),
    "head_sha_drift": _set(("source", "head_sha"), "0" * 40),
    "run_id_missing": _delete(("source", "workflow_run_id")),
    "run_attempt_drift": _set(("source", "workflow_run_attempt"), 2),
    "artifact_id_missing": _delete(("source", "artifact_id")),
    "artifact_digest_drift": _set(("source", "artifact_digest"), "sha256:" + "0" * 64),
    "source_file_digest_drift": _set(("source", "pipeline_results_file_sha256"), "0" * 64),
    "plan_blob_binding_drift": _set(("bindings", "remediation_plan_git_blob_sha"), "0" * 40),
    "canonical_results_drift": _set(("bindings", "canonical_results_sha256"), "0" * 64),
    "canonical_decision_drift": _set(("bindings", "canonical_decision_sha256"), "0" * 64),
    "corpus_binding_drift": _set(("bindings", "corpus_or_case_set_sha256"), "0" * 64),
    "missing_scope_case": _set(("scope", "case_ids"), ["qualification-public-10", "qualification-public-14"]),
    "extra_scope_case": _append(("scope", "case_ids"), "qualification-public-01"),
    "observation_order_drift": lambda p: p["observations"].reverse(),
    "missing_observation": lambda p: p["observations"].pop(),
    "duplicate_observation": _append(("observations",), copy.deepcopy(load_json(DEFAULT_EVIDENCE)["observations"][0])),
    "selected_path_missing": _set(("observations", 0, "selected_svg_path_present"), False),
    "selected_file_missing": _set(("observations", 0, "selected_svg_file_present"), False),
    "evaluator_not_attempted": _set(("observations", 0, "exact_evaluator_attempted"), False),
    "evaluator_completed_fabricated": _set(("observations", 0, "exact_evaluator_completed"), True),
    "unknown_failure_class": _set(("observations", 0, "exact_evaluator_failure_class"), "unknown"),
    "fallback_disabled": _set(("observations", 0, "fallback_used"), False),
    "fallback_source_drift": _set(("observations", 0, "fallback_source"), "other"),
    "metric_source_drift": _set(("observations", 0, "metric_source"), "exact_final_artifact"),
    "component_gap_drift": _set(("observations", 0, "missing_exact_component_metrics"), ["ssim"]),
    "artifact_identity_drift": _set(("observations", 0, "artifact_sha256"), "0" * 64),
    "selected_artifact_identity_drift": _set(("observations", 0, "selected_svg_sha256"), "0" * 64),
    "repeat_gap": lambda p: p["observations"][0]["repeat_audit"].pop(),
    "duplicate_repeat": _set(("observations", 0, "repeat_audit", 2, "repeat_index"), 2),
    "repeat_failure": _set(("observations", 0, "repeat_audit", 0, "status"), "failed"),
    "retry_fabricated": _set(("observations", 0, "repeat_audit", 0, "retried"), True),
    "repeat_attempt_drift": _set(("observations", 0, "repeat_audit", 0, "attempt_count"), 2),
    "repeat_provenance_fabricated": _set(("observations", 0, "repeat_metric_provenance_available"), True),
    "root_cause_promoted": _set(("analysis", "root_cause_status"), "proven"),
    "production_fix_enabled": _set(("analysis", "production_fix_allowed"), True),
    "production_fix_scope_added": _append(("analysis", "production_fix_scope"), "routing fix"),
    "routing_hypothesis_restored": _set(("analysis", "original_routing_hypothesis_status"), "proven"),
    "next_branch_drift": _set(("analysis", "next_branch"), "agent/rfv-3e-exact-metric-path-fallback-fix"),
    "plan_assessment_drift": _set(("plan_claim_assessment", "status"), "current"),
    "plan_correction_gate_removed": _set(("plan_claim_assessment", "requires_plan_correction_before_production_fix"), False),
    "release_go_fabricated": _set(("release_decision",), "go"),
    "rfv4_unlocked": _set(("rfv4_allowed",), True),
    "posix_path_leak": _set(("analysis", "unresolved_reason"), "found at /home/runner/work/private/file"),
    "windows_path_leak": _set(("analysis", "unresolved_reason"), r"found at C:\\Users\\runner\\private.txt"),
    "secret_leak": _set(("analysis", "unresolved_reason"), "Authorization: bearer abcdefghijklmnop"),
    "traceback_leak": _set(("analysis", "unresolved_reason"), "Traceback (most recent call last): hidden"),
}


def _make_mutation_test(mutator: Mutator):
    def test(self: RFV3EExactMetricPathDiagnosticsTests) -> None:
        self._reject_evidence(mutator)
    return test


for _name, _mutator in EVIDENCE_MUTATIONS.items():
    setattr(
        RFV3EExactMetricPathDiagnosticsTests,
        f"test_{_name}_rejected",
        _make_mutation_test(_mutator),
    )


if __name__ == "__main__":
    unittest.main()
