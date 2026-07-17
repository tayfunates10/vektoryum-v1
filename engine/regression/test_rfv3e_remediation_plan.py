"""RFV-3E remediation planının fail-closed doğrulaması.

Committed plan gerçek evidence'e SHA ile bağlı olmalı; 24 vakanın tamamı
kapsanmalı, her failing vaka tam bir kümede olmalı, geçen vaka remediation
listesine girmemeli, eşikler değişmemeli ve NO-GO/rfv4_allowed=false korunmalı.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from regression.rfv3e_remediation_plan import (
    DEFAULT_DECISION,
    DEFAULT_PLAN,
    DEFAULT_RESULTS,
    REQUIRED_THRESHOLDS,
    RemediationPlanError,
    case_violations,
    verify_plan,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CommittedPlanTest(unittest.TestCase):
    def test_committed_plan_verifies(self) -> None:
        report = verify_plan()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["case_count"], 24)
        self.assertEqual(report["failing_case_count"], 24)
        self.assertEqual(report["passing_case_count"], 0)
        self.assertEqual(report["release_decision"], "no_go")
        self.assertIs(report["rfv4_allowed"], False)

    def test_every_case_clustered_exactly_once(self) -> None:
        plan = _load(DEFAULT_PLAN)
        results = _load(DEFAULT_RESULTS)
        clustered = [
            case_id
            for cluster in plan["clusters"]
            for case_id in cluster["evidence_case_ids"]
        ]
        expected = sorted(row["case_id"] for row in results["results"])
        self.assertEqual(sorted(clustered), expected)
        self.assertEqual(len(clustered), len(set(clustered)))

    def test_clustered_cases_really_fail(self) -> None:
        plan = _load(DEFAULT_PLAN)
        results = {row["case_id"]: row for row in _load(DEFAULT_RESULTS)["results"]}
        for cluster in plan["clusters"]:
            for case_id in cluster["evidence_case_ids"]:
                violated = case_violations(results[case_id]["metrics"], REQUIRED_THRESHOLDS)
                self.assertTrue(violated, f"{case_id} passes all thresholds but is listed")

    def test_null_required_metric_counts_as_violation(self) -> None:
        violated = case_violations({"fidelity": 100.0, "ssim": None, "edge_f1": 0.99, "alpha_iou": 0.99}, REQUIRED_THRESHOLDS)
        self.assertIn("ssim", violated)

    def test_measurement_fallback_cluster_matches_rfv3d1_diagnosis(self) -> None:
        plan = _load(DEFAULT_PLAN)
        clusters = {cluster["cluster_id"]: cluster for cluster in plan["clusters"]}
        fallback = clusters["exact-metric-path-fallback"]
        self.assertEqual(
            fallback["evidence_case_ids"],
            ["qualification-public-10", "qualification-public-14", "qualification-public-18"],
        )
        self.assertEqual(fallback["confidence"], "proven")

    def test_thresholds_unchanged(self) -> None:
        plan = _load(DEFAULT_PLAN)
        decision = _load(DEFAULT_DECISION)
        self.assertEqual(plan["thresholds"], {"fidelity": 99.0, "ssim": 0.98, "edge_f1": 0.98, "alpha_iou": 0.98})
        for name, expected in plan["thresholds"].items():
            self.assertEqual(decision["quality_gate"]["metrics"][name]["threshold"], expected)

    def test_release_stays_no_go(self) -> None:
        plan = _load(DEFAULT_PLAN)
        self.assertEqual(plan["release_decision"], "no_go")
        self.assertIs(plan["rfv4_allowed"], False)


class TamperedPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _load(DEFAULT_PLAN)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _write(self, plan: dict) -> Path:
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def _assert_rejected(self, plan: dict, fragment: str) -> None:
        with self.assertRaises(RemediationPlanError) as ctx:
            verify_plan(self._write(plan))
        self.assertIn(fragment, str(ctx.exception))

    def test_stale_results_binding_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["source_results_sha256"] = "0" * 64
        self._assert_rejected(tampered, "not bound to the committed pipeline results")

    def test_stale_decision_binding_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["source_decision_sha256"] = "0" * 64
        self._assert_rejected(tampered, "canonical decision digest")

    def test_corpus_identity_drift_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["source_cases_sha256"] = "0" * 64
        self._assert_rejected(tampered, "case-set SHA")

    def test_missing_case_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        removed = tampered["clusters"][0]["evidence_case_ids"].pop()
        tampered["clusters"][0]["verification_cases"] = tampered["clusters"][0]["evidence_case_ids"]
        tampered["clusters"][0]["regression_cases"] = [
            case_id for case_id in tampered["clusters"][0]["regression_cases"] if case_id != removed
        ] or ["qualification-public-01"]
        self._assert_rejected(tampered, "partition mismatch")

    def test_duplicate_case_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        case_id = tampered["clusters"][0]["evidence_case_ids"][0]
        tampered["clusters"][1]["evidence_case_ids"] = tampered["clusters"][1]["evidence_case_ids"] + [case_id]
        self._assert_rejected(tampered, "appears in both")

    def test_threshold_drift_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["thresholds"] = dict(tampered["thresholds"], fidelity=90.0)
        self._assert_rejected(tampered, "threshold drift")

    def test_go_decision_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["release_decision"] = "go"
        self._assert_rejected(tampered, "no_go")

    def test_rfv4_unlock_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["rfv4_allowed"] = True
        self._assert_rejected(tampered, "rfv4_allowed")

    def test_fabricated_failed_metrics_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        tampered["clusters"][-1]["failed_metrics"] = ["fidelity", "ssim"]
        self._assert_rejected(tampered, "do not match measured violations")

    def test_tentative_cluster_requires_diagnostics_scope(self) -> None:
        tampered = copy.deepcopy(self.plan)
        for cluster in tampered["clusters"]:
            if cluster["confidence"] == "tentative":
                cluster["allowed_change_scope"] = ["production scoring rewrite"]
        self._assert_rejected(tampered, "diagnostics-first")

    def test_regression_overlap_rejected(self) -> None:
        tampered = copy.deepcopy(self.plan)
        cluster = tampered["clusters"][0]
        cluster["regression_cases"] = list(cluster["regression_cases"]) + [cluster["evidence_case_ids"][0]]
        self._assert_rejected(tampered, "overlap evidence")


if __name__ == "__main__":
    unittest.main()
