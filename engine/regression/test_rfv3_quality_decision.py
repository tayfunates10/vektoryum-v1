import copy
import unittest
from pathlib import Path

from engine.regression.rfv3_quality_decision import (
    DECISION_PATH,
    MEASUREMENT_ENVELOPE_PATH,
    PIPELINE_RESULTS_PATH,
    POLICY_PATH,
    PUBLICATION_ENVELOPE_PATH,
    QUALIFICATION_MANIFEST_PATH,
    RETRY_AUDIT_PATH,
    DecisionError,
    evaluate_committed_evidence,
    evaluate_quality_decision,
    load_json,
    validate_policy,
)

ROADMAP_PATH = Path(__file__).resolve().parents[2] / "docs" / "real_world_fidelity_roadmap.json"


class RFV3QualityDecisionTests(unittest.TestCase):
    def setUp(self):
        self.policy = validate_policy(load_json(POLICY_PATH))
        self.qualification = load_json(QUALIFICATION_MANIFEST_PATH)
        self.pipeline = load_json(PIPELINE_RESULTS_PATH)
        self.retry = load_json(RETRY_AUDIT_PATH)
        self.measurement = load_json(MEASUREMENT_ENVELOPE_PATH)
        self.publication = load_json(PUBLICATION_ENVELOPE_PATH)

    def evaluate(self, **overrides):
        return evaluate_quality_decision(
            overrides.get("policy", self.policy),
            overrides.get("qualification", self.qualification),
            overrides.get("pipeline", self.pipeline),
            overrides.get("retry", self.retry),
            overrides.get("measurement", self.measurement),
            overrides.get("publication", self.publication),
        )

    def test_committed_decision_is_reproducible_and_honest(self):
        decision = evaluate_committed_evidence()
        self.assertEqual(decision, load_json(DECISION_PATH))
        self.assertEqual(decision["case_count"], 24)
        self.assertEqual(decision["repeat_sample_count"], 72)
        self.assertEqual(decision["completeness_gate"]["status"], "passed")
        self.assertEqual(decision["retry_gate"]["status"], "passed")
        self.assertEqual(decision["artifact_determinism_gate"]["status"], "passed")
        self.assertEqual(decision["quality_gate"]["status"], "failed")
        self.assertEqual(decision["quality_gate"]["violation_case_count"], 24)
        self.assertEqual(decision["release_decision"], "no_go")
        self.assertFalse(decision["rfv4_allowed"])
        self.assertFalse(decision["raw_assets_in_repository"])

    def test_exact_empirical_threshold_counts_are_frozen(self):
        decision = self.evaluate()
        metrics = decision["quality_gate"]["metrics"]
        self.assertEqual(metrics["fidelity"]["pass_count"], 0)
        self.assertEqual(metrics["fidelity"]["violation_count"], 24)
        self.assertEqual(metrics["ssim"]["missing_count"], 3)
        self.assertEqual(metrics["ssim"]["pass_count"], 8)
        self.assertEqual(metrics["edge_f1"]["missing_count"], 3)
        self.assertEqual(metrics["edge_f1"]["pass_count"], 8)
        self.assertEqual(metrics["alpha_iou"]["pass_count"], 3)
        self.assertEqual(decision["retry_gate"]["retried_sample_count"], 0)
        self.assertEqual(decision["retry_gate"]["max_attempt_count"], 1)

    def test_shrinkage_duplicates_and_mixed_engine_fail_closed(self):
        shrunk = copy.deepcopy(self.pipeline)
        shrunk["results"].pop()
        shrunk["case_count"] = 23
        with self.assertRaises(DecisionError):
            self.evaluate(pipeline=shrunk)

        duplicate = copy.deepcopy(self.pipeline)
        duplicate["results"][1]["case_id"] = duplicate["results"][0]["case_id"]
        with self.assertRaises(DecisionError):
            self.evaluate(pipeline=duplicate)

        mixed = copy.deepcopy(self.pipeline)
        mixed["results"][0]["engine_version"] = "0" * 40
        with self.assertRaises(DecisionError):
            self.evaluate(pipeline=mixed)

    def test_retry_budget_and_retry_class_fail_closed(self):
        over_budget = copy.deepcopy(self.retry)
        sample = over_budget["samples"][0]
        sample["attempt_count"] = 3
        sample["retried"] = True
        sample["attempts"].append({"attempt": 2, "status": "failed", "retry_class": "TimeoutError", "error": "timeout"})
        sample["attempts"].append({"attempt": 3, "status": "success", "retry_class": None, "error": None})
        with self.assertRaises(DecisionError):
            self.evaluate(retry=over_budget)

        unapproved = copy.deepcopy(self.retry)
        sample = unapproved["samples"][0]
        sample["attempt_count"] = 2
        sample["retried"] = True
        sample["attempts"] = [
            {"attempt": 1, "status": "failed", "retry_class": "network_error", "error": "network"},
            {"attempt": 2, "status": "success", "retry_class": None, "error": None},
        ]
        with self.assertRaises(DecisionError):
            self.evaluate(retry=unapproved)

    def test_missing_quality_metric_forces_no_go_without_fabrication(self):
        pipeline = copy.deepcopy(self.pipeline)
        pipeline["results"][0]["metrics"]["ssim"] = None
        decision = self.evaluate(pipeline=pipeline)
        self.assertEqual(decision["quality_gate"]["status"], "failed")
        self.assertEqual(decision["release_decision"], "no_go")
        self.assertTrue(any(item["case_id"] == pipeline["results"][0]["case_id"] for item in decision["quality_gate"]["missing_metric_cases"]))

    def test_roadmap_records_reviewed_no_go_but_waits_for_phase_closure(self):
        roadmap = load_json(ROADMAP_PATH)
        phases = roadmap["phases"]
        self.assertEqual([phase["id"] for phase in phases], ["RFV-1", "RFV-2", "RFV-3", "RFV-4"])
        self.assertEqual([phase["status"] for phase in phases], ["merged", "merged", "pending", "pending"])
        self.assertEqual(phases[2]["release_decision"], "no_go")
        self.assertEqual(phases[2]["decision_evidence"], "docs/real_world_fidelity/evidence/rfv3_quality_decision.json")
        self.assertIn("phase closure", phases[2]["blocked_on"])
        self.assertIn("passing real-world quality rerun", phases[3]["blocked_on"])


if __name__ == "__main__":
    unittest.main()
