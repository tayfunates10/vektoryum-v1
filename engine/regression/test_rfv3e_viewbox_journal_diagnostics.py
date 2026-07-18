"""Tests for the fail-closed RFV-3E viewBox journal diagnosis."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from engine.regression.rfv3e_viewbox_journal_diagnostics import (
    CASES,
    SCHEMA,
    run_diagnostic,
    validate_evidence,
)


class RFV3EViewBoxJournalDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = run_diagnostic()

    def test_exact_live_scope_and_bindings(self):
        self.assertEqual(self.payload["schema"], SCHEMA)
        self.assertEqual(self.payload["scope"]["case_ids"], CASES)
        self.assertEqual(self.payload["scope"]["observed_hard_fail_code"], "viewbox_missing")
        self.assertEqual(self.payload["source"]["pull_request"], 103)
        self.assertEqual(self.payload["source"]["workflow_run_id"], 29623130466)
        self.assertEqual(self.payload["source"]["aggregate_artifact_id"], 8424383328)

    def test_restore_can_add_viewbox_but_alpha_requirement_rolls_it_back(self):
        diagnosis = self.payload["diagnosis"]
        self.assertTrue(diagnosis["direct_restore_added_viewbox"])
        self.assertEqual(diagnosis["repaired_viewbox"], "0 0 48 32")
        self.assertTrue(diagnosis["stage_measurement_structural_safe"])
        self.assertTrue(diagnosis["stage_measurement_visual_metrics_complete"])
        self.assertEqual(diagnosis["stage_measurement_required_unmeasured"], ["alpha_fidelity"])
        self.assertEqual(
            diagnosis["journal_with_alpha_requirement"],
            {
                "accepted": False,
                "status": "rolled_back",
                "reason_codes": ["required_metric_unmeasured"],
                "output_viewbox_present": False,
            },
        )

    def test_same_transform_is_accepted_when_no_required_metric_is_unmeasured(self):
        self.assertEqual(
            self.payload["diagnosis"]["journal_without_alpha_requirement"],
            {
                "accepted": True,
                "status": "accepted",
                "reason_codes": ["metrics_non_regressing"],
                "output_viewbox_present": True,
            },
        )

    def test_root_cause_is_proven_but_fix_is_not_authorized(self):
        diagnosis = self.payload["diagnosis"]
        self.assertEqual(diagnosis["root_cause_status"], "proven")
        self.assertEqual(
            diagnosis["root_cause_class"],
            "transform_journal_required_alpha_metric_deadlock",
        )
        self.assertFalse(diagnosis["production_fix_authorized"])
        self.assertEqual(self.payload["release_decision"], "no_go")
        self.assertFalse(self.payload["rfv4_allowed"])

    def test_serialization_is_deterministic(self):
        first = json.dumps(run_diagnostic(), sort_keys=True, separators=(",", ":"))
        second = json.dumps(run_diagnostic(), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_committed_shape_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        validate_evidence(loaded)

    def test_tampered_required_metric_signature_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["stage_measurement_required_unmeasured"] = []
        with self.assertRaisesRegex(ValueError, "required-unmeasured"):
            validate_evidence(tampered)

    def test_tampered_rollback_reason_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["journal_with_alpha_requirement"]["reason_codes"] = ["viewbox_missing"]
        with self.assertRaisesRegex(ValueError, "rollback signature"):
            validate_evidence(tampered)

    def test_tampered_control_acceptance_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["journal_without_alpha_requirement"]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "control acceptance"):
            validate_evidence(tampered)

    def test_production_fix_authorization_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["production_fix_authorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            validate_evidence(tampered)

    def test_release_or_rfv4_drift_is_rejected(self):
        for field, value in (("release_decision", "go"), ("rfv4_allowed", True)):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.payload)
                tampered[field] = value
                with self.assertRaisesRegex(ValueError, "decision drift"):
                    validate_evidence(tampered)

    def test_path_or_secret_leakage_is_rejected(self):
        for value in ("/tmp/raw/winner.svg", "C:\\runner\\winner.svg", "Bearer secret"):
            with self.subTest(value=value):
                tampered = copy.deepcopy(self.payload)
                tampered["diagnosis"]["root_cause_summary"] = value
                with self.assertRaisesRegex(ValueError, "leaked"):
                    validate_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
