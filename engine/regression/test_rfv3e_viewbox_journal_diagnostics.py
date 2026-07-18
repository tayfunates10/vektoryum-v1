"""Tests for immutable historical RFV-3E viewBox evidence."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from engine.regression.rfv3e_viewbox_journal_diagnostics import (
    CASES, SCHEMA, validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/real_world_fidelity/evidence/rfv3e_viewbox_journal_diagnostics.json"


class RFV3EViewBoxJournalDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_committed_historical_evidence_is_valid(self):
        validate_evidence(self.payload)

    def test_exact_historical_bindings(self):
        self.assertEqual(self.payload["schema"], SCHEMA)
        self.assertEqual(self.payload["scope"]["case_ids"], CASES)
        self.assertEqual(self.payload["source"]["pull_request"], 103)
        self.assertEqual(self.payload["source"]["workflow_run_id"], 29623130466)
        self.assertEqual(self.payload["source"]["aggregate_artifact_id"], 8424383328)

    def test_historical_root_cause_signature(self):
        diagnosis = self.payload["diagnosis"]
        self.assertEqual(diagnosis["stage_measurement_required_unmeasured"], ["alpha_fidelity"])
        self.assertEqual(diagnosis["root_cause_class"], "transform_journal_required_alpha_metric_deadlock")
        self.assertFalse(diagnosis["production_fix_authorized"])

    def test_tampered_historical_signature_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["stage_measurement_required_unmeasured"] = []
        with self.assertRaisesRegex(ValueError, "required-unmeasured"):
            validate_evidence(tampered)

    def test_release_or_rfv4_drift_is_rejected(self):
        for field, value in (("release_decision", "go"), ("rfv4_allowed", True)):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.payload)
                tampered[field] = value
                with self.assertRaisesRegex(ValueError, "decision drift"):
                    validate_evidence(tampered)

    def test_path_or_secret_leakage_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["root_cause_summary"] = "/tmp/raw/winner.svg"
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
