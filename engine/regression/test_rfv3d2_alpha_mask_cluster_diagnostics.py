"""Tests for RFV-3D2 alpha/mask cluster diagnostics."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from engine.regression.rfv3d2_alpha_mask_cluster_diagnostics import (
    CASES,
    SCHEMA,
    diagnose_alpha_pair,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/real_world_fidelity/evidence/rfv3d2_alpha_mask_cluster_diagnostics.json"


class RFV3D2AlphaMaskClusterDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_committed_evidence_is_valid(self):
        validate_evidence(self.payload)
        self.assertEqual(self.payload["schema"], SCHEMA)
        self.assertEqual(self.payload["scope"]["case_ids"], CASES)

    def test_full_canvas_opaque_render_is_classified(self):
        source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        render = np.full_like(source, 255)
        result = diagnose_alpha_pair(source, render)
        self.assertEqual(result["diagnosis"], "opaque_canvas_collapse")
        self.assertAlmostEqual(result["metrics"]["alpha_iou"], result["source"]["soft_coverage"])
        self.assertAlmostEqual(result["render"]["soft_coverage"], 1.0)

    def test_preserved_alpha_is_classified(self):
        source = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        result = diagnose_alpha_pair(source, source.copy())
        self.assertEqual(result["diagnosis"], "alpha_preserved")
        self.assertAlmostEqual(result["metrics"]["alpha_iou"], 1.0)
        self.assertNotIn("raw_alpha", json.dumps(result, sort_keys=True))

    def test_opaque_source_is_not_alpha_remediation_scope(self):
        source = np.full((4, 4), 255, dtype=np.uint8)
        render = np.zeros((4, 4), dtype=np.uint8)
        result = diagnose_alpha_pair(source, render)
        self.assertEqual(result["diagnosis"], "not_applicable_opaque_source")

    def test_tampered_binding_or_signature_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["source"]["workflow_run_id"] += 1
        with self.assertRaisesRegex(ValueError, "source binding"):
            validate_evidence(tampered)

        tampered = copy.deepcopy(self.payload)
        tampered["observations"][0]["absolute_signature_gap"] = 0.5
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_evidence(tampered)

    def test_release_or_authorization_drift_is_rejected(self):
        for field, value in (("release_decision", "go"), ("rfv4_allowed", True)):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.payload)
                tampered[field] = value
                with self.assertRaisesRegex(ValueError, "decision drift"):
                    validate_evidence(tampered)
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["production_fix_authorized"] = True
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            validate_evidence(tampered)

    def test_path_or_secret_leakage_is_rejected(self):
        tampered = copy.deepcopy(self.payload)
        tampered["diagnosis"]["summary"] = "/tmp/raw/selected.svg"
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
