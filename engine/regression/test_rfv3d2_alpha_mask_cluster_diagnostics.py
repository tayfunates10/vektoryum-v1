"""Tests for RFV-3D2 alpha/mask cluster diagnostics."""
from __future__ import annotations
import copy, json, unittest
from pathlib import Path
import numpy as np
from engine.regression.rfv3d2_alpha_mask_cluster_diagnostics import CASES, LIVE_ARTIFACTS, SCHEMA, diagnose_alpha_pair, validate_evidence
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/real_world_fidelity/evidence/rfv3d2_alpha_mask_cluster_diagnostics.json"

class RFV3D2AlphaMaskClusterDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    def test_committed_evidence_is_valid(self):
        validate_evidence(self.payload); self.assertEqual(self.payload["schema"], SCHEMA); self.assertEqual(self.payload["scope"]["case_ids"], CASES)
    def test_live_artifact_bindings_and_scalars_are_committed(self):
        live=self.payload["live_confirmation"]; self.assertEqual([r["case_id"] for r in live["cases"]], CASES)
        for row in live["cases"]:
            self.assertEqual((row["artifact_id"],row["artifact_digest"]), LIVE_ARTIFACTS[row["case_id"]])
            self.assertEqual(row["diagnosis"], "opaque_canvas_collapse")
            self.assertGreaterEqual(row["render_soft_coverage"], .995)
            self.assertLessEqual(abs(row["alpha_iou"]-row["source_soft_coverage"]), 1e-4)
    def test_full_canvas_opaque_render_is_classified(self):
        source=np.array([[0,64],[128,255]],dtype=np.uint8); render=np.full_like(source,255); result=diagnose_alpha_pair(source,render)
        self.assertEqual(result["diagnosis"],"opaque_canvas_collapse"); self.assertAlmostEqual(result["metrics"]["alpha_iou"],result["source"]["soft_coverage"])
    def test_preserved_alpha_is_classified(self):
        source=np.array([[0,64],[128,255]],dtype=np.uint8); result=diagnose_alpha_pair(source,source.copy()); self.assertEqual(result["diagnosis"],"alpha_preserved")
    def test_opaque_source_is_not_scope(self):
        source=np.full((4,4),255,dtype=np.uint8); self.assertEqual(diagnose_alpha_pair(source,np.zeros((4,4),dtype=np.uint8))["diagnosis"],"not_applicable_opaque_source")
    def test_tampered_live_binding_metric_or_hash_is_rejected(self):
        for field,value in (("artifact_id",1),("render_soft_coverage",.5),("selected_svg_sha256","bad")):
            with self.subTest(field=field):
                tampered=copy.deepcopy(self.payload); tampered["live_confirmation"]["cases"][0][field]=value
                with self.assertRaises(ValueError): validate_evidence(tampered)
    def test_release_or_authorization_drift_is_rejected(self):
        for field,value in (("release_decision","go"),("rfv4_allowed",True)):
            tampered=copy.deepcopy(self.payload); tampered[field]=value
            with self.assertRaisesRegex(ValueError,"decision drift"): validate_evidence(tampered)
        tampered=copy.deepcopy(self.payload); tampered["diagnosis"]["production_fix_authorized"]=True
        with self.assertRaisesRegex(ValueError,"cannot authorize"): validate_evidence(tampered)
    def test_path_or_secret_leakage_is_rejected(self):
        tampered=copy.deepcopy(self.payload); tampered["diagnosis"]["summary"]="/tmp/raw/selected.svg"
        with self.assertRaisesRegex(ValueError,"leaked"): validate_evidence(tampered)

if __name__ == "__main__": unittest.main()
