from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class RQ4ReleaseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(Path("engine/regression/rq4_release_manifest.json").read_text())
        cls.roadmap = json.loads(Path("docs/release_qualification_roadmap.json").read_text())

    def test_manifest_is_finite_and_secret_safe(self) -> None:
        self.assertEqual(self.manifest["schema"], "vektoryum-rq4-beta-release-v1")
        self.assertRegex(self.manifest["version"], r"^\d+\.\d+\.\d+-beta\.\d+$")
        self.assertEqual(sorted(self.manifest["required_secret_names"]), ["HF_SPACE", "HF_TOKEN"])
        text = json.dumps(self.manifest).lower()
        self.assertNotIn("hf_", text)
        self.assertGreaterEqual(len(self.manifest["release_notes"]), 4)

    def test_gate_fails_closed_without_runtime_evidence(self) -> None:
        evidence = self.manifest["evidence"]
        required = {
            "mandatory_ci_green",
            "user_acceptance_complete",
            "deploy_revision_equal",
            "live_health_green",
            "release_notes_present",
            "rollback_qualified",
        }
        self.assertEqual(set(evidence), required)
        approved = all(evidence.values()) and self.manifest["decision"] == "approved"
        self.assertFalse(approved)

    def test_revision_and_rollback_contract(self) -> None:
        candidate = self.manifest["candidate_sha"]
        deployed = self.manifest["deployed_sha"]
        rollback = self.manifest["rollback_sha"]
        self.assertRegex(candidate, SHA_RE)
        self.assertRegex(deployed, SHA_RE)
        self.assertRegex(rollback, SHA_RE)
        self.assertNotEqual(rollback, candidate)

    def test_roadmap_is_complete_prefix(self) -> None:
        phases = self.roadmap["phases"]
        self.assertEqual(self.roadmap["phase_count"], 4)
        self.assertEqual([p["id"] for p in phases], ["RQ-1", "RQ-2", "RQ-3", "RQ-4"])
        self.assertEqual([p["status"] for p in phases], ["implemented"] * 4)
        self.assertTrue(Path(phases[3]["evidence"]).is_file())
        self.assertEqual(len(phases[3]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
