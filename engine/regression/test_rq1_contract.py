from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


class RQ1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.roadmap = json.loads(Path("docs/release_qualification_roadmap.json").read_text())
        spec = importlib.util.spec_from_file_location("rq1_live_probe", "engine/regression/rq1_live_probe.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load RQ-1 probe module")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.sha = "a" * 40

    def test_finite_roadmap(self) -> None:
        self.assertEqual(self.roadmap["schema"], "vektoryum-release-qualification-v1")
        self.assertEqual(self.roadmap["phase_count"], 4)
        self.assertEqual(self.roadmap["calculation"], "merged_phases / 4")
        phases = self.roadmap["phases"]
        self.assertEqual([phase["id"] for phase in phases], ["RQ-1", "RQ-2", "RQ-3", "RQ-4"])

        statuses = [phase["status"] for phase in phases]
        self.assertTrue(all(status in {"implemented", "pending"} for status in statuses))
        self.assertEqual(statuses[0], "implemented")
        first_pending = next((index for index, status in enumerate(statuses) if status == "pending"), len(statuses))
        self.assertTrue(all(status == "implemented" for status in statuses[:first_pending]))
        self.assertTrue(all(status == "pending" for status in statuses[first_pending:]))

        self.assertEqual(len(phases[0]["acceptance"]), 5)
        self.assertTrue(Path(phases[0]["evidence"]).is_file())

    def test_positive_health_contract(self) -> None:
        payloads = {
            "/livez": {"status": "ok", "check": "liveness", "mode": "beta", "revision": self.sha},
            "/readyz": {"status": "ready", "check": "readiness", "mode": "beta", "revision": self.sha, "reasons": [], "active_requests": 0},
        }
        self.module._get_json = lambda _base, path: payloads[path]
        self.module.verify("https://example.invalid", self.sha)

    def test_revision_mismatch_fails_closed(self) -> None:
        payloads = {
            "/livez": {"status": "ok", "check": "liveness", "mode": "beta", "revision": self.sha},
            "/readyz": {"status": "ready", "check": "readiness", "mode": "beta", "revision": "b" * 40, "reasons": [], "active_requests": 0},
        }
        self.module._get_json = lambda _base, path: payloads[path]
        with self.assertRaisesRegex(RuntimeError, "revision"):
            self.module.verify("https://example.invalid", self.sha)

    def test_https_downgrade_changes_origin(self) -> None:
        self.assertEqual(self.module._origin("https://example.invalid/x"), ("https", "example.invalid", None))
        self.assertNotEqual(
            self.module._origin("https://example.invalid/x"),
            self.module._origin("http://example.invalid/x"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
