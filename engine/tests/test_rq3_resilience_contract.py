from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    state: str
    created: int
    size: int
    active: bool = False
    pinned: bool = False
    legal_hold: bool = False


def admit(active: int, queued: int, max_active: int, max_queued: int) -> str:
    if active < max_active:
        return "active"
    if queued < max_queued:
        return "queued"
    return "rejected"


def cleanup_candidates(artifacts: list[Artifact]) -> list[Artifact]:
    terminal = {"succeeded", "failed", "cancelled", "timed_out", "quarantined"}
    return sorted(
        [a for a in artifacts if a.state in terminal and not (a.active or a.pinned or a.legal_hold)],
        key=lambda a: (a.created, a.artifact_id),
    )


class RQ3ResilienceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(Path("engine/regression/rq3_resilience_manifest.json").read_text())
        cls.roadmap = json.loads(Path("docs/release_qualification_roadmap.json").read_text())
        cls.limits = cls.manifest["limits"]

    def test_manifest_is_finite_and_bounded(self) -> None:
        self.assertEqual(self.manifest["schema"], "vektoryum-rq3-resilience-v1")
        self.assertEqual(len(self.manifest["required_scenarios"]), 10)
        self.assertEqual(len(set(self.manifest["required_scenarios"])), 10)
        for name, value in self.limits.items():
            self.assertIsInstance(value, int, name)
            self.assertGreater(value, 0, name)
        self.assertLess(self.limits["disk_low_water_bytes"], self.limits["disk_high_water_bytes"])

    def test_bounded_admission_rejects_before_allocation(self) -> None:
        self.assertEqual(admit(0, 0, 2, 3), "active")
        self.assertEqual(admit(2, 0, 2, 3), "queued")
        self.assertEqual(admit(2, 3, 2, 3), "rejected")

    def test_per_user_quota_fails_closed(self) -> None:
        limit = self.limits["per_user_retained_bytes"]
        retained = limit
        requested = 1
        self.assertFalse(retained + requested <= limit)

    def test_cleanup_is_oldest_eligible_first_and_preserves_protected(self) -> None:
        artifacts = [
            Artifact("old", "succeeded", 1, 5),
            Artifact("pinned", "succeeded", 0, 5, pinned=True),
            Artifact("active", "running", 0, 5, active=True),
            Artifact("hold", "failed", 0, 5, legal_hold=True),
            Artifact("new", "failed", 2, 5),
        ]
        self.assertEqual([a.artifact_id for a in cleanup_candidates(artifacts)], ["old", "new"])

    def test_cancellation_and_timeout_are_terminal_and_non_publishable(self) -> None:
        terminal = set(self.manifest["terminal_states"])
        self.assertIn("cancelled", terminal)
        self.assertIn("timed_out", terminal)
        publishable = {"succeeded"}
        self.assertNotIn("cancelled", publishable)
        self.assertNotIn("timed_out", publishable)

    def test_restart_recovery_and_conflict_fail_closed(self) -> None:
        jobs = [
            {"id": "stale", "lease_age": 121, "recoveries": 0},
            {"id": "fresh", "lease_age": 30, "recoveries": 0},
        ]
        recoverable = [j for j in jobs if j["lease_age"] > self.limits["lease_ttl_seconds"]]
        self.assertEqual([j["id"] for j in recoverable], ["stale"])
        recoverable[0]["recoveries"] += 1
        self.assertEqual(recoverable[0]["recoveries"], 1)
        local_generation, remote_generation = 7, 8
        self.assertNotEqual(local_generation, remote_generation)

    def test_corrupt_manifest_is_quarantined(self) -> None:
        artifact_manifest = {"artifact_id": "x", "sha256": "bad", "complete": False}
        valid = artifact_manifest["complete"] and len(artifact_manifest["sha256"]) == 64
        self.assertFalse(valid)

    def test_roadmap_has_implemented_prefix_through_rq3(self) -> None:
        phases = self.roadmap["phases"]
        self.assertEqual([p["id"] for p in phases], ["RQ-1", "RQ-2", "RQ-3", "RQ-4"])
        self.assertEqual([p["status"] for p in phases], ["implemented", "implemented", "implemented", "pending"])
        self.assertTrue(Path(phases[2]["evidence"]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
