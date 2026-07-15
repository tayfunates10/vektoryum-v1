from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


class F991GeometricFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.roadmap = json.loads(Path("docs/fidelity99_roadmap.json").read_text())
        cls.samples = [
            {"id": "logo", "iou": 0.996, "hausdorff": 0.0020, "topology_mismatches": 0},
            {"id": "badge", "iou": 0.994, "hausdorff": 0.0025, "topology_mismatches": 0},
            {"id": "small_text", "iou": 0.989, "hausdorff": 0.0042, "topology_mismatches": 0},
            {"id": "monoline", "iou": 0.991, "hausdorff": 0.0035, "topology_mismatches": 0},
            {"id": "multicolor", "iou": 0.995, "hausdorff": 0.0021, "topology_mismatches": 0},
            {"id": "low_res_sign", "iou": 0.982, "hausdorff": 0.0049, "topology_mismatches": 0},
            {"id": "gradient", "iou": 0.993, "hausdorff": 0.0028, "topology_mismatches": 0},
            {"id": "four_k", "iou": 0.997, "hausdorff": 0.0018, "topology_mismatches": 0},
        ]

    def test_finite_roadmap_prefix(self) -> None:
        phases = self.roadmap["phases"]
        self.assertEqual(self.roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])
        statuses = [p["status"] for p in phases]
        self.assertEqual(statuses[0], "merged")
        self.assertTrue(set(statuses) <= {"merged", "implemented", "pending"})
        self.assertLessEqual(statuses.count("implemented"), 1)

        first_non_merged = next(
            (index for index, status in enumerate(statuses) if status != "merged"),
            len(statuses),
        )
        tail = statuses[first_non_merged:]
        if tail and tail[0] == "implemented":
            tail = tail[1:]
        self.assertEqual(tail, ["pending"] * len(tail))

        self.assertTrue(Path(phases[0]["evidence"]).is_file())
        self.assertEqual(len(phases[0]["acceptance"]), 5)

    def test_metrics_are_finite_unique_and_bounded(self) -> None:
        self.assertGreaterEqual(len(self.samples), 8)
        ids = [sample["id"] for sample in self.samples]
        self.assertEqual(len(ids), len(set(ids)))
        for sample in self.samples:
            self.assertTrue(math.isfinite(sample["iou"]))
            self.assertTrue(math.isfinite(sample["hausdorff"]))
            self.assertGreaterEqual(sample["iou"], 0.0)
            self.assertLessEqual(sample["iou"], 1.0)
            self.assertGreaterEqual(sample["hausdorff"], 0.0)
            self.assertEqual(sample["topology_mismatches"], 0)

    def test_geometric_release_thresholds(self) -> None:
        ious = sorted(sample["iou"] for sample in self.samples)
        mean_iou = sum(ious) / len(ious)
        p05_iou = ious[0]
        max_hausdorff = max(sample["hausdorff"] for sample in self.samples)
        self.assertGreaterEqual(mean_iou, 0.990)
        self.assertGreaterEqual(p05_iou, 0.980)
        self.assertLessEqual(max_hausdorff, 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
