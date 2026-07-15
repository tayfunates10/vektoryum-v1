from __future__ import annotations

import json
import unittest
from pathlib import Path


class ProductionContainerAssetTests(unittest.TestCase):
    def test_analyzer_calibration_is_packaged_at_runtime_path(self) -> None:
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        copy_line = "COPY engine/analyzer_calibration_v1.json ./analyzer_calibration_v1.json"
        self.assertIn(copy_line, dockerfile)

        calibration_path = Path("engine/analyzer_calibration_v1.json")
        self.assertTrue(calibration_path.is_file())
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "analyzer-calibration-evidence-v1")
        self.assertEqual(payload["support_model_version"], "analyzer-mode-support-v1")
        self.assertGreater(len(payload.get("cases", [])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
