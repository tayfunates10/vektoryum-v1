from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from engine.regression.rfv3_timeout_diagnostic import DiagnosticTracker, _svg_metadata


class TimeoutDiagnosticTest(unittest.TestCase):
    def test_svg_metadata_contains_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            svg = Path(directory) / "sample.svg"
            svg.write_text('<svg><path d="M0 0L1 1"/><path d="M1 1L2 2"/></svg>', encoding="utf-8")
            metadata = _svg_metadata(svg)
            self.assertEqual(metadata["path_count"], 2)
            self.assertEqual(metadata["svg_bytes"], svg.stat().st_size)
            self.assertRegex(metadata["svg_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("svg_content", metadata)

    def test_profile_records_root_stage_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def classifier(frame):
                if frame.f_code.co_name == "measured_call":
                    return "unit_stage", "measured_call"
                return None, None

            tracker = DiagnosticTracker(
                output_dir=Path(directory),
                case_id="qualification-public-test",
                source_sha256="a" * 64,
                engine_version="b" * 40,
                heartbeat_seconds=60.0,
                near_timeout_seconds=3000.0,
                classifier=classifier,
            )

            def measured_call() -> None:
                time.sleep(0.01)

            tracker.start()
            sys.setprofile(tracker.profile)
            measured_call()
            sys.setprofile(None)
            tracker.finish(status="success")

            payload = json.loads((Path(directory) / "diagnostic.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["stages"]["unit_stage"]["call_count"], 1)
            self.assertGreater(payload["stages"]["unit_stage"]["total_seconds"], 0.0)
            self.assertFalse(payload["raw_assets_in_artifact"])
            self.assertFalse(payload["production_behavior_changed"])

    def test_near_timeout_snapshot_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = DiagnosticTracker(
                output_dir=Path(directory),
                case_id="qualification-public-test",
                source_sha256="c" * 64,
                engine_version="d" * 40,
            )
            tracker.write_snapshot("near-timeout.json")
            payload = json.loads((Path(directory) / "near-timeout.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "vektoryum-rfv3d3-timeout-diagnostic-v1")
            self.assertEqual(payload["last_events"], [])


if __name__ == "__main__":
    unittest.main()
