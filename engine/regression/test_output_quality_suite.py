from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from engine.regression.output_quality_suite import (
    SCHEMA,
    build_report,
    difference_map,
    generate_cases,
    should_fail,
)


class OutputQualitySuiteContractTests(unittest.TestCase):
    def test_generated_case_pack_is_complete_deterministic_and_binary_free(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = generate_cases(Path(first_tmp))
            second = generate_cases(Path(second_tmp))

            expected_ids = [
                "qa-gray-border-counter",
                "qa-shared-boundary",
                "qa-ring-holes",
                "qa-monoline",
                "qa-small-details",
                "qa-transparent-overlap",
                "qa-lowres-badge",
            ]
            self.assertEqual([item.case_id for item in first], expected_ids)
            self.assertEqual([item.case_id for item in second], expected_ids)
            self.assertEqual(
                [item.source_sha256 for item in first],
                [item.source_sha256 for item in second],
            )
            self.assertEqual(len({item.source_sha256 for item in first}), len(first))
            for item in first:
                self.assertTrue(item.source_path.is_file())
                with Image.open(item.source_path) as image:
                    self.assertEqual(image.size, (256, 256))
                    self.assertEqual(image.mode, "RGBA")

            transparent = next(item for item in first if item.case_id == "qa-transparent-overlap")
            self.assertEqual(transparent.required_metrics, frozenset({"alpha_fidelity"}))

    def test_difference_map_is_deterministic_and_shape_preserving(self) -> None:
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        same = difference_map(source, source.copy())
        changed_input = source.copy()
        changed_input[2:8, 4:10] = 255
        changed = difference_map(source, changed_input)

        self.assertEqual(same.shape, source.shape)
        self.assertEqual(changed.shape, source.shape)
        self.assertTrue(np.array_equal(same, difference_map(source, source.copy())))
        self.assertFalse(np.array_equal(same, changed))

    def test_report_orders_severity_and_preserves_failure_codes(self) -> None:
        results = [
            {
                "case_id": "medium",
                "severity": "medium",
                "structural_failures": [],
                "hard_fail_codes": [],
                "selected_svg_present": True,
                "deterministic": True,
            },
            {
                "case_id": "critical",
                "severity": "critical",
                "structural_failures": ["render_failed"],
                "hard_fail_codes": ["render_failed"],
                "selected_svg_present": False,
                "deterministic": False,
            },
            {
                "case_id": "pass",
                "severity": "pass",
                "structural_failures": [],
                "hard_fail_codes": [],
                "selected_svg_present": True,
                "deterministic": True,
            },
        ]
        report = build_report(
            results,
            engine_version="deadbeef",
            repeat_count=2,
            fail_on="structural",
        )

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual([item["case_id"] for item in report["cases"]], ["critical", "medium", "pass"])
        self.assertEqual(report["severity_counts"]["critical"], 1)
        self.assertEqual(report["severity_counts"]["medium"], 1)
        self.assertEqual(report["severity_counts"]["pass"], 1)
        self.assertEqual(report["structural_failure_codes"], ["render_failed"])
        self.assertEqual(report["hard_failure_codes"], ["render_failed"])
        self.assertFalse(report["all_outputs_present"])
        self.assertFalse(report["all_deterministic"])

    def test_gate_modes_are_explicit(self) -> None:
        report = {
            "structural_failure_codes": ["render_failed"],
            "hard_failure_codes": ["render_failed", "ssim_below_min"],
        }
        self.assertFalse(should_fail(report, "none"))
        self.assertTrue(should_fail(report, "structural"))
        self.assertTrue(should_fail(report, "hard"))

        visual_only = {
            "structural_failure_codes": [],
            "hard_failure_codes": ["ssim_below_min"],
        }
        self.assertFalse(should_fail(visual_only, "structural"))
        self.assertTrue(should_fail(visual_only, "hard"))

        with self.assertRaises(ValueError):
            should_fail(report, "unknown")


if __name__ == "__main__":
    unittest.main()
