from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import engine.regression.output_quality_suite as suite
from engine.regression.output_quality_suite import (
    SCHEMA,
    build_report,
    difference_map,
    generate_cases,
    run_diagnostic_case,
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

    def test_production_contract_creates_job_dir_and_emits_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as fixture_tmp, tempfile.TemporaryDirectory() as output_tmp:
            case = generate_cases(Path(fixture_tmp))[0]
            output_root = Path(output_tmp)

            def fake_pipeline(image, original_path, trace_mode, job_dir):
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(Path(original_path), case.source_path)
                self.assertEqual(trace_mode, "auto")
                self.assertTrue(Path(job_dir).is_dir())
                selected = Path(job_dir) / "selected.svg"
                selected.write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
                    '<rect width="256" height="256" fill="#ffffff"/></svg>',
                    encoding="utf-8",
                )
                return {"best": {"svg_path": str(selected)}}

            evaluator = SimpleNamespace(
                verdict="production_ready",
                hard_fail_codes=[],
                soft_warning_codes=[],
                unmeasured_required=[],
                metrics={
                    "B_visual": {"ssim": 1.0, "ms_ssim": 1.0},
                    "C_color": {"de00_mean": 0.0, "de00_p95": 0.0, "palette_agree": 1.0},
                    "D_edge_geometry": {
                        "edge_f1_1px": 1.0,
                        "edge_f1_2px": 1.0,
                        "chamfer_p95": 0.0,
                        "hausdorff_max": 0.0,
                    },
                    "E_topology": {"component_delta": 0, "hole_delta": 0},
                    "F_small_detail": {"min_component_iou": 1.0, "mean_component_iou": 1.0},
                    "G_gradient_alpha": {"seam_ratio": 0.0, "alpha_iou": 1.0, "alpha_mae": 0.0},
                    "H_editability": {"path_count": 1, "node_count": 1, "nodes_per_path": 1.0},
                },
            )
            rendered = np.full((256, 256, 4), 255, dtype=np.uint8)

            with (
                patch.object(suite, "run_pipeline", side_effect=fake_pipeline),
                patch.object(suite, "evaluate_final_svg", return_value=evaluator),
                patch.object(suite, "render_svg_to_rgba", return_value=rendered),
            ):
                result = run_diagnostic_case(
                    case,
                    output_root=output_root,
                    engine_version="test-sha",
                    repeat_count=1,
                )

            self.assertEqual(result["pipeline_status"], "ok")
            self.assertTrue(result["selected_svg_present"])
            self.assertEqual(result["render_status"], "ok")
            self.assertEqual(result["severity"], "pass")
            self.assertEqual(result["structural_failures"], [])
            self.assertTrue(result["deterministic"])
            self.assertTrue((output_root / "cases" / case.case_id / "selected.svg").is_file())
            self.assertTrue((output_root / "cases" / case.case_id / "render.png").is_file())
            self.assertTrue((output_root / "cases" / case.case_id / "difference.png").is_file())

    def test_pipeline_failure_is_always_critical(self) -> None:
        with tempfile.TemporaryDirectory() as fixture_tmp, tempfile.TemporaryDirectory() as output_tmp:
            case = generate_cases(Path(fixture_tmp))[0]

            with patch.object(suite, "run_pipeline", side_effect=RuntimeError("synthetic failure")):
                result = run_diagnostic_case(
                    case,
                    output_root=Path(output_tmp),
                    engine_version="test-sha",
                    repeat_count=1,
                )

            self.assertEqual(result["pipeline_status"], "failed")
            self.assertEqual(result["severity"], "critical")
            self.assertIn("pipeline_exception", result["structural_failures"])
            self.assertIn("selected_svg_file_missing", result["structural_failures"])
            self.assertIn("render_failed", result["structural_failures"])


if __name__ == "__main__":
    unittest.main()
