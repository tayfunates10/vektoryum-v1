from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.regression import output_quality_residual_suite as suite


RAW_RGBA_SHA256 = {
    "qa-gray-border-counter": "0b0849117b89ff804f002d405e10faf69989f2b72c4ce0a04ae43b10552fb365",
    "qa-shared-boundary": "2a169cf8a837a07b634201c67cc58a867e4bb29d00031c069fd5f47e669266d3",
    "qa-ring-holes": "18340abea2375f2ce4067ed9879025d7dcf74148d23d8917a8d74be530a169d7",
    "qa-monoline": "15996ead6abdffb8c8b056a0a1ab9e3f22c86b07ff2a5304a2657fee16585f2d",
    "qa-small-details": "215f59fedcff79049678717257527a061d104aeec0c2f5cf6d0e6b2c1ceae98a",
    "qa-transparent-overlap": "a5600f3ce08388eb75e3a657ecba9a118524c6ab7936300a247f504a1dd0ac50",
    "qa-lowres-badge": "da5a92de42d5049eb1f3280ee699e67ae4714fa9e7ae1c01e731b63418d77278",
    "qa-micro-component-ladder": "6f794ba753eccb35be4ccbc222ae0dc14e0e8ecd2c71d56d68b7174f6deed378",
    "qa-thin-negative-space": "cd46545ff90c0f39549cb935c65d0fb3e10f158653b228847b069f9b62f5b441",
    "qa-hard-stop-gradient": "610cf3f9a21acf84ac7bbfdba8fb60193e7fc00866f8e77b7e5d1f32542b4334",
    "qa-soft-alpha-shadow": "18f45af7651377c7d0f2a49b4c3098b459b9334ea369eba9dad4e27a12e73a8f",
    "qa-neutral-tone-steps": "6ee86fceb6fb20986eb86c3d1f8a0a3fac78f99a5d462d1123db62457764393c",
}


class ResidualOutputQualitySuiteTests(unittest.TestCase):
    def test_case_pack_is_exactly_twelve_and_raw_rgba_is_immutable(self) -> None:
        expected_ids = [
            "qa-gray-border-counter",
            "qa-shared-boundary",
            "qa-ring-holes",
            "qa-monoline",
            "qa-small-details",
            "qa-transparent-overlap",
            "qa-lowres-badge",
            "qa-micro-component-ladder",
            "qa-thin-negative-space",
            "qa-hard-stop-gradient",
            "qa-soft-alpha-shadow",
            "qa-neutral-tone-steps",
        ]
        self.assertEqual([item[0] for item in suite._CASE_BUILDERS], expected_ids)
        self.assertEqual(set(RAW_RGBA_SHA256), set(expected_ids))

        for case_id, _category, _image_class, builder, _required in suite._CASE_BUILDERS:
            image = builder(256).convert("RGBA")
            self.assertEqual(image.size, (256, 256))
            self.assertEqual(suite.raw_rgba_sha256(image), RAW_RGBA_SHA256[case_id])

    def test_generated_pack_is_repeatable_and_alpha_requirements_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first = suite.generate_cases(Path(first_tmp))
            second = suite.generate_cases(Path(second_tmp))

        self.assertEqual(len(first), 12)
        self.assertEqual([item.case_id for item in first], [item.case_id for item in second])
        self.assertEqual([item.source_sha256 for item in first], [item.source_sha256 for item in second])
        self.assertEqual(len({item.source_sha256 for item in first}), 12)
        alpha_cases = {
            item.case_id
            for item in first
            if "alpha_fidelity" in item.required_metrics
        }
        self.assertEqual(alpha_cases, {"qa-transparent-overlap", "qa-soft-alpha-shadow"})

    def test_near_zero_report_aggregates_independent_blockers(self) -> None:
        passing = {
            "case_id": "pass",
            "severity": "pass",
            "structural_failures": [],
            "hard_fail_codes": [],
            "selected_svg_present": True,
            "deterministic": True,
            "near_zero_measured": True,
            "near_zero": {
                "policy_version": "vektoryum-near-zero-residual-v1",
                "ready": True,
                "blocker_codes": [],
            },
        }
        failing = {
            "case_id": "fail",
            "severity": "high",
            "structural_failures": [],
            "hard_fail_codes": ["ssim_below_min"],
            "selected_svg_present": True,
            "deterministic": True,
            "near_zero_measured": True,
            "near_zero": {
                "policy_version": "vektoryum-near-zero-residual-v1",
                "ready": False,
                "blocker_codes": ["visible_error_ratio", "min_component_iou"],
            },
        }
        report = suite.build_report(
            [passing, failing],
            engine_version="test",
            repeat_count=2,
            fail_on="none",
        )

        self.assertEqual(report["near_zero_pass_count"], 1)
        self.assertEqual(report["near_zero_measured_count"], 2)
        self.assertFalse(report["all_near_zero"])
        self.assertEqual(
            report["near_zero_blocker_histogram"],
            {"min_component_iou": 1, "visible_error_ratio": 1},
        )
        self.assertTrue(suite.should_fail(report, "near-zero"))
        self.assertFalse(suite.should_fail(report, "none"))
        self.assertTrue(suite.should_fail(report, "hard"))

    def test_near_zero_gate_passes_only_when_every_case_is_ready(self) -> None:
        report = {
            "cases": [
                {"near_zero": {"ready": True}},
                {"near_zero": {"ready": True}},
            ]
        }
        self.assertFalse(suite.should_fail(report, "near-zero"))
        report["cases"][1]["near_zero"]["ready"] = False
        self.assertTrue(suite.should_fail(report, "near-zero"))
        self.assertTrue(suite.should_fail({"cases": []}, "near-zero"))

    def test_unmeasured_multiscale_is_explicitly_incomplete(self) -> None:
        value = suite._unmeasured_multiscale()
        self.assertFalse(value["all_scales_measured"])
        self.assertEqual(value["missing_scales"], value["scales_requested"])
        self.assertEqual(len(value["scales_requested"]), 5)
        self.assertEqual(value["levels"], [])


if __name__ == "__main__":
    unittest.main()
