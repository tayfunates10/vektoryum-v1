from __future__ import annotations

import unittest

from engine.regression.output_quality_root_cause import (
    candidate_snapshot,
    infer_root_causes,
    pipeline_snapshot,
)


class OutputQualityRootCauseContractTests(unittest.TestCase):
    def test_pipeline_snapshot_keeps_only_decision_evidence(self) -> None:
        output = {
            "mode_used": "logo_color",
            "mode_warning": None,
            "selection_reason": "highest_fidelity",
            "analysis": {
                "recommended_mode": "logo_color",
                "estimated_color_count": 4,
                "likely_color_logo": True,
                "private_blob": {"must": "not leak"},
            },
            "preprocess_report": {
                "auto_color_count": 4,
                "actual_color_count": 5,
                "absolute_path": "/tmp/private.png",
            },
            "best": {
                "name": "logo_gradient",
                "engine": "gradient",
                "rendered_ok": True,
                "fidelity_score": 98.125678901,
                "total_score": 91.0,
                "svg_path": "/tmp/private.svg",
                "score_details": {
                    "path_count": 2,
                    "edge_f1": 0.92,
                    "secret": "omit",
                },
            },
            "raw_best": None,
            "scored": [
                {
                    "name": "lower",
                    "engine": "vtracer",
                    "rendered_ok": True,
                    "fidelity_score": 90.0,
                    "total_score": 99.0,
                    "score_details": {"path_count": 9},
                },
                {
                    "name": "logo_gradient",
                    "engine": "gradient",
                    "rendered_ok": True,
                    "fidelity_score": 98.125678901,
                    "total_score": 91.0,
                    "score_details": {"path_count": 2, "edge_f1": 0.92},
                },
            ],
        }

        snapshot = pipeline_snapshot(output)

        self.assertEqual(snapshot["capture_status"], "ok")
        self.assertEqual(snapshot["mode_used"], "logo_color")
        self.assertEqual(snapshot["analysis"]["estimated_color_count"], 4)
        self.assertNotIn("private_blob", snapshot["analysis"])
        self.assertNotIn("absolute_path", snapshot["preprocess"])
        self.assertNotIn("svg_path", snapshot["best"])
        self.assertEqual(snapshot["best"]["fidelity_score"], 98.1256789)
        self.assertEqual([item["name"] for item in snapshot["candidates"]], ["logo_gradient", "lower"])

    def test_candidate_snapshot_rejects_non_mapping(self) -> None:
        self.assertIsNone(candidate_snapshot(None))
        self.assertIsNone(candidate_snapshot("candidate"))

    def test_inference_identifies_single_color_collapse(self) -> None:
        result = {
            "severity": "high",
            "metrics": {
                "component_delta": 21,
                "seam_ratio": 0.36,
                "ssim": 0.56,
                "edge_f1_1px": 0.28,
                "de00_p95": 26.0,
                "min_component_iou": 0.002,
                "alpha_iou": None,
            },
            "pipeline_diagnostics": [
                {"mode_used": "single_color", "best": {"engine": "potrace", "name": "single_contour"}}
            ],
        }

        codes = {item["code"] for item in infer_root_causes(result)}

        self.assertIn("auto_mode_color_collapse", codes)
        self.assertIn("missing_or_collapsed_region", codes)

    def test_inference_separates_metric_sensitivity_from_geometry_loss(self) -> None:
        result = {
            "severity": "high",
            "metrics": {
                "component_delta": 0,
                "seam_ratio": 0.20,
                "ssim": 0.74,
                "edge_f1_1px": 1.0,
                "de00_p95": 5.3,
                "min_component_iou": 1.0,
                "alpha_iou": None,
            },
            "pipeline_diagnostics": [
                {"mode_used": "geometric_logo", "best": {"engine": "vtracer", "name": "geo_clean"}}
            ],
        }

        codes = {item["code"] for item in infer_root_causes(result)}

        self.assertIn("seam_metric_light_region_sensitivity", codes)
        self.assertIn("ssim_tone_canonicalization_sensitivity", codes)
        self.assertNotIn("missing_or_collapsed_region", codes)

    def test_inference_flags_alpha_preserved_gradient_boundary_drift(self) -> None:
        result = {
            "severity": "medium",
            "metrics": {
                "component_delta": 0,
                "seam_ratio": 0.002,
                "ssim": 0.987,
                "edge_f1_1px": 0.917,
                "de00_p95": 10.14,
                "min_component_iou": 0.98,
                "alpha_iou": 0.997,
            },
            "pipeline_diagnostics": [
                {"mode_used": "logo_color", "best": {"engine": "gradient", "name": "logo_gradient"}}
            ],
        }

        codes = {item["code"] for item in infer_root_causes(result)}

        self.assertIn("alpha_preserved_color_boundary_drift", codes)
        self.assertIn("gradient_candidate_color_transition_error", codes)


if __name__ == "__main__":
    unittest.main()
