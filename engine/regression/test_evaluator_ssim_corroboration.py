from __future__ import annotations

import unittest

from app.final_artifact_evaluator import (
    FinalArtifactReport,
    apply_geometry_corroborated_ssim,
)


class EvaluatorSsimCorroborationTests(unittest.TestCase):
    @staticmethod
    def _report(*, exact_geometry: bool = True) -> FinalArtifactReport:
        return FinalArtifactReport(
            sha256="a" * 64,
            byte_read_stable=True,
            verdict="failed",
            hard_fails=["SSIM 0.7623 < 0.9897"],
            soft_warnings=[],
            unmeasured_required=[],
            metrics={
                "B_visual": {"ssim": 0.7622949483, "ms_ssim": 0.7911930427},
                "C_color": {
                    "de00_p95": 3.218791008,
                    "palette_agree": 1.0 if exact_geometry else 0.875,
                },
                "D_edge_geometry": {
                    "edge_f1_1px": 1.0,
                    "edge_f1_2px": 1.0,
                    "chamfer_p95": 0.0 if exact_geometry else 70.9,
                    "hausdorff_max": 0.0 if exact_geometry else 84.7,
                },
                "E_topology": {"component_delta": 0, "hole_delta": 0},
                "F_small_detail": {
                    "min_component_iou": 1.0 if exact_geometry else 0.0,
                    "mean_component_iou": 1.0 if exact_geometry else 0.69,
                },
                "G_gradient_alpha": {"seam_ratio": 0.0},
            },
            hard_fail_codes=["ssim_below_min"],
            soft_warning_codes=[],
        )

    def test_exact_geometry_corroborates_canonical_tone_ssim(self) -> None:
        report = self._report(exact_geometry=True)

        returned = apply_geometry_corroborated_ssim(
            report,
            image_class="clean_logo",
        )

        self.assertIs(returned, report)
        self.assertEqual(report.hard_fail_codes, [])
        self.assertEqual(report.hard_fails, [])
        self.assertEqual(report.verdict, "production_ready")
        visual = report.metrics["B_visual"]
        self.assertEqual(
            visual["ssim_gate_status"],
            "geometry_corroborated_canonical_tone",
        )
        self.assertEqual(visual["ssim_corroboration"]["status"], "corroborated")
        self.assertTrue(all(visual["ssim_corroboration"]["checks"].values()))

    def test_missing_palette_class_keeps_legacy_hard_fail(self) -> None:
        report = self._report(exact_geometry=False)

        apply_geometry_corroborated_ssim(
            report,
            image_class="illustration",
        )

        self.assertEqual(report.hard_fail_codes, ["ssim_below_min"])
        self.assertEqual(report.verdict, "failed")
        visual = report.metrics["B_visual"]
        self.assertEqual(visual["ssim_gate_status"], "legacy_hard_fail_retained")
        self.assertEqual(visual["ssim_corroboration"]["status"], "not_corroborated")
        self.assertFalse(
            visual["ssim_corroboration"]["checks"]["component_iou_exact"]
        )

    def test_other_hard_fail_remains_after_ssim_is_corroborated(self) -> None:
        report = self._report(exact_geometry=True)
        report.hard_fail_codes.append("embedded_raster")
        report.hard_fails.append("gömülü raster bitmap içeriyor")

        apply_geometry_corroborated_ssim(
            report,
            image_class="clean_logo",
        )

        self.assertEqual(report.hard_fail_codes, ["embedded_raster"])
        self.assertEqual(report.hard_fails, ["gömülü raster bitmap içeriyor"])
        self.assertEqual(report.verdict, "failed")

    def test_no_legacy_ssim_failure_is_unchanged(self) -> None:
        report = self._report(exact_geometry=True)
        report.hard_fail_codes.clear()
        report.hard_fails.clear()
        report.verdict = "production_ready"

        apply_geometry_corroborated_ssim(
            report,
            image_class="clean_logo",
        )

        self.assertEqual(report.verdict, "production_ready")
        self.assertEqual(
            report.metrics["B_visual"]["ssim_gate_status"],
            "legacy_not_triggered",
        )


if __name__ == "__main__":
    unittest.main()
