from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.alpha_candidate_validation import validate_alpha_reconstruction_contract


class AlphaCandidateValidationTests(unittest.TestCase):
    @staticmethod
    def _source() -> np.ndarray:
        rgba = np.zeros((8, 12, 4), dtype=np.uint8)
        rgba[2:7, 3:10, :3] = 32
        rgba[2:7, 3:10, 3] = 255
        return rgba

    @staticmethod
    def _structure(*_args, **_kwargs):
        return (
            {"path_count": 2, "node_count": 8},
            [],
            [],
            object(),
        )

    def test_composite_appearance_codes_are_delegated_to_transform_journal(self) -> None:
        source = self._source()
        evaluator_report = SimpleNamespace(
            metrics={
                "G_gradient_alpha": {
                    "alpha_iou": 1.0,
                    "alpha_mae": 0.0,
                    "alpha_fidelity_status": "failed",
                }
            },
            hard_fail_codes=[
                "alpha_white_ssim_below_min",
                "alpha_white_mae_above_max",
                "alpha_black_ssim_below_min",
                "alpha_checker_mae_above_max",
            ],
            verdict="failed",
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.svg"
            candidate.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            with (
                patch(
                    "app.alpha_candidate_validation.render_svg_to_rgba",
                    return_value=source.copy(),
                ),
                patch(
                    "app.final_artifact_evaluator.evaluate_final_svg",
                    return_value=evaluator_report,
                ),
                patch(
                    "app.final_artifact_evaluator._structure_check",
                    side_effect=self._structure,
                ),
            ):
                report = validate_alpha_reconstruction_contract(
                    candidate,
                    source,
                    "logo_color",
                    (2, 8),
                )

        self.assertEqual(report["final_evaluator_alpha_plane_status"], "passed")
        self.assertEqual(report["final_evaluator_alpha_iou"], 1.0)
        self.assertEqual(report["final_evaluator_alpha_mae"], 0.0)
        self.assertEqual(
            report["appearance_regression_authority"],
            "transform_journal_parent_delta",
        )
        self.assertIn(
            "alpha_white_ssim_below_min",
            report["final_evaluator_alpha_appearance_codes"],
        )

    def test_evaluator_alpha_plane_failure_remains_fail_closed(self) -> None:
        source = self._source()
        evaluator_report = SimpleNamespace(
            metrics={
                "G_gradient_alpha": {
                    "alpha_iou": 0.99,
                    "alpha_mae": 0.001,
                    "alpha_fidelity_status": "failed",
                }
            },
            hard_fail_codes=["alpha_iou_below_min"],
            verdict="failed",
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.svg"
            candidate.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            with (
                patch(
                    "app.alpha_candidate_validation.render_svg_to_rgba",
                    return_value=source.copy(),
                ),
                patch(
                    "app.final_artifact_evaluator.evaluate_final_svg",
                    return_value=evaluator_report,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "evaluator_rejected:alpha_iou_below_min",
                ):
                    validate_alpha_reconstruction_contract(
                        candidate,
                        source,
                        "logo_color",
                        (2, 8),
                    )

    def test_direct_alpha_plane_failure_cannot_reach_evaluator(self) -> None:
        source = self._source()
        transparent = source.copy()
        transparent[:, :, 3] = 0
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.svg"
            candidate.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            with (
                patch(
                    "app.alpha_candidate_validation.render_svg_to_rgba",
                    return_value=transparent,
                ),
                patch(
                    "app.final_artifact_evaluator.evaluate_final_svg"
                ) as evaluator,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "candidate_knockout_iou_gate_failed",
                ):
                    validate_alpha_reconstruction_contract(
                        candidate,
                        source,
                        "logo_color",
                        (2, 8),
                    )
                evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
