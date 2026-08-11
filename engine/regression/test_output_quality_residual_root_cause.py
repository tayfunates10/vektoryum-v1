from __future__ import annotations

import unittest

from engine.regression.output_quality_residual_root_cause import infer_root_causes


class ResidualRootCauseTests(unittest.TestCase):
    @staticmethod
    def _codes(case: dict) -> set[str]:
        return {item["code"] for item in infer_root_causes(case)}

    def test_component_and_micro_loss_are_reported_from_true_components(self) -> None:
        case = {
            "severity": "high",
            "metrics": {"min_component_iou": 0.99},
            "residual": {
                "source_component_recall": 0.6666667,
                "small_component_recall": 0.6666667,
                "min_component_iou": 0.0,
                "render_component_precision": 1.0,
                "visible_error_ratio": 0.0042,
            },
            "multi_scale": {},
            "pipeline_diagnostics": [],
        }
        codes = self._codes(case)
        self.assertIn("connected_component_loss", codes)
        self.assertIn("micro_detail_loss", codes)
        self.assertIn("aggregate_component_metric_blind_spot", codes)

    def test_spurious_components_boundary_and_multiscale_instability_are_reported(self) -> None:
        case = {
            "severity": "high",
            "metrics": {},
            "residual": {
                "source_component_recall": 1.0,
                "small_component_recall": 1.0,
                "min_component_iou": 0.86,
                "render_component_precision": 0.75,
                "boundary_p95_px": 1.0,
                "edge_error_ratio": 0.01,
                "interior_error_ratio": 0.08,
                "visible_error_ratio": 0.03,
            },
            "multi_scale": {
                "min_source_component_recall": 0.8,
                "min_small_component_recall": 0.6,
                "max_normalized_boundary_excess_px": 1.25,
            },
            "pipeline_diagnostics": [],
        }
        codes = self._codes(case)
        self.assertIn("spurious_render_component", codes)
        self.assertIn("color_boundary_drift", codes)
        self.assertIn("interior_tone_residual", codes)
        self.assertIn("multi_scale_detail_instability", codes)
        self.assertIn("multi_scale_boundary_instability", codes)

    def test_alpha_mask_can_pass_while_visible_composite_fails(self) -> None:
        case = {
            "severity": "medium",
            "metrics": {},
            "residual": {
                "source_has_alpha": True,
                "alpha": {"alpha_iou": 0.997672, "alpha_mae": 0.000781},
                "de00_p95": 10.106,
                "visible_error_ratio": 0.10806,
            },
            "multi_scale": {},
            "pipeline_diagnostics": [],
        }
        self.assertIn("alpha_mask_pass_visible_composite_fail", self._codes(case))

    def test_alpha_budget_exception_is_classified_as_fallback_exhaustion(self) -> None:
        case = {
            "severity": "critical",
            "pipeline_error": (
                "RuntimeError: source_alpha_candidate_knockout_byte_budget_rejected:"
                "463175>259130"
            ),
            "metrics": {},
            "residual": {},
            "multi_scale": {},
            "pipeline_diagnostics": [],
        }
        self.assertIn("alpha_fallback_exhausted", self._codes(case))

    def test_pareto_dominated_winner_is_detected_without_complexity_increase(self) -> None:
        case = {
            "severity": "high",
            "metrics": {},
            "residual": {},
            "multi_scale": {},
            "pipeline_diagnostics": [
                {
                    "best": {
                        "name": "logo_gradient",
                        "fidelity_score": 94.18,
                        "total_score": 68.41,
                        "score_details": {"path_count": 3, "node_count": 51},
                    },
                    "candidates": [
                        {
                            "name": "logo_gradient_bnd",
                            "fidelity_score": 97.63,
                            "total_score": 69.45,
                            "score_details": {"path_count": 3, "node_count": 51},
                        }
                    ],
                }
            ],
        }
        self.assertIn("pareto_dominated_winner", self._codes(case))


if __name__ == "__main__":
    unittest.main()
