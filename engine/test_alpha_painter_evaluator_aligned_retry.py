"""Task B — evaluator-grid terminal painter retry contracts."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.alpha_candidate_evaluator_retry import (
    alpha_only_evaluator_retry_labels,
    eligible_retry_specs,
    evaluator_aligned_source_rgba,
)


class EvaluatorAlignedRetryTests(unittest.TestCase):
    def test_source_alpha_matches_evaluator_inter_area_grid(self) -> None:
        height, width = 1200, 2400
        source = np.zeros((height, width, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]
        source[:, :, 0] = (xx % 251).astype(np.uint8)
        source[:, :, 1] = (yy % 239).astype(np.uint8)
        source[:, :, 2] = ((xx + yy) % 233).astype(np.uint8)
        source[:, :, 3] = ((3 * xx + 5 * yy) % 256).astype(np.uint8)

        aligned = evaluator_aligned_source_rgba(source)

        self.assertEqual(aligned.shape, (512, 1024, 4))
        expected_alpha = cv2.resize(
            source[:, :, 3],
            (1024, 512),
            interpolation=cv2.INTER_AREA,
        )
        np.testing.assert_array_equal(aligned[:, :, 3], expected_alpha)

    def test_small_source_is_not_upscaled_and_result_does_not_alias(self) -> None:
        source = np.zeros((200, 320, 4), dtype=np.uint8)
        source[:, :, 3] = np.arange(320, dtype=np.uint8)[None, :]

        aligned = evaluator_aligned_source_rgba(source)

        self.assertEqual(aligned.shape, source.shape)
        np.testing.assert_array_equal(aligned, source)
        aligned[0, 0, 3] ^= 0xFF
        self.assertNotEqual(int(aligned[0, 0, 3]), int(source[0, 0, 3]))

    def test_public04_alpha_only_ladder_rejections_are_replayed_in_legacy_order(self) -> None:
        attempts = []
        for label in (
            "paint-deficit-cumulative",
            "paint-deficit-cumulative-q20",
            "paint-deficit-cumulative-q16",
            "paint-deficit-cumulative-q12",
            "paint-deficit-cumulative-q8",
        ):
            attempts.append(
                {
                    "encoding_label": label,
                    "status": "evaluator_rejected",
                    "validation_stage": "evaluator",
                    "exact_error_code": (
                        "source_alpha_candidate_painter_evaluator_rejected:"
                        "alpha_iou_below_min,alpha_mae_above_max"
                    ),
                }
            )

        self.assertEqual(
            alpha_only_evaluator_retry_labels(attempts),
            {
                "paint-deficit-cumulative",
                "paint-deficit-cumulative-q20",
                "paint-deficit-cumulative-q16",
                "paint-deficit-cumulative-q12",
                "paint-deficit-cumulative-q8",
            },
        )
        self.assertEqual(
            eligible_retry_specs(attempts),
            (
                ("paint-deficit-cumulative", "cumulative", 24),
                ("paint-deficit-cumulative-q20", "cumulative", 20),
                ("paint-deficit-cumulative-q16", "cumulative", 16),
                ("paint-deficit-cumulative-q12", "cumulative", 12),
                ("paint-deficit-cumulative-q8", "cumulative", 8),
            ),
        )

    def test_public15_byte_budget_failures_do_not_trigger_retry(self) -> None:
        attempts = [
            {
                "encoding_label": "paint-deficit-q24",
                "status": "byte_rejected",
                "validation_stage": None,
                "exact_error_code": (
                    "source_alpha_candidate_painter_byte_budget_rejected:"
                    "paint-deficit-q24:1194110>304990"
                ),
            },
            {
                "encoding_label": "paint-deficit-cumulative-q8",
                "status": "byte_rejected",
                "validation_stage": None,
                "exact_error_code": (
                    "source_alpha_candidate_painter_byte_budget_rejected:"
                    "paint-deficit-cumulative-q8:717866>304990"
                ),
            },
        ]
        self.assertEqual(alpha_only_evaluator_retry_labels(attempts), set())
        self.assertEqual(eligible_retry_specs(attempts), ())

    def test_public05_native_alpha_failure_does_not_trigger_retry(self) -> None:
        attempts = [
            {
                "encoding_label": "paint-deficit-cumulative-q20",
                "status": "native_alpha_rejected",
                "validation_stage": "native_alpha",
                "exact_error_code": (
                    "source_alpha_candidate_painter_native_iou_gate_failed:"
                    "0.427402<0.995"
                ),
            }
        ]
        self.assertEqual(eligible_retry_specs(attempts), ())

    def test_mixed_or_unmeasured_evaluator_failure_is_not_retry_eligible(self) -> None:
        attempts = [
            {
                "encoding_label": "paint-deficit-cumulative-q20",
                "status": "evaluator_rejected",
                "validation_stage": "evaluator",
                "exact_error_code": (
                    "source_alpha_candidate_painter_evaluator_rejected:"
                    "alpha_iou_below_min,unexpected_code"
                ),
            },
            {
                "encoding_label": "paint-deficit-cumulative-q16",
                "status": "evaluator_rejected",
                "validation_stage": "evaluator",
                "exact_error_code": (
                    "source_alpha_candidate_painter_evaluator_rejected:"
                    "alpha_plane_unmeasured"
                ),
            },
        ]
        self.assertEqual(eligible_retry_specs(attempts), ())


if __name__ == "__main__":
    unittest.main()
