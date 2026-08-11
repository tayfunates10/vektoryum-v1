from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import engine.regression.residual_error_metrics as metrics
from engine.regression.residual_error_metrics import (
    blocker_histogram,
    build_near_zero_contract,
    connected_component_fidelity,
    measure_multiscale_svg,
    measure_residual_error,
)


class ResidualErrorMetricTests(unittest.TestCase):
    @staticmethod
    def _solid_rgba(size: int, value: int = 255) -> np.ndarray:
        image = np.full((size, size, 4), value, dtype=np.uint8)
        image[:, :, 3] = 255
        return image

    @staticmethod
    def _two_component_source(size: int = 64) -> np.ndarray:
        image = np.full((size, size, 4), 255, dtype=np.uint8)
        image[:, :, 3] = 255
        image[12:44, 12:44, :3] = 0
        image[52:55, 52:55, :3] = 0
        return image

    def test_identical_rgba_is_near_zero(self) -> None:
        source = self._solid_rgba(48, 236)
        residual = measure_residual_error(source, source.copy())

        self.assertEqual(residual["de00_p95"], 0.0)
        self.assertEqual(residual["visible_error_ratio"], 0.0)
        self.assertEqual(residual["boundary_p95_px"], 0.0)
        self.assertEqual(residual["source_component_recall"], 1.0)
        self.assertEqual(residual["render_component_precision"], 1.0)
        self.assertEqual(residual["small_component_recall"], 1.0)
        self.assertEqual(residual["min_component_iou"], 1.0)

        contract = build_near_zero_contract(
            residual,
            deterministic=True,
            structural_failures=[],
        )
        self.assertTrue(contract["ready"])
        self.assertEqual(contract["blocker_codes"], [])
        self.assertFalse(contract["production_gate"])

    def test_micro_component_loss_is_not_hidden_by_global_error_area(self) -> None:
        source = self._two_component_source()
        rendered = source.copy()
        rendered[52:55, 52:55, :3] = 255

        residual = measure_residual_error(source, rendered)

        # The lost detail is under the 1% global visible-area target, so the
        # independent component axes are what make this fail closed.
        self.assertLess(residual["visible_error_ratio"], 0.01)
        self.assertEqual(residual["source_component_count"], 2)
        self.assertEqual(residual["matched_source_count"], 1)
        self.assertEqual(residual["source_component_recall"], 0.5)
        self.assertEqual(residual["small_component_count"], 1)
        self.assertEqual(residual["small_component_recall"], 0.0)
        self.assertEqual(residual["min_component_iou"], 0.0)

        contract = build_near_zero_contract(
            residual,
            deterministic=True,
            structural_failures=[],
        )
        self.assertFalse(contract["ready"])
        self.assertIn("source_component_recall", contract["blocker_codes"])
        self.assertIn("small_component_recall", contract["blocker_codes"])
        self.assertIn("min_component_iou", contract["blocker_codes"])

    def test_extra_render_component_reduces_precision(self) -> None:
        source = np.zeros((32, 32), dtype=np.uint8)
        rendered = source.copy()
        source[4:14, 4:14] = 1
        rendered[4:14, 4:14] = 1
        rendered[22:26, 22:26] = 1

        result = connected_component_fidelity(
            source,
            rendered,
            min_area=1,
            small_area=32,
        )

        self.assertEqual(result["source_component_recall"], 1.0)
        self.assertEqual(result["render_component_count"], 2)
        self.assertEqual(result["matched_render_count"], 1)
        self.assertEqual(result["render_component_precision"], 0.5)
        self.assertEqual(result["unmatched_render_count"], 1)

    def test_contract_is_fail_closed_for_unmeasured_values(self) -> None:
        contract = build_near_zero_contract(
            None,
            deterministic=None,
            structural_failures=[],
        )

        self.assertFalse(contract["ready"])
        self.assertIn("byte_determinism", contract["blocker_codes"])
        for code in metrics.NEAR_ZERO_TARGETS:
            self.assertIn(code, contract["blocker_codes"])
        checks = {item["code"]: item for item in contract["checks"]}
        self.assertFalse(checks["byte_determinism"]["measured"])
        self.assertFalse(checks["de00_p95"]["measured"])

    def test_alpha_sources_require_alpha_iou_and_mae(self) -> None:
        residual = {
            "de00_p95": 0.0,
            "visible_error_ratio": 0.0,
            "boundary_p95_px": 0.0,
            "source_component_recall": 1.0,
            "render_component_precision": 1.0,
            "small_component_recall": 1.0,
            "min_component_iou": 1.0,
            "source_has_alpha": True,
            "alpha": {"alpha_iou": 0.99, "alpha_mae": 0.01},
        }
        contract = build_near_zero_contract(
            residual,
            deterministic=True,
            structural_failures=[],
        )

        self.assertFalse(contract["ready"])
        self.assertIn("alpha_iou", contract["blocker_codes"])
        self.assertIn("alpha_mae", contract["blocker_codes"])

    def test_multiscale_identical_render_is_complete(self) -> None:
        def source_builder(size: int) -> Image.Image:
            rgba = np.full((size, size, 4), 255, dtype=np.uint8)
            inset = max(2, size // 5)
            rgba[inset:-inset, inset:-inset, :3] = 0
            return Image.fromarray(rgba, mode="RGBA")

        def render(_path, width: int, height: int):
            self.assertEqual(width, height)
            return np.asarray(source_builder(width), dtype=np.uint8).copy()

        with patch.object(metrics, "render_svg_to_rgba", side_effect=render):
            result = measure_multiscale_svg(
                "unused.svg",
                source_builder,
                base_size=64,
                scales=(0.5, 1.0, 2.0),
            )

        self.assertTrue(result["all_scales_measured"])
        self.assertEqual(result["missing_scales"], [])
        self.assertEqual(result["min_source_component_recall"], 1.0)
        self.assertEqual(result["min_small_component_recall"], 1.0)
        self.assertEqual(result["max_normalized_boundary_excess_px"], 0.0)

    def test_multiscale_missing_render_cannot_pass_complete_gate(self) -> None:
        def source_builder(size: int) -> Image.Image:
            return Image.fromarray(self._solid_rgba(size), mode="RGBA")

        def render(_path, width: int, _height: int):
            if width == 32:
                return None
            return np.asarray(source_builder(width), dtype=np.uint8).copy()

        with patch.object(metrics, "render_svg_to_rgba", side_effect=render):
            result = measure_multiscale_svg(
                "unused.svg",
                source_builder,
                base_size=64,
                scales=(0.5, 1.0),
            )

        self.assertFalse(result["all_scales_measured"])
        self.assertEqual(result["missing_scales"], ["0.5x"])
        contract = build_near_zero_contract(
            measure_residual_error(self._solid_rgba(64), self._solid_rgba(64)),
            deterministic=True,
            structural_failures=[],
            multi_scale=result,
        )
        self.assertFalse(contract["ready"])
        self.assertIn("multi_scale_complete", contract["blocker_codes"])

    def test_blocker_histogram_is_sorted_and_counts_all_contracts(self) -> None:
        histogram = blocker_histogram(
            [
                {"blocker_codes": ["z", "a", "z"]},
                None,
                {"blocker_codes": ["a"]},
            ]
        )
        self.assertEqual(histogram, {"a": 2, "z": 2})


if __name__ == "__main__":
    unittest.main()
