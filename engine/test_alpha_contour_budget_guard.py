from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import alpha_mask_adaptive
from app.alpha_contour_budget_guard import (
    _FRAGMENTATION_BUDGET_PREFIX,
    _fragmented_one_cell_stats,
    _preflight_fragmented_contour_nodes,
)


class AlphaContourBudgetGuardTests(unittest.TestCase):
    def test_fragmented_cells_are_rejected_from_exact_node_count(self) -> None:
        yy, xx = np.indices((128, 128))
        quantized = np.where((xx + yy) % 2 == 0, 255, 0).astype(np.uint8)
        opacity = {255: 1.0}

        stats = _fragmented_one_cell_stats(
            quantized,
            opacity,
            max_compact_contours=4096,
        )
        self.assertTrue(stats["fragmented"])
        self.assertEqual(stats["cell_count"], 8192)
        self.assertEqual(stats["command_count"], 40960)

        with self.assertRaisesRegex(
            RuntimeError,
            r"source_alpha_mask_contour_fragmentation_budget_rejected:"
            r"path_nodes=40960/1000,alpha_cells=8192",
        ):
            _preflight_fragmented_contour_nodes(
                quantized,
                opacity,
                available_nodes=1000,
                max_compact_contours=4096,
            )

    def test_nonfragmented_region_keeps_legacy_contour_path(self) -> None:
        quantized = np.zeros((96, 96), dtype=np.uint8)
        quantized[12:84, 18:78] = 255
        stats = _preflight_fragmented_contour_nodes(
            quantized,
            {255: 1.0},
            available_nodes=1,
            max_compact_contours=4096,
        )
        self.assertFalse(stats["fragmented"])
        self.assertEqual(stats["command_count"], 0)

    def test_fragmented_plan_with_real_capacity_is_not_rejected(self) -> None:
        yy, xx = np.indices((32, 32))
        quantized = np.where((xx + yy) % 2 == 0, 255, 0).astype(np.uint8)
        stats = _preflight_fragmented_contour_nodes(
            quantized,
            {255: 1.0},
            available_nodes=3000,
            max_compact_contours=64,
        )
        self.assertTrue(stats["fragmented"])
        self.assertEqual(stats["command_count"], 2560)

    def test_impossible_contour_retry_preserves_original_alpha_gate(self) -> None:
        def guarded_builder(_svg: Path, _source: Path, _mode: str):
            raise RuntimeError("source_alpha_mask_iou_gate_failed:0.900000<0.995")

        wrapped = alpha_mask_adaptive.make_rect_fidelity_fallback(guarded_builder)
        contour_error = RuntimeError(
            _FRAGMENTATION_BUDGET_PREFIX
            "path_nodes=4058631/95784,alpha_cells=811726,"
            "compact_contours=4097,pruned_contours=0"
        )
        with patch.object(
            alpha_mask_adaptive,
            "_contour_fallback_plan",
            side_effect=contour_error,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"source_alpha_mask_iou_gate_failed:0.900000<0.995",
            ) as raised:
                wrapped(Path("selected.svg"), Path("source.png"), "logo_color")

        self.assertIs(raised.exception.__cause__, contour_error)

    def test_unexpected_contour_error_is_not_reclassified(self) -> None:
        def guarded_builder(_svg: Path, _source: Path, _mode: str):
            raise RuntimeError("source_alpha_mask_mae_gate_failed:0.100000>0.005")

        wrapped = alpha_mask_adaptive.make_rect_fidelity_fallback(guarded_builder)
        with patch.object(
            alpha_mask_adaptive,
            "_contour_fallback_plan",
            side_effect=RuntimeError("unexpected_contour_bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected_contour_bug"):
                wrapped(Path("selected.svg"), Path("source.png"), "logo_color")


if __name__ == "__main__":
    unittest.main()
