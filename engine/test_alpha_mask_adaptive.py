from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from app import alpha_mask_adaptive, alpha_svg_mask
from app.alpha_contour_budget_guard import (
    _FRAGMENTATION_BUDGET_PREFIX,
    _fragmented_one_cell_stats,
    _preflight_fragmented_contour_nodes,
)
from app.alpha_mask_adaptive import (
    _compact_mask_rectangles,
    make_adaptive_apply_source_alpha_mask,
    make_rect_fidelity_fallback,
)
from app.alpha_mask_budget import _preflight, wrap_apply_source_alpha_mask
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class AlphaMaskAdaptiveTests(unittest.TestCase):
    def test_preflight_selects_path_only_when_all_journal_budgets_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "checkerboard.png"
            svg_path = root / "node-rich.svg"

            source = np.zeros((128, 128, 4), dtype=np.uint8)
            source[:, :, :3] = (20, 30, 40)
            yy, xx = np.indices((128, 128))
            source[:, :, 3] = np.where((xx + yy) % 2 == 0, 255, 0).astype(
                np.uint8
            )
            Image.fromarray(source, mode="RGBA").save(source_path)

            # Many compact path commands provide genuine parent node budget
            # without inflating the byte budget enough for the verbose rect XML.
            parent_d = "M0 0Z" * 20500
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="128" '
                'height="128" viewBox="0 0 128 128">'
                f'<path fill="#141e28" d="{parent_d}"/>'
                "</svg>",
                encoding="utf-8",
            )

            report = _preflight(svg_path, source_path)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report["mask_encoding"], "path")
            self.assertGreater(
                report["preflight_rect_projected_byte_size"],
                report["preflight_byte_limit"],
            )
            self.assertLessEqual(
                report["preflight_path_projected_byte_size"],
                report["preflight_byte_limit"],
            )
            self.assertLessEqual(
                report["preflight_path_count_after"],
                report["preflight_path_limit"],
            )
            self.assertLessEqual(
                report["preflight_path_node_count_after"],
                report["preflight_node_limit"],
            )

    def test_compact_path_encoding_preserves_exact_alpha_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "selected.svg"

            source = np.zeros((64, 64, 4), dtype=np.uint8)
            source[12:52, 12:52, :3] = (214, 32, 48)
            source[12:52, 12:52, 3] = 255
            source[11, 12:52, :3] = (214, 32, 48)
            source[11, 12:52, 3] = 128
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" '
                'height="64" viewBox="0 0 64 64">'
                '<path fill="#d62030" d="M0 0h64v64H0Z"/>'
                "</svg>",
                encoding="utf-8",
            )

            base_builder = inspect.unwrap(alpha_svg_mask.apply_source_alpha_mask)
            adaptive = make_adaptive_apply_source_alpha_mask(base_builder)
            with patch(
                "app.alpha_mask_budget.current_alpha_mask_encoding",
                return_value="path",
            ):
                report = adaptive(svg_path, source_path, "logo_color")

            self.assertEqual(report["mask_encoding"], "path")
            self.assertGreater(report["mask_path_count"], 0)
            self.assertGreater(report["mask_rectangle_count"], 0)
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("<path", svg_text)
            self.assertNotIn("<rect", svg_text)
            self.assertNotIn("<image", svg_text)

            rendered = render_svg_to_rgba(svg_path, 64, 64)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.995, metrics)
            self.assertLessEqual(metrics["alpha_mae"], 0.005, metrics)

    def test_rect_alpha_seams_retry_as_budgeted_contours(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "rounded-source.png"
            svg_path = root / "selected.svg"
            size = 1990

            source_image = Image.new("RGBA", (size, size), (214, 32, 48, 0))
            alpha = Image.new("L", (size, size), 0)
            ImageDraw.Draw(alpha).rounded_rectangle(
                (20, 20, size - 21, size - 21),
                radius=250,
                fill=255,
            )
            source_image.putalpha(alpha)
            source_image.save(source_path)
            svg_path.write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
                f'height="{size}" viewBox="0 0 {size} {size}">'
                f'<path fill="#d62030" d="M0 0h{size}v{size}H0Z"/>'
                "</svg>",
                encoding="utf-8",
            )

            base_builder = inspect.unwrap(alpha_svg_mask.apply_source_alpha_mask)
            guarded = make_rect_fidelity_fallback(
                wrap_apply_source_alpha_mask(
                    make_adaptive_apply_source_alpha_mask(base_builder)
                )
            )
            report = guarded(svg_path, source_path, "logo_color")

            self.assertEqual(report["preflight_mask_encoding"], "rect")
            self.assertEqual(report["mask_encoding"], "path")
            self.assertEqual(
                report["mask_fallback_reason"],
                "rect_exact_alpha_gate_failure",
            )
            self.assertTrue(
                str(report["mask_fallback_trigger"]).startswith(
                    "source_alpha_mask_iou_gate_failed:"
                ),
                report,
            )
            self.assertLessEqual(
                report["fallback_path_projected_byte_size"],
                report["fallback_byte_limit"],
            )
            self.assertLessEqual(
                report["fallback_path_count_after"],
                report["fallback_path_limit"],
            )
            self.assertLessEqual(
                report["fallback_path_node_count_after"],
                report["fallback_node_limit"],
            )
            self.assertGreaterEqual(report["alpha_iou"], 0.995, report)
            self.assertLessEqual(report["alpha_mae"], 0.005, report)
            self.assertEqual(report["rollback_guard"], "armed_and_committed")

            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("<path", svg_text)
            self.assertNotIn("<rect", svg_text)
            self.assertNotIn("<image", svg_text)

    def test_fragmented_cells_reject_before_svg_path_construction(self) -> None:
        yy, xx = np.indices((128, 128))
        quantized = np.where((xx + yy) % 2 == 0, 255, 0).astype(np.uint8)
        stats = _fragmented_one_cell_stats(
            quantized,
            {255: 1.0},
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
                {255: 1.0},
                available_nodes=1000,
                max_compact_contours=4096,
            )

    def test_nonfragmented_contour_keeps_legacy_path(self) -> None:
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

    def test_fragmented_contour_with_real_node_capacity_remains_eligible(self) -> None:
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

        wrapped = make_rect_fidelity_fallback(guarded_builder)
        contour_error = RuntimeError(
            _FRAGMENTATION_BUDGET_PREFIX
            + "path_nodes=4058631/95784,alpha_cells=811726,"
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

        wrapped = make_rect_fidelity_fallback(guarded_builder)
        with patch.object(
            alpha_mask_adaptive,
            "_contour_fallback_plan",
            side_effect=RuntimeError("unexpected_contour_bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected_contour_bug"):
                wrapped(Path("selected.svg"), Path("source.png"), "logo_color")

    def test_compaction_requires_existing_rect_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            svg_path = Path(directory) / "plain.svg"
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="8" '
                'height="8" viewBox="0 0 8 8"><path d="M0 0Z"/></svg>',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "source_alpha_compact_mask_missing",
            ):
                _compact_mask_rectangles(svg_path)


if __name__ == "__main__":
    unittest.main()
