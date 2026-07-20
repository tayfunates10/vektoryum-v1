from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import make_candidate_geometry_knockout_fallback
from app.alpha_candidate_support import make_candidate_support_reconstruction_fallback
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class CandidateSupportReconstructionTests(unittest.TestCase):
    @staticmethod
    def _source(path: Path) -> np.ndarray:
        rgba = np.zeros((64, 96, 4), dtype=np.uint8)
        rgba[15:50, 20:76, :3] = 0
        rgba[15:50, 20:76, 3] = 128
        rgba[17:48, 22:74, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(path)
        return rgba

    @staticmethod
    def _undercovered_candidate(path: Path) -> bytes:
        # The black candidate is deliberately one pixel inside the source-alpha
        # support. Plain clipping cannot create the missing boundary; a measured
        # same-color support stroke can, without changing the path data.
        data = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
            'viewBox="0 0 96 64">'
            '<path d="M0 0H96V64H0Z" fill="#FFFFFF"/>'
            '<path d="M21 16H75V49H21Z" fill="#000000"/>'
            '</svg>'
        ).encode("utf-8")
        path.write_bytes(data)
        return data

    def test_support_fallback_passes_unchanged_alpha_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = self._source(source_path)
            original = self._undercovered_candidate(svg_path)
            original_path_data = b'M21 16H75V49H21Z'
            self.assertIn(original_path_data, original)

            def rejected(*_args):
                raise RuntimeError("source_alpha_mask_iou_gate_failed:0.95<0.995")

            ordinary_knockout = make_candidate_geometry_knockout_fallback(rejected)
            wrapped = make_candidate_support_reconstruction_fallback(
                ordinary_knockout
            )
            report = wrapped(svg_path, source_path, "logo_color")

            self.assertTrue(report["applied"])
            self.assertEqual(
                report["mask_encoding"], "candidate_support_native_grid_use"
            )
            self.assertEqual(
                report["schema"], "rfv3d2-candidate-support-reconstruction-v2"
            )
            self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.995)
            self.assertLessEqual(report["source_truth_alpha_mae"], 0.005)
            self.assertGreater(report["candidate_support_stroke_width_pixels"], 0)
            self.assertGreater(report["reconstruction_rect_symbol_count"], 0)
            self.assertGreater(report["reconstruction_use_count"], 0)
            self.assertEqual(report["renderer_requested_alpha_width"], 96)
            self.assertEqual(report["renderer_requested_alpha_height"], 64)
            self.assertEqual(report["renderer_native_alpha_width"], 96)
            self.assertEqual(report["renderer_native_alpha_height"], 64)
            self.assertEqual(
                report["preflight_parent_path_count"], report["preserved_path_count"]
            )
            self.assertEqual(
                report["preflight_parent_node_count"], report["preserved_node_count"]
            )
            self.assertLessEqual(
                report["after_byte_size"], report["preflight_byte_limit"]
            )
            text = svg_path.read_bytes()
            self.assertIn(original_path_data, text)
            self.assertIn(b"native-grid-use-v1", text)
            self.assertIn(b"<use", text)
            self.assertNotIn(b"<image", text.lower())
            self.assertNotIn(b"data:image", text.lower())

            rendered = render_svg_to_rgba(svg_path, 96, 64)
            self.assertIsNotNone(rendered)
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.995)
            self.assertLessEqual(metrics["alpha_mae"], 0.005)

    def test_non_alpha_failure_is_not_intercepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            self._source(source_path)
            original = self._undercovered_candidate(svg_path)

            def rejected(*_args):
                raise RuntimeError("unrelated_failure")

            wrapped = make_candidate_support_reconstruction_fallback(rejected)
            with self.assertRaisesRegex(RuntimeError, "unrelated_failure"):
                wrapped(svg_path, source_path, "logo_color")
            self.assertEqual(svg_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
