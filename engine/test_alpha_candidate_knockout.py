from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import (
    apply_candidate_geometry_knockout,
    make_candidate_geometry_knockout_fallback,
)
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class CandidateGeometryKnockoutTests(unittest.TestCase):
    @staticmethod
    def _source(path: Path, *, opaque: bool = False) -> np.ndarray:
        rgba = np.zeros((64, 96, 4), dtype=np.uint8)
        if opaque:
            rgba[:, :, :3] = 255
            rgba[:, :, 3] = 255
        else:
            rgba[15:50, 20:76, :3] = 0
            rgba[15:50, 20:76, 3] = 128
            rgba[17:48, 22:74, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(path)
        return rgba

    @staticmethod
    def _candidate(path: Path, *, canvas: bool = True) -> bytes:
        canvas_path = (
            '<path d="M0 0H96V64H0Z" fill="#FFFFFF"/>' if canvas else ""
        )
        data = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
            'viewBox="0 0 96 64">'
            f'{canvas_path}'
            '<path d="M20 15H76V50H20Z" fill="#000000"/>'
            '</svg>'
        ).encode("utf-8")
        path.write_bytes(data)
        return data

    def test_knockout_reconstructs_alpha_and_preserves_candidate_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = self._source(source_path)
            self._candidate(svg_path)

            report = apply_candidate_geometry_knockout(
                svg_path, source_path, "logo_color"
            )

            self.assertTrue(report["applied"])
            self.assertEqual(report["mask_encoding"], "candidate_geometry_knockout")
            self.assertTrue(report["candidate_geometry_preserved"])
            self.assertEqual(report["preserved_path_count"], 2)
            self.assertEqual(
                report["preflight_parent_path_count"], report["preserved_path_count"]
            )
            self.assertEqual(
                report["preflight_parent_node_count"], report["preserved_node_count"]
            )
            self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.995)
            self.assertLessEqual(report["source_truth_alpha_mae"], 0.005)
            text = svg_path.read_text(encoding="utf-8")
            self.assertNotIn("<image", text.lower())
            self.assertNotIn("data:image", text.lower())
            self.assertIn("candidate-geometry-knockout", text)

            rendered = render_svg_to_rgba(svg_path, 96, 64)
            self.assertIsNotNone(rendered)
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.995)
            self.assertLessEqual(metrics["alpha_mae"], 0.005)

    def test_wrapper_retries_only_exact_alpha_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            self._source(source_path)
            self._candidate(svg_path)

            def rejected(*_args):
                raise RuntimeError("source_alpha_mask_iou_gate_failed:0.95<0.995")

            wrapped = make_candidate_geometry_knockout_fallback(rejected)
            report = wrapped(svg_path, source_path, "logo_color")
            self.assertEqual(
                report["mask_fallback_reason"], "source_alpha_exact_gate_failure"
            )
            self.assertEqual(report["rollback_guard"], "armed_and_committed")

    def test_failed_knockout_restores_exact_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            self._source(source_path)
            original = self._candidate(svg_path, canvas=False)

            def rejected(*_args):
                raise RuntimeError("source_alpha_mask_iou_gate_failed:0.95<0.995")

            wrapped = make_candidate_geometry_knockout_fallback(rejected)
            with self.assertRaisesRegex(
                RuntimeError, "candidate_knockout_(parent_not_collapsed|canvas_not_proven)"
            ):
                wrapped(svg_path, source_path, "logo_color")
            self.assertEqual(svg_path.read_bytes(), original)

    def test_opaque_source_is_not_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            self._source(source_path, opaque=True)
            self._candidate(svg_path)
            with self.assertRaisesRegex(RuntimeError, "opaque_source"):
                apply_candidate_geometry_knockout(
                    svg_path, source_path, "logo_color"
                )


if __name__ == "__main__":
    unittest.main()
