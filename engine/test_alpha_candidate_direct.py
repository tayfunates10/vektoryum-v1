from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_direct import (
    apply_direct_element_alpha,
    make_direct_element_alpha_first,
)
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class DirectElementAlphaTests(unittest.TestCase):
    def test_binary_source_archives_only_proven_canvas_without_mask_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "binary.png"
            svg_path = root / "candidate.svg"
            source = np.zeros((64, 64, 4), dtype=np.uint8)
            source[16:48, 16:48, :3] = (20, 30, 40)
            source[16:48, 16:48, 3] = 255
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
                'viewBox="0 0 64 64">'
                '<path fill="#ffffff" d="M0 0h64v64H0Z"/>'
                '<path fill="#141e28" d="M16 16h32v32H16Z"/>'
                '</svg>',
                encoding="utf-8",
            )

            report = apply_direct_element_alpha(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertEqual(report["mask_encoding"], "direct_existing_elements")
            self.assertTrue(report["comparison_canvas_archived"])
            self.assertEqual(report["direct_translucent_child_count"], 0)
            text = svg_path.read_text(encoding="utf-8")
            self.assertNotIn("<clipPath", text)
            self.assertNotIn("<mask", text)
            self.assertNotIn("<rect", text)
            self.assertIn("renderer-proven-comparison-canvas-v1", text)

            parsed = ET.parse(svg_path).getroot()
            paths = [element for element in parsed.iter() if element.tag.endswith("path")]
            self.assertEqual(len(paths), 2)
            rendered = render_svg_to_rgba(svg_path, 64, 64)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.99)
            self.assertLessEqual(metrics["alpha_mae"], 0.01)

    def test_canvas_coloured_negative_space_is_archived_not_raster_masked(self):
        """Canvas-coloured paint over the canvas alone is archived, not rasterised."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "technical.png"
            svg_path = root / "candidate.svg"
            source = np.zeros((64, 64, 4), dtype=np.uint8)
            source[8:28, 8:28, :3] = (0, 0, 0)
            source[8:28, 8:28, 3] = 255
            Image.fromarray(source, mode="RGBA").save(source_path)
            # The trailing white square sits on the white canvas only, so removing
            # it genuinely leaves the source-transparent region unpainted.
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
                'viewBox="0 0 64 64">'
                '<path fill="#ffffff" d="M0 0h64v64H0Z"/>'
                '<path fill="#000000" d="M8 8h20v20H8Z"/>'
                '<path fill="#ffffff" d="M40 40h16v16H40Z"/>'
                '</svg>',
                encoding="utf-8",
            )

            report = apply_direct_element_alpha(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertTrue(report["comparison_canvas_archived"])
            self.assertEqual(
                report["canvas_colour_negative_space_archive_count"],
                1,
            )
            text = svg_path.read_text(encoding="utf-8")
            self.assertNotIn("<clipPath", text)
            self.assertNotIn("<mask", text)
            self.assertNotIn("<rect", text)
            self.assertIn("measured-canvas-colour-negative-space-v1", text)
            parsed = ET.parse(svg_path).getroot()
            paths = [element for element in parsed.iter() if element.tag.endswith("path")]
            self.assertEqual(len(paths), 3)
            rendered = render_svg_to_rgba(svg_path, 64, 64)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.99)
            self.assertLessEqual(metrics["alpha_mae"], 0.01)

    def test_negative_space_over_retained_paint_delegates_instead_of_filling_hole(self):
        """A hole punched over kept artwork cannot be archived away.

        Removing the canvas-coloured child reveals the artwork beneath, so the
        region stays opaque where the source is fully transparent. The hole is
        small enough for the whole-image reconstruction gate to tolerate, so the
        direct candidate must reject it and let the guarded chain represent it.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "holed.png"
            svg_path = root / "candidate.svg"
            side = 256
            source = np.zeros((side, side, 4), dtype=np.uint8)
            source[8:248, 8:248, :3] = (0, 0, 0)
            source[8:248, 8:248, 3] = 255
            source[120:136, 120:136, :] = 0
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
                'viewBox="0 0 256 256">'
                '<path fill="#ffffff" d="M0 0h256v256H0Z"/>'
                '<path fill="#000000" d="M8 8h240v240H8Z"/>'
                '<path fill="#ffffff" d="M120 120h16v16H120Z"/>'
                '</svg>',
                encoding="utf-8",
            )
            parent = svg_path.read_bytes()

            with self.assertRaises(RuntimeError) as caught:
                apply_direct_element_alpha(svg_path, source_path, "logo_color")
            self.assertIn(
                "source_alpha_direct_negative_space_occluded_by_retained_paint",
                str(caught.exception),
            )
            self.assertEqual(svg_path.read_bytes(), parent)

            def guarded(path, source, mode):
                del source, mode
                self.assertEqual(Path(path).read_bytes(), parent)
                return {"status": "guarded"}

            result = make_direct_element_alpha_first(guarded)(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertEqual(result["status"], "guarded")
            self.assertEqual(
                result["direct_element_alpha"]["status"],
                "rejected_fallback_used",
            )

    def test_soft_levels_use_existing_path_opacity_without_rect_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "soft.png"
            svg_path = root / "candidate.svg"
            source = np.zeros((32, 64, 4), dtype=np.uint8)
            source[8:24, 8:28, :3] = (255, 40, 80)
            source[8:24, 8:28, 3] = 180
            source[8:24, 36:56, :3] = (40, 160, 255)
            source[8:24, 36:56, 3] = 128
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="32" '
                'viewBox="0 0 64 32">'
                '<path fill="#ff2850" d="M8 8h20v16H8Z"/>'
                '<path fill="#28a0ff" d="M36 8h20v16H36Z"/>'
                '</svg>',
                encoding="utf-8",
            )

            report = apply_direct_element_alpha(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertFalse(report["comparison_canvas_archived"])
            self.assertEqual(report["direct_translucent_child_count"], 2)
            text = svg_path.read_text(encoding="utf-8")
            self.assertNotIn("<clipPath", text)
            self.assertNotIn("<mask", text)
            self.assertNotIn("<rect", text)
            self.assertIn('data-vektoryum-source-alpha-direct="180"', text)
            self.assertIn('data-vektoryum-source-alpha-direct="128"', text)
            rendered = render_svg_to_rgba(svg_path, 64, 32)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.99)
            self.assertLessEqual(metrics["alpha_mae"], 0.01)

    def test_rejected_direct_candidate_delegates_with_parent_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = np.zeros((32, 32, 4), dtype=np.uint8)
            source[12:20, 12:20, :3] = (20, 30, 40)
            source[12:20, 12:20, 3] = 255
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
                'viewBox="0 0 32 32">'
                '<path fill="#ffffff" d="M0 0h32v32H0Z"/>'
                '</svg>',
                encoding="utf-8",
            )
            parent = svg_path.read_bytes()
            delegated = []

            def guarded(path, source, mode):
                self.assertEqual(Path(path).read_bytes(), parent)
                delegated.append((Path(source), mode))
                return {"status": "guarded"}

            report = make_direct_element_alpha_first(guarded)(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertEqual(report["status"], "guarded")
            self.assertEqual(delegated, [(source_path, "logo_color")])
            self.assertEqual(svg_path.read_bytes(), parent)
            self.assertEqual(
                report["direct_element_alpha"]["status"],
                "rejected_fallback_used",
            )

    def test_existing_content_alpha_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = np.zeros((32, 32, 4), dtype=np.uint8)
            source[8:24, 8:24, :3] = (20, 30, 40)
            source[8:24, 8:24, 3] = 128
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
                'viewBox="0 0 32 32">'
                '<path fill="#141e28" opacity="0.5" d="M8 8h16v16H8Z"/>'
                '</svg>',
                encoding="utf-8",
            )
            parent = svg_path.read_bytes()
            called = False

            def guarded(path, source, mode):
                nonlocal called
                del source, mode
                called = True
                self.assertEqual(Path(path).read_bytes(), parent)
                return {"status": "guarded"}

            result = make_direct_element_alpha_first(guarded)(
                svg_path,
                source_path,
                "logo_color",
            )
            self.assertTrue(called)
            self.assertEqual(result["status"], "guarded")


if __name__ == "__main__":
    unittest.main()
