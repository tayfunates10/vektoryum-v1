from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.alpha_preprocess import wrap_gradient_vectorizer, wrap_preprocess_for_mode
from app.final_artifact_evaluator import _thresholds
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class AlphaPreprocessUnitTests(unittest.TestCase):
    def test_transparent_color_preprocess_restores_exact_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "processed.png"
            source = np.zeros((4, 4, 4), dtype=np.uint8)
            source[:, :, :3] = (180, 20, 30)
            source[:, :, 3] = np.array(
                [
                    [0, 0, 64, 255],
                    [0, 128, 192, 255],
                    [0, 128, 255, 255],
                    [0, 0, 0, 255],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(source, mode="RGBA").save(source_path)

            def opaque_preprocess(*args, **kwargs):
                del args, kwargs
                Image.new("RGB", (4, 4), (220, 220, 220)).save(output_path)
                return output_path, {"mode": "logo_color", "steps": ["stub"]}

            wrapped = wrap_preprocess_for_mode(opaque_preprocess)
            processed_path, report = wrapped(
                source_path,
                "logo_color",
                root,
            )

            with Image.open(processed_path) as verified_image:
                self.assertEqual(verified_image.mode, "RGBA")
                verified = np.asarray(verified_image.convert("RGBA"), dtype=np.uint8)
            np.testing.assert_array_equal(verified[:, :, 3], source[:, :, 3])
            self.assertTrue(np.all(verified[source[:, :, 3] == 0, :3] == 0))
            self.assertEqual(report["source_alpha"]["status"], "preserved")
            self.assertIn("source_alpha_preserved", report["steps"])
            self.assertEqual(len(report["source_alpha"]["alpha_sha256"]), 64)

    def test_opaque_source_keeps_existing_rgb_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "processed.png"
            Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(source_path)

            def opaque_preprocess(*args, **kwargs):
                del args, kwargs
                Image.new("RGB", (8, 8), (10, 20, 30)).save(output_path)
                return output_path, {"mode": "logo_color", "steps": []}

            wrapped = wrap_preprocess_for_mode(opaque_preprocess)
            processed_path, report = wrapped(source_path, "logo_color", root)
            with Image.open(processed_path) as verified_image:
                self.assertEqual(verified_image.mode, "RGB")
            self.assertNotIn("source_alpha", report)

    def test_non_color_mode_retains_existing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "processed.png"
            Image.new("RGBA", (8, 8), (20, 30, 40, 0)).save(source_path)

            def lineart_preprocess(*args, **kwargs):
                del args, kwargs
                Image.new("L", (8, 8), 255).save(output_path)
                return output_path, {"mode": "lineart", "steps": []}

            wrapped = wrap_preprocess_for_mode(lineart_preprocess)
            processed_path, report = wrapped(source_path, "lineart", root)
            with Image.open(processed_path) as verified_image:
                self.assertEqual(verified_image.mode, "L")
            self.assertNotIn("source_alpha", report)

    def test_transparent_gradient_candidate_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "candidate.svg"
            Image.new("RGBA", (8, 8), (10, 20, 30, 0)).save(source_path)
            called = False

            def gradient(*args, **kwargs):
                nonlocal called
                called = True
                del args, kwargs

            wrapped = wrap_gradient_vectorizer(gradient)
            with self.assertRaisesRegex(
                RuntimeError,
                "transparent_gradient_candidate_requires_alpha_aware_mask",
            ):
                wrapped(source_path, output_path, {})
            self.assertFalse(called)
            self.assertFalse(output_path.exists())

    def test_opaque_gradient_candidate_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "candidate.svg"
            Image.new("RGBA", (8, 8), (10, 20, 30, 255)).save(source_path)

            def gradient(input_path, destination, params):
                self.assertEqual(Path(input_path), source_path)
                self.assertEqual(params, {"epsilon": 0.3})
                Path(destination).write_text("<svg/>", encoding="utf-8")

            wrapped = wrap_gradient_vectorizer(gradient)
            wrapped(source_path, output_path, {"epsilon": 0.3})
            self.assertEqual(output_path.read_text(encoding="utf-8"), "<svg/>")


class AlphaPreprocessProductionIntegrationTests(unittest.TestCase):
    def test_vtracer_output_does_not_collapse_to_opaque_canvas(self) -> None:
        import vtracer
        from app.preprocess import preprocess_for_mode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "transparent-logo.png"
            image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((32, 32, 95, 95), fill=(214, 32, 48, 255))
            image.save(source_path)

            processed_path, report = preprocess_for_mode(
                source_path,
                "logo_color",
                root,
                analysis={"estimated_color_count": 2},
            )
            self.assertIn("source_alpha_preserved", report["steps"])
            with Image.open(processed_path) as processed_image:
                processed_rgba = np.asarray(
                    processed_image.convert("RGBA"), dtype=np.uint8
                ).copy()

            svg_path = root / "candidate.svg"
            vtracer.convert_image_to_svg_py(
                str(processed_path),
                str(svg_path),
                colormode="color",
                mode="spline",
                color_precision=5,
                filter_speckle=4,
                layer_difference=24,
                corner_threshold=55,
                length_threshold=4.0,
                path_precision=5,
            )
            rendered = render_svg_to_rgba(
                svg_path,
                processed_rgba.shape[1],
                processed_rgba.shape[0],
            )
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(
                processed_rgba[:, :, 3], rendered[:, :, 3]
            )
            thresholds = _thresholds("clean_logo", None)
            self.assertGreaterEqual(
                metrics["alpha_iou"], thresholds["alpha_iou_min"]
            )
            self.assertLessEqual(
                metrics["alpha_mae"], thresholds["alpha_mae_max"]
            )
            self.assertLess(metrics["render_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
