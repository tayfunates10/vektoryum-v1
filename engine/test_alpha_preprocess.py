from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_preprocess import wrap_gradient_vectorizer, wrap_preprocess_for_mode
from app.alpha_svg_mask import (
    apply_source_alpha_mask,
    wrap_run_pipeline_with_alpha_mask,
)
from app.final_artifact_evaluator import _thresholds
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba


class AlphaPreprocessUnitTests(unittest.TestCase):
    def test_transparent_color_preprocess_stages_exact_alpha_contract(self) -> None:
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
                self.assertEqual(verified_image.mode, "RGB")
                verified = np.asarray(verified_image, dtype=np.uint8).copy()
            self.assertTrue(np.all(verified[source[:, :, 3] == 0] == 0))
            partial = (source[:, :, 3] > 0) & (source[:, :, 3] < 255)
            np.testing.assert_array_equal(
                verified[partial], source[:, :, :3][partial]
            )
            self.assertEqual(
                report["source_alpha"]["status"], "staged_for_vector_mask"
            )
            self.assertEqual(report["source_alpha"]["trace_input_mode"], "RGB")
            self.assertIn("source_alpha_staged", report["steps"])
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

    def test_non_color_mode_retains_existing_preprocess_contract(self) -> None:
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

    def test_non_color_mode_bypasses_final_alpha_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "lineart.svg"
            Image.new("RGBA", (8, 8), (20, 30, 40, 0)).save(source_path)
            original_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" '
                'viewBox="0 0 8 8"><path fill="#000" d="M0 0h8v8H0Z"/></svg>'
            )
            svg_path.write_text(original_svg, encoding="utf-8")

            def lineart_pipeline(*args, **kwargs):
                del args, kwargs
                return {
                    "mode_used": "lineart",
                    "best": {"name": "lineart", "svg_path": svg_path},
                    "scored": [],
                }

            wrapped = wrap_run_pipeline_with_alpha_mask(lineart_pipeline)
            result = wrapped(
                Image.open(source_path),
                source_path,
                "lineart",
                root,
            )
            self.assertEqual(
                result["alpha_mask_report"]["reason"],
                "unsupported_non_color_mode",
            )
            self.assertFalse(result["alpha_mask_report"]["applied"])
            self.assertEqual(svg_path.read_text(encoding="utf-8"), original_svg)

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
    def test_vector_mask_repairs_live_opaque_canvas_signature(self) -> None:
        import vtracer
        from app.preprocess import preprocess_for_mode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "transparent-logo.png"
            source = np.zeros((128, 128, 4), dtype=np.uint8)
            source[31:97, 31:97, :3] = (214, 32, 48)
            source[31:97, 31:97, 3] = 255
            source[30, 31:97, :3] = (214, 32, 48)
            source[97, 31:97, :3] = (214, 32, 48)
            source[31:97, 30, :3] = (214, 32, 48)
            source[31:97, 97, :3] = (214, 32, 48)
            source[30, 31:97, 3] = 128
            source[97, 31:97, 3] = 128
            source[31:97, 30, 3] = 128
            source[31:97, 97, 3] = 128
            Image.fromarray(source, mode="RGBA").save(source_path)

            processed_path, report = preprocess_for_mode(
                source_path,
                "logo_color",
                root,
                analysis={"estimated_color_count": 2},
            )
            self.assertIn("source_alpha_staged", report["steps"])
            with Image.open(processed_path) as processed_image:
                self.assertEqual(processed_image.mode, "RGB")
                width, height = processed_image.size

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

            opaque_render = render_svg_to_rgba(svg_path, width, height)
            self.assertIsNotNone(opaque_render)
            assert opaque_render is not None
            opaque_coverage = float(
                opaque_render[:, :, 3].astype(np.float32).mean() / 255.0
            )
            self.assertGreaterEqual(opaque_coverage, 0.995)

            mask_report = apply_source_alpha_mask(
                svg_path, source_path, "logo_color"
            )
            self.assertTrue(mask_report["applied"])
            self.assertEqual(mask_report["status"], "accepted")
            self.assertGreater(mask_report["mask_path_count"], 0)
            self.assertNotIn("<image", svg_path.read_text(encoding="utf-8"))

            rendered = render_svg_to_rgba(svg_path, width, height)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            with Image.open(source_path) as source_image:
                source_resized = np.asarray(
                    source_image.convert("RGBA").resize(
                        (width, height), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                ).copy()
            metrics = alpha_plane_metrics(
                source_resized[:, :, 3], rendered[:, :, 3]
            )
            thresholds = _thresholds("clean_logo", None)
            self.assertGreaterEqual(
                metrics["alpha_iou"], thresholds["alpha_iou_min"], metrics
            )
            self.assertLessEqual(
                metrics["alpha_mae"], thresholds["alpha_mae_max"], metrics
            )
            self.assertLess(metrics["render_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
