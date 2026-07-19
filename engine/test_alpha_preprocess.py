from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            transparent = source[:, :, 3] == 0
            self.assertTrue(np.all(verified[transparent] == 220))
            partial = (source[:, :, 3] > 0) & (source[:, :, 3] < 255)
            np.testing.assert_array_equal(
                verified[partial], source[:, :, :3][partial]
            )
            self.assertEqual(
                report["source_alpha"]["status"], "staged_for_vector_mask"
            )
            self.assertEqual(report["source_alpha"]["trace_input_mode"], "RGB")
            self.assertEqual(
                report["source_alpha"]["trace_background_policy"],
                "retain_processed_composite",
            )
            self.assertEqual(
                report["source_alpha"]["soft_boundary_rgb_policy"],
                "straight_source_rgb",
            )
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
            with Image.open(source_path) as opened:
                source_image = opened.copy()
            result = wrapped(
                source_image,
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

    def test_color_finalizer_uses_real_transform_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "selected.svg"
            source = np.zeros((64, 64, 4), dtype=np.uint8)
            source[16:48, 16:48, :3] = (214, 32, 48)
            source[16:48, 16:48, 3] = 255
            Image.fromarray(source, mode="RGBA").save(source_path)
            svg_path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
                'viewBox="0 0 64 64">'
                '<path fill="#000000" d="M0 0h64v64H0Z"/>'
                '<path fill="#d62030" d="M16 16h32v32H16Z"/>'
                '</svg>',
                encoding="utf-8",
            )
            parent_sha = hashlib.sha256(svg_path.read_bytes()).hexdigest()

            def color_pipeline(*args, **kwargs):
                del args, kwargs
                return {
                    "analysis": {},
                    "mode_used": "logo_color",
                    "best": {
                        "name": "selected",
                        "svg_path": svg_path,
                        "rendered_ok": True,
                        "fidelity_score": 50.0,
                    },
                    "scored": [],
                    "selection_reason": "fidelity_best",
                    "refit_info": {},
                    "transform_journal": {
                        "schema_version": 1,
                        "baseline_sha256": parent_sha,
                        "final_accepted_sha256": parent_sha,
                        "stages": [],
                        "budget": {
                            "seconds": 0.0,
                            "elapsed_seconds": 0.0,
                            "wall_seconds": 0.0,
                            "stage_timeout_seconds": 0.0,
                            "max_side": 0,
                        },
                        "budget_exhausted": False,
                        "chain_valid": True,
                        "chain_failure_codes": [],
                    },
                }

            def measured_candidate(candidate, *args, **kwargs):
                del args, kwargs
                return {
                    **candidate,
                    "rendered_ok": True,
                    "fidelity_score": 99.0,
                    "total_score": 99.0,
                    "score_details": {"path_count": 3},
                }

            wrapped = wrap_run_pipeline_with_alpha_mask(color_pipeline)
            with Image.open(source_path) as opened:
                source_image = opened.copy()
            with patch(
                "app.pipeline.score_candidate",
                side_effect=measured_candidate,
            ), patch(
                "app.pipeline.score_structure_integrity",
                return_value={"status": "measured"},
            ):
                result = wrapped(
                    source_image,
                    source_path,
                    "logo_color",
                    root,
                )

            self.assertTrue(result["alpha_mask_report"]["applied"])
            self.assertEqual(
                result["alpha_mask_report"]["journal_status"], "accepted"
            )
            self.assertTrue(result["transform_journal"]["chain_valid"])
            self.assertEqual(
                result["transform_journal"]["final_accepted_sha256"],
                result["alpha_mask_report"]["after_sha256"],
            )
            self.assertEqual(
                result["transform_journal"]["stages"][-1]["stage_id"],
                "source_alpha_vector_mask",
            )
            self.assertGreater(
                result["alpha_mask_report"]["preflight_rectangle_limit"],
                result["alpha_mask_report"]["preflight_rectangle_count"],
            )
            self.assertNotEqual(Path(result["best"]["svg_path"]), svg_path)

    def test_noisy_alpha_fails_before_svg_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "checkerboard.png"
            svg_path = root / "selected.svg"
            source = np.zeros((128, 128, 4), dtype=np.uint8)
            source[:, :, :3] = (20, 30, 40)
            yy, xx = np.indices((128, 128))
            source[:, :, 3] = np.where((xx + yy) % 2 == 0, 255, 0).astype(
                np.uint8
            )
            Image.fromarray(source, mode="RGBA").save(source_path)
            original_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" '
                'viewBox="0 0 128 128"><path fill="#141e28" '
                'd="M0 0h128v128H0Z"/></svg>'
            )
            svg_path.write_text(original_svg, encoding="utf-8")
            before_bytes = svg_path.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                "source_alpha_mask_(rectangle|byte)_budget_exceeded",
            ):
                apply_source_alpha_mask(svg_path, source_path, "logo_color")

            self.assertEqual(svg_path.read_bytes(), before_bytes)
            self.assertNotIn(
                "vektoryum-source-alpha",
                svg_path.read_text(encoding="utf-8"),
            )

    def test_transparent_gradient_candidate_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_path = root / "candidate.svg"
            Image.new("RGBA", (8, 8), (10, 20, 30, 0)).save(source_path)
            called = False

            def gradient(input_path, destination, params):
                nonlocal called
                called = True
                self.assertEqual(Path(input_path), source_path)
                self.assertEqual(params, {"epsilon": 0.3})
                Path(destination).write_text("<svg/>", encoding="utf-8")

            wrapped = wrap_gradient_vectorizer(gradient)
            wrapped(source_path, output_path, {"epsilon": 0.3})
            self.assertTrue(called)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "<svg/>")

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
    def test_vector_gradient_survives_final_alpha_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "transparent-gradient.png"
            svg_path = root / "gradient.svg"
            width, height = 128, 64
            source = np.zeros((height, width, 4), dtype=np.uint8)
            source[8:56, 12:116, 3] = 255
            Image.fromarray(source, mode="RGBA").save(source_path)

            svg_path.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="64" '
                'viewBox="0 0 128 64">'
                '<defs><linearGradient id="g0" gradientUnits="userSpaceOnUse" '
                'x1="0" y1="0" x2="128" y2="0">'
                '<stop offset="0" stop-color="#f02828"/>'
                '<stop offset="1" stop-color="#2828dc"/>'
                '</linearGradient></defs>'
                '<path fill="url(#g0)" d="M0 0h128v64H0Z"/>'
                '</svg>',
                encoding="utf-8",
            )
            before = svg_path.read_text(encoding="utf-8")
            self.assertIn("<linearGradient", before)

            report = apply_source_alpha_mask(
                svg_path, source_path, "logo_color"
            )
            self.assertTrue(report["applied"])
            after = svg_path.read_text(encoding="utf-8")
            self.assertIn("linearGradient", after)
            self.assertIn("url(#g0)", after)
            self.assertNotIn("<image", after)

            rendered = render_svg_to_rgba(svg_path, width, height)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            metrics = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            thresholds = _thresholds("clean_logo", None)
            self.assertGreaterEqual(
                metrics["alpha_iou"], thresholds["alpha_iou_min"], metrics
            )
            self.assertLessEqual(
                metrics["alpha_mae"], thresholds["alpha_mae_max"], metrics
            )

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
            self.assertEqual(mask_report["mask_path_count"], 0)
            self.assertGreater(mask_report["mask_group_count"], 0)
            self.assertGreater(mask_report["mask_rectangle_count"], 0)
            self.assertLessEqual(
                mask_report["preflight_rectangle_count"],
                mask_report["preflight_rectangle_limit"],
            )
            self.assertLessEqual(
                mask_report["preflight_projected_upper_bound"],
                mask_report["preflight_byte_limit"],
            )
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertNotIn("<image", svg_text)
            self.assertIn("<rect", svg_text)

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
