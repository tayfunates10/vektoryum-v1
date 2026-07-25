from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.alpha_parent_selection import _bind_parent_selection_journal
from app.alpha_pipeline_retry import (
    alpha_remediation_enabled,
    wrap_run_pipeline_with_alpha_retry,
)


class AlphaPipelineRetryTests(unittest.TestCase):
    def _transparent_source(self, root: Path) -> Path:
        path = root / "source.png"
        image = Image.new("RGBA", (8, 8), (20, 30, 40, 0))
        for y in range(2, 6):
            for x in range(2, 6):
                image.putpixel((x, y), (20, 30, 40, 255))
        image.save(path)
        return path

    def test_successful_legacy_pass_is_returned_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._transparent_source(root)
            calls: list[bool] = []
            expected = {"status": "legacy-success"}

            def pipeline(*args, **kwargs):
                del args, kwargs
                calls.append(alpha_remediation_enabled())
                return expected

            result = wrap_run_pipeline_with_alpha_retry(pipeline)(
                object(), source, "logo_color", root
            )
            self.assertIs(result, expected)
            self.assertEqual(calls, [False])
            self.assertNotIn("alpha_retry_report", result)

    def test_source_alpha_rejection_runs_one_enabled_remediation_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._transparent_source(root)
            calls: list[bool] = []

            def pipeline(*args, **kwargs):
                del args, kwargs
                enabled = alpha_remediation_enabled()
                calls.append(enabled)
                if not enabled:
                    raise RuntimeError(
                        "source_alpha_mask_transform_gate_rejected:ssim_regression"
                    )
                return {"status": "remediated"}

            result = wrap_run_pipeline_with_alpha_retry(pipeline)(
                object(), source, "logo_color", root
            )
            self.assertEqual(calls, [False, True])
            self.assertEqual(result["status"], "remediated")
            self.assertEqual(
                result["alpha_retry_report"]["status"],
                "remediation_succeeded",
            )
            self.assertEqual(result["alpha_retry_report"]["attempt_count"], 2)

    def test_unrelated_runtime_failure_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._transparent_source(root)
            calls: list[bool] = []

            def pipeline(*args, **kwargs):
                del args, kwargs
                calls.append(alpha_remediation_enabled())
                raise RuntimeError("unrelated_pipeline_failure")

            with self.assertRaisesRegex(RuntimeError, "unrelated_pipeline_failure"):
                wrap_run_pipeline_with_alpha_retry(pipeline)(
                    object(), source, "logo_color", root
                )
            self.assertEqual(calls, [False])

    def test_opaque_source_does_not_authorize_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "opaque.png"
            Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(source)
            calls: list[bool] = []

            def pipeline(*args, **kwargs):
                del args, kwargs
                calls.append(alpha_remediation_enabled())
                raise RuntimeError("source_alpha_mask_transform_gate_rejected:ssim")

            with self.assertRaisesRegex(RuntimeError, "source_alpha_mask_transform"):
                wrap_run_pipeline_with_alpha_retry(pipeline)(
                    object(), source, "logo_color", root
                )
            self.assertEqual(calls, [False])

    def test_alternate_parent_creates_valid_journal_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            best = root / "best.svg"
            selected = root / "selected.svg"
            Image.new("RGBA", (16, 16), (20, 30, 40, 255)).save(source)
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
                'viewBox="0 0 16 16"><path fill="#141e28" '
                'd="M0 0h16v16H0Z"/></svg>'
            )
            best.write_text(svg, encoding="utf-8")
            selected.write_text(svg, encoding="utf-8")
            digest = hashlib.sha256(best.read_bytes()).hexdigest()
            result = {
                "transform_journal": {
                    "schema_version": 1,
                    "baseline_sha256": digest,
                    "final_accepted_sha256": digest,
                    "stages": [],
                    "budget": {
                        "seconds": 0.0,
                        "elapsed_seconds": 0.0,
                        "wall_seconds": 0.0,
                        "stage_timeout_seconds": 0.0,
                        "max_side": 0,
                    },
                    "budget_exhausted": False,
                }
            }
            merged, stage, rejection = _bind_parent_selection_journal(
                result,
                best_path=best,
                selected_path=selected,
                source_path=source,
                mode="logo_color",
                trial={
                    "fidelity_score": 99.9,
                    "edge_f1": 1.0,
                    "path_count": 1,
                    "byte_size": selected.stat().st_size,
                },
            )
            self.assertIsNone(rejection)
            self.assertIsNotNone(stage)
            self.assertIsNotNone(merged)
            assert merged is not None
            self.assertTrue(merged["chain_valid"])
            self.assertEqual(merged["chain_failure_codes"], [])
            self.assertEqual(
                merged["stages"][-1]["stage_id"],
                "source_alpha_parent_selection",
            )


if __name__ == "__main__":
    unittest.main()
