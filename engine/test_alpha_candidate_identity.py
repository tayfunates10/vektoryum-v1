from __future__ import annotations

import unittest
from pathlib import Path

from app.alpha_candidate_identity import (
    wrap_run_pipeline_preserving_candidate_identity,
)


class AlphaCandidateIdentityTests(unittest.TestCase):
    def test_alpha_artifact_keeps_engine_candidate_name(self) -> None:
        original_candidate = {
            "name": "geo_standard",
            "svg_path": Path("geo_standard.svg"),
            "rendered_ok": True,
        }
        final_candidate = {
            "name": "geo_standard_alpha",
            "svg_path": Path("geo_standard_alpha.svg"),
            "rendered_ok": True,
            "alpha_mask_report": {"applied": True},
        }
        result = {
            "best": final_candidate,
            "results": [original_candidate],
            "scored": [original_candidate, final_candidate],
            "alpha_mask_report": final_candidate["alpha_mask_report"],
            "selection_reason": "fidelity_best+source_alpha_vector_mask",
        }

        wrapped = wrap_run_pipeline_preserving_candidate_identity(
            lambda *args, **kwargs: result
        )
        actual = wrapped()

        self.assertEqual(actual["best"]["name"], "geo_standard")
        self.assertEqual(actual["scored"][-1]["name"], "geo_standard")
        self.assertEqual(
            actual["alpha_mask_report"]["source_candidate_name"],
            "geo_standard",
        )
        self.assertEqual(
            actual["candidate_identity"],
            {
                "status": "preserved",
                "source_candidate_name": "geo_standard",
                "artifact_transform": "source_alpha_vector_mask",
            },
        )
        self.assertEqual(
            actual["best"]["svg_path"], Path("geo_standard_alpha.svg")
        )
        self.assertIn("source_alpha_vector_mask", actual["selection_reason"])

    def test_identity_is_fail_closed_when_source_candidate_is_unbound(self) -> None:
        result = {
            "best": {
                "name": "invented_alpha",
                "svg_path": Path("invented_alpha.svg"),
            },
            "results": [],
            "scored": [],
            "alpha_mask_report": {"applied": True},
        }
        wrapped = wrap_run_pipeline_preserving_candidate_identity(
            lambda *args, **kwargs: result
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "source_alpha_candidate_identity_unbound",
        ):
            wrapped()

    def test_non_applied_alpha_report_is_unchanged(self) -> None:
        result = {
            "best": {
                "name": "logo_gradient",
                "svg_path": Path("logo_gradient.svg"),
            },
            "scored": [],
            "alpha_mask_report": {"applied": False, "reason": "opaque_source"},
        }
        wrapped = wrap_run_pipeline_preserving_candidate_identity(
            lambda *args, **kwargs: result
        )
        self.assertIs(wrapped(), result)
        self.assertEqual(result["best"]["name"], "logo_gradient")
        self.assertNotIn("candidate_identity", result)


if __name__ == "__main__":
    unittest.main()
