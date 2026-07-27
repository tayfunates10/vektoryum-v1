from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import app.alpha_parent_selection as selection
from engine.regression.output_quality_root_cause import pipeline_snapshot


class AlphaParentTrialDiagnosticTests(unittest.TestCase):
    def test_snapshot_is_json_safe_and_preserves_decision_fields(self) -> None:
        trials = [
            {
                "candidate": {
                    "name": "flat-parent",
                    "engine": "vtracer",
                    "svg_path": Path("/private/flat.svg"),
                },
                "rendered_ok": True,
                "fidelity_score": 98.123456789,
                "edge_f1": 0.9987654321,
                "path_count": 4,
                "byte_size": 2048,
                "private": object(),
            }
        ]

        snapshot = selection.alpha_parent_trial_snapshot(trials)

        self.assertEqual(
            snapshot,
            [
                {
                    "trial_index": 1,
                    "candidate_name": "flat-parent",
                    "candidate_engine": "vtracer",
                    "rendered_ok": True,
                    "fidelity_score": 98.12345679,
                    "edge_f1": 0.99876543,
                    "path_count": 4,
                    "byte_size": 2048,
                }
            ],
        )
        self.assertNotIn("svg_path", snapshot[0])
        self.assertNotIn("private", snapshot[0])

    def test_shortlist_includes_highest_fidelity_alternate_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def candidate(name: str, engine: str, fidelity: float, paths: int):
                svg = root / f"{name}.svg"
                svg.write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>',
                    encoding="utf-8",
                )
                return {
                    "name": name,
                    "engine": engine,
                    "svg_path": svg,
                    "rendered_ok": True,
                    "fidelity_score": fidelity,
                    "score_details": {
                        "path_count": paths,
                        "edge_f1": 0.9 + fidelity / 1000.0,
                        "has_bitmap": False,
                    },
                }

            gradient = candidate("gradient", "gradient", 92.5, 2)
            gradient_bnd = candidate("gradient_bnd", "gradient", 92.4, 3)
            gradient_refit = candidate("gradient_refit", "gradient", 92.3, 4)
            gradient_extra = candidate("gradient_extra", "gradient", 91.0, 5)
            flat_best = candidate("flat_best", "vtracer", 55.0, 16)
            flat_lower = candidate("flat_lower", "vtracer", 50.0, 8)
            result = {
                "best": gradient,
                "scored": [
                    gradient,
                    gradient_bnd,
                    gradient_refit,
                    gradient_extra,
                    flat_best,
                    flat_lower,
                ],
            }

            legacy = selection._legacy_shortlist(result)
            self.assertTrue(
                all(str(item["engine"]) == "gradient" for item in legacy),
                legacy,
            )

            shortlist = selection.engine_diverse_shortlist(result)

            self.assertIs(shortlist[0], gradient)
            self.assertIn(flat_best, shortlist)
            self.assertNotIn(flat_lower, shortlist)
            self.assertLessEqual(len(shortlist), selection._base._MAX_TRIALS)
            self.assertGreaterEqual(
                len({str(item["engine"]) for item in shortlist}),
                2,
            )

    def test_quality_rule_and_diagnostic_attachment_remain_explicit(self) -> None:
        dense = {
            "candidate": {"name": "dense", "engine": "gradient"},
            "rendered_ok": True,
            "fidelity_score": 99.9,
            "edge_f1": 0.99,
            "path_count": 2,
            "byte_size": 1000,
        }
        flat = {
            "candidate": {"name": "flat", "engine": "vtracer"},
            "rendered_ok": True,
            "fidelity_score": 99.7,
            "edge_f1": 1.0,
            "path_count": 4,
            "byte_size": 2000,
        }

        chosen = selection._base.choose_alpha_parent_trial([dense, flat])

        self.assertIs(chosen, dense)
        result: dict = {}
        returned = selection._attach_diagnostics(
            result,
            status="original_parent_retained",
            trials=[dense, flat],
            chosen=chosen,
        )
        self.assertIs(returned, result)
        diagnostics = result["alpha_parent_trial_diagnostics"]
        self.assertEqual(diagnostics["status"], "original_parent_retained")
        self.assertEqual(diagnostics["trial_count"], 2)
        self.assertEqual(diagnostics["chosen_candidate_name"], "dense")
        self.assertEqual(
            diagnostics["shortlist_policy"],
            "legacy_plus_highest_fidelity_alternate_engine",
        )

    def test_root_cause_snapshot_captures_alpha_trial_evidence(self) -> None:
        output = {
            "mode_used": "logo_color",
            "selection_reason": "highest_fidelity",
            "analysis": {},
            "preprocess_report": {},
            "scored": [],
            "alpha_parent_trial_diagnostics": {
                "schema": "vektoryum-alpha-parent-trial-diagnostics-v2",
                "shortlist_policy": "legacy_plus_highest_fidelity_alternate_engine",
                "status": "original_parent_retained",
                "shortlist_count": 2,
                "shortlist": [
                    {
                        "shortlist_index": 1,
                        "candidate_name": "gradient",
                        "candidate_engine": "gradient",
                        "fidelity_score_before_alpha": 92.5,
                        "edge_f1_before_alpha": 0.91,
                        "path_count_before_alpha": 2,
                    }
                ],
                "trial_count": 1,
                "chosen_candidate_name": "gradient",
                "failures": [
                    {
                        "stage": "alpha_transform",
                        "shortlist_index": 2,
                        "candidate_name": "flat",
                        "candidate_engine": "vtracer",
                        "error_class": "ValueError",
                        "error_message": "safe message",
                    }
                ],
                "trials": [
                    {
                        "trial_index": 1,
                        "candidate_name": "gradient",
                        "candidate_engine": "gradient",
                        "rendered_ok": True,
                        "fidelity_score": 92.5,
                        "edge_f1": 0.91,
                        "path_count": 2,
                        "byte_size": 5000,
                        "private": "/tmp/no-leak",
                    }
                ],
            },
        }

        snapshot = pipeline_snapshot(output)

        alpha = snapshot["alpha_parent_trials"]
        self.assertEqual(alpha["status"], "original_parent_retained")
        self.assertEqual(alpha["chosen_candidate_name"], "gradient")
        self.assertEqual(
            alpha["shortlist_policy"],
            "legacy_plus_highest_fidelity_alternate_engine",
        )
        self.assertEqual(alpha["shortlist"][0]["candidate_engine"], "gradient")
        self.assertEqual(alpha["failures"][0]["stage"], "alpha_transform")
        self.assertEqual(alpha["trials"][0]["candidate_engine"], "gradient")
        self.assertNotIn("private", alpha["trials"][0])


if __name__ == "__main__":
    unittest.main()
