from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_select_attaches_diagnostics_without_changing_choice(self) -> None:
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
        original_result = {"best": {"name": "dense"}}

        def fake_select(result, source_path, job_dir):
            chosen = selection._base.choose_alpha_parent_trial([dense, flat])
            result["chosen_by_base"] = chosen["candidate"]["name"]
            return result

        with patch.object(selection._base, "_select", side_effect=fake_select):
            returned = selection._select(
                original_result,
                Path("source.png"),
                Path("job"),
            )

        self.assertIs(returned, original_result)
        self.assertEqual(returned["chosen_by_base"], "dense")
        diagnostics = returned["alpha_parent_trial_diagnostics"]
        self.assertEqual(diagnostics["trial_count"], 2)
        self.assertEqual(diagnostics["chosen_candidate_name"], "dense")
        self.assertEqual(
            [item["candidate_name"] for item in diagnostics["trials"]],
            ["dense", "flat"],
        )

    def test_root_cause_snapshot_captures_alpha_trial_evidence(self) -> None:
        output = {
            "mode_used": "logo_color",
            "selection_reason": "highest_fidelity",
            "analysis": {},
            "preprocess_report": {},
            "scored": [],
            "alpha_parent_trial_diagnostics": {
                "schema": "vektoryum-alpha-parent-trial-diagnostics-v1",
                "trial_count": 1,
                "chosen_candidate_name": "gradient",
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
        self.assertEqual(alpha["chosen_candidate_name"], "gradient")
        self.assertEqual(alpha["trials"][0]["candidate_engine"], "gradient")
        self.assertNotIn("private", alpha["trials"][0])


if __name__ == "__main__":
    unittest.main()
