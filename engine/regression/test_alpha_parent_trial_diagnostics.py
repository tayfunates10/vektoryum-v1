from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import app.alpha_parent_selection as selection
from app import alpha_candidate_knockout as knockout
from app.alpha_candidate_knockout_compact import build_compact_knockout_reconstruction_tree
from engine.regression.output_quality_root_cause import pipeline_snapshot


class AlphaParentTrialDiagnosticTests(unittest.TestCase):
    def test_snapshot_is_json_safe(self) -> None:
        snapshot = selection.alpha_parent_trial_snapshot(
            [
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
        )
        self.assertEqual(snapshot[0]["candidate_name"], "flat-parent")
        self.assertEqual(snapshot[0]["fidelity_score"], 98.12345679)
        self.assertEqual(snapshot[0]["edge_f1"], 0.99876543)
        self.assertNotIn("svg_path", snapshot[0])
        self.assertNotIn("private", snapshot[0])

    def test_shortlist_includes_best_alternate_engine(self) -> None:
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
            self.assertTrue(all(item["engine"] == "gradient" for item in legacy))
            shortlist = selection.engine_diverse_shortlist(result)
            self.assertIs(shortlist[0], gradient)
            self.assertIn(flat_best, shortlist)
            self.assertNotIn(flat_lower, shortlist)
            self.assertLessEqual(len(shortlist), selection._base._MAX_TRIALS)
            self.assertGreaterEqual(len({item["engine"] for item in shortlist}), 2)

    def test_quality_rule_is_unchanged_and_diagnostics_are_explicit(self) -> None:
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
        selection._attach_diagnostics(
            result,
            status="original_parent_retained",
            trials=[dense, flat],
            chosen=chosen,
        )
        diagnostics = result["alpha_parent_trial_diagnostics"]
        self.assertEqual(diagnostics["status"], "original_parent_retained")
        self.assertEqual(diagnostics["trial_count"], 2)
        self.assertEqual(diagnostics["chosen_candidate_name"], "dense")

    def test_root_cause_snapshot_keeps_shortlist_trials_and_failures(self) -> None:
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
        alpha = pipeline_snapshot(output)["alpha_parent_trials"]
        self.assertEqual(alpha["status"], "original_parent_retained")
        self.assertEqual(alpha["shortlist"][0]["candidate_engine"], "gradient")
        self.assertEqual(alpha["failures"][0]["stage"], "alpha_transform")
        self.assertNotIn("private", alpha["trials"][0])


class CompactCandidateKnockoutTests(unittest.TestCase):
    def test_compact_encoder_reduces_bytes_and_preserves_path_geometry(self) -> None:
        namespace = "http://www.w3.org/2000/svg"
        qname = lambda name: f"{{{namespace}}}{name}"
        root = ET.Element(qname("svg"), {"viewBox": "0 0 64 64"})
        canvas = ET.SubElement(
            root,
            qname("path"),
            {"d": "M0 0H64V64H0Z", "fill": "#fff"},
        )
        ET.SubElement(
            root,
            qname("path"),
            {"d": "M8 8H56V56H8Z", "fill": "#1677ff"},
        )
        yy, xx = np.indices((64, 64))
        alpha = np.where((xx + yy) % 2 == 0, 128, 255).astype(np.uint8)
        opacity = {128: 128 / 255.0, 255: 1.0}

        legacy_root, legacy_stats = knockout._legacy_build_reconstruction_tree(
            root, canvas, alpha, opacity
        )
        compact_root, compact_stats = build_compact_knockout_reconstruction_tree(
            root, canvas, alpha, opacity
        )

        legacy_bytes = ET.tostring(legacy_root, encoding="utf-8")
        compact_bytes = ET.tostring(compact_root, encoding="utf-8")
        original_counts = knockout._path_node_counts(root)
        self.assertEqual(knockout._path_node_counts(legacy_root), original_counts)
        self.assertEqual(knockout._path_node_counts(compact_root), original_counts)
        self.assertEqual(
            compact_stats["reconstruction_rectangle_count"],
            legacy_stats["reconstruction_rectangle_count"],
        )
        self.assertLess(len(compact_bytes), int(len(legacy_bytes) * 0.80))
        use_nodes = [
            node
            for node in compact_root.iter()
            if str(node.tag).rsplit("}", 1)[-1] == "use"
        ]
        self.assertGreater(len(use_nodes), 0)
        self.assertEqual(
            compact_stats["reconstruction_encoding"],
            "compact-user-space-use-v1",
        )


if __name__ == "__main__":
    unittest.main()
