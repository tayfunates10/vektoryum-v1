from __future__ import annotations

import unittest

from app.alpha_parent_selection import choose_alpha_parent_trial


class AlphaParentSelectionTests(unittest.TestCase):
    def test_prefers_materially_leaner_trial_inside_strict_quality_margin(self):
        dense = {
            "candidate": {"name": "dense"},
            "rendered_ok": True,
            "fidelity_score": 99.90,
            "edge_f1": 0.999,
            "path_count": 356,
            "byte_size": 81000,
        }
        lean = {
            "candidate": {"name": "lean"},
            "rendered_ok": True,
            "fidelity_score": 99.82,
            "edge_f1": 0.998,
            "path_count": 2,
            "byte_size": 5200,
        }
        self.assertIs(choose_alpha_parent_trial([dense, lean]), lean)

    def test_rejects_leaner_trial_outside_fidelity_margin(self):
        dense = {
            "candidate": {"name": "dense"},
            "rendered_ok": True,
            "fidelity_score": 99.90,
            "edge_f1": 0.999,
            "path_count": 356,
            "byte_size": 81000,
        }
        degraded = {
            "candidate": {"name": "degraded"},
            "rendered_ok": True,
            "fidelity_score": 99.70,
            "edge_f1": 0.999,
            "path_count": 1,
            "byte_size": 1000,
        }
        self.assertIs(choose_alpha_parent_trial([dense, degraded]), dense)

    def test_rejects_edge_regression_inside_fidelity_margin(self):
        dense = {
            "candidate": {"name": "dense"},
            "rendered_ok": True,
            "fidelity_score": 99.90,
            "edge_f1": 0.999,
            "path_count": 356,
            "byte_size": 81000,
        }
        edge_regression = {
            "candidate": {"name": "edge-regression"},
            "rendered_ok": True,
            "fidelity_score": 99.88,
            "edge_f1": 0.990,
            "path_count": 1,
            "byte_size": 1000,
        }
        self.assertIs(choose_alpha_parent_trial([dense, edge_regression]), dense)

    def test_ignores_unrendered_trials(self):
        valid = {
            "candidate": {"name": "valid"},
            "rendered_ok": True,
            "fidelity_score": 99.0,
            "edge_f1": 0.99,
            "path_count": 10,
            "byte_size": 2000,
        }
        invalid = {
            "candidate": {"name": "invalid"},
            "rendered_ok": False,
            "fidelity_score": 100.0,
            "edge_f1": 1.0,
            "path_count": 1,
            "byte_size": 100,
        }
        self.assertIs(choose_alpha_parent_trial([invalid, valid]), valid)


if __name__ == "__main__":
    unittest.main()
