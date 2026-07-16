from __future__ import annotations

import unittest

from engine.regression.rfv3_metric_coverage import (
    MetricCoverageError,
    classify_missing_metrics,
    diagnose,
    load_results,
    verify_expected_diagnosis,
    DEFAULT_RESULTS,
)


class RFV3MetricCoverageTests(unittest.TestCase):
    def test_complete_exact_metrics_are_not_flagged(self) -> None:
        missing, reason = classify_missing_metrics({
            "ssim": 0.99,
            "edge_f1": 0.98,
            "alpha_iou": 1.0,
            "delta_e00": 0.5,
        })
        self.assertEqual(missing, [])
        self.assertIsNone(reason)

    def test_partial_report_signature_is_classified(self) -> None:
        missing, reason = classify_missing_metrics({
            "ssim": None,
            "edge_f1": None,
            "alpha_iou": 1.0,
            "delta_e00": 0.5,
        })
        self.assertEqual(missing, ["edge_f1", "ssim"])
        self.assertEqual(reason, "partial_quality_report_fallback")

    def test_unknown_gap_fails_closed_without_guessing(self) -> None:
        report = diagnose({"results": [{
            "case_id": "x",
            "metrics": {"ssim": None, "edge_f1": 0.9, "alpha_iou": 1.0, "delta_e00": 0.5},
        }]})
        self.assertTrue(report["fail_closed"])
        self.assertEqual(report["release_decision"], "no_go")
        self.assertEqual(report["cases"][0]["diagnosis"], "unclassified_required_metric_gap")

    def test_committed_evidence_has_expected_three_fallback_cases(self) -> None:
        report = diagnose(load_results(DEFAULT_RESULTS))
        verify_expected_diagnosis(report)
        self.assertEqual(report["missing_metric_case_count"], 3)

    def test_expected_diagnosis_rejects_case_drift(self) -> None:
        with self.assertRaises(MetricCoverageError):
            verify_expected_diagnosis({
                "cases": [],
                "fail_closed": True,
                "release_decision": "no_go",
                "rfv4_allowed": False,
            })


if __name__ == "__main__":
    unittest.main()
