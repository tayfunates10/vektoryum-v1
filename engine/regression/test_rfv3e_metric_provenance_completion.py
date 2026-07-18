"""RFV-3E repeat-level exact evaluator provenance completion tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from benchmark.manifest import BenchmarkResult, REQUIRED_METRICS
from benchmark.pipeline_results import (
    EVALUATOR_DETAIL_SCHEMA,
    PROVENANCE_SCHEMA,
    _exact_winner_metrics,
)
from benchmark.pipeline_smoke import REPEAT_PROVENANCE_SCHEMA, aggregate_repeats


class _Report:
    def __init__(
        self,
        *,
        ssim=0.995,
        edge=0.996,
        de00=1.2,
        verdict="production_ready",
        hard_codes=(),
        soft_codes=(),
        unmeasured=(),
        groups=True,
    ):
        self.sha256 = "b" * 64
        self.byte_read_stable = True
        self.deterministic = None
        self.verdict = verdict
        self.hard_fail_codes = list(hard_codes)
        self.soft_warning_codes = list(soft_codes)
        self.unmeasured_required = list(unmeasured)
        self.metrics = {
            "A_structure": {"path_count": 1},
            "B_visual": {"ssim": ssim, "ms_ssim": None},
            "C_color": {"de00_mean": de00},
            "D_edge_geometry": {"edge_f1_1px": edge},
            "G_gradient_alpha": {"alpha_iou": 0.999},
            "H_editability": {"path_count": 1},
        } if groups else {"A_structure": {"path_count": 1}}


def _svg(root: Path) -> Path:
    path = root / "winner.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        '<rect width="8" height="8" fill="#f00"/></svg>',
        encoding="utf-8",
    )
    return path


def _call(path: Path, report: _Report):
    output = {"best": {"svg_path": str(path), "fidelity_score": 99.5}}
    source = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    with patch("benchmark.pipeline_results.evaluate_final_svg", return_value=report), patch(
        "benchmark.pipeline_results.render_svg_to_rgba", return_value=None
    ):
        return _exact_winner_metrics(output, source, elapsed_ms=5.0, peak_rss_mb=10.0)


def _result(provenance: dict, *, artifact="b" * 64) -> BenchmarkResult:
    metrics = {name: 1.0 for name in REQUIRED_METRICS}
    metrics["path_count"] = 1
    metrics["svg_bytes"] = 100
    return BenchmarkResult(
        case_id="qualification-public-10",
        engine_version="test",
        metrics=metrics,
        artifact_sha256=artifact,
        metric_provenance=provenance,
    )


class RFV3EProvenanceCompletionTests(unittest.TestCase):
    def test_success_publishes_sanitized_evaluator_report_detail(self):
        with TemporaryDirectory() as temp:
            metrics, artifact, provenance = _call(_svg(Path(temp)), _Report())
        self.assertEqual(provenance["schema"], PROVENANCE_SCHEMA)
        self.assertEqual(provenance["exact_evaluator_detail_schema"], EVALUATOR_DETAIL_SCHEMA)
        self.assertEqual(provenance["exact_evaluator_report_status"], "returned")
        self.assertEqual(provenance["exact_evaluator_reason_code"], "exact_metrics_complete")
        self.assertEqual(provenance["exact_evaluator_render_outcome"], "rendered")
        self.assertEqual(provenance["exact_evaluator_verdict"], "production_ready")
        self.assertTrue(provenance["exact_evaluator_byte_read_stable"])
        self.assertEqual(len(provenance["exact_evaluator_report_summary_sha256"]), 64)
        self.assertTrue(all(provenance["exact_evaluator_metric_group_presence"].values()))
        self.assertEqual(
            provenance["exact_evaluator_component_status"],
            {"ssim": "finite", "edge_f1": "finite", "delta_e00": "finite"},
        )
        self.assertEqual(provenance["exact_evaluator_missing_component_metrics"], [])
        self.assertEqual(artifact, "b" * 64)
        self.assertIsNotNone(metrics["ssim"])

    def test_missing_groups_are_explicit_and_fail_closed(self):
        with TemporaryDirectory() as temp:
            metrics, artifact, provenance = _call(
                _svg(Path(temp)),
                _Report(groups=False, verdict="failed", hard_codes=("render_failed",)),
            )
        self.assertEqual(provenance["exact_evaluator_failure_class"], "render_failure")
        self.assertEqual(provenance["exact_evaluator_reason_code"], "render_failed")
        self.assertEqual(provenance["exact_evaluator_render_outcome"], "failed")
        self.assertEqual(
            provenance["exact_evaluator_missing_component_metrics"],
            ["delta_e00", "edge_f1", "ssim"],
        )
        self.assertFalse(provenance["exact_evaluator_metric_group_presence"]["B_visual"])
        self.assertEqual(provenance["exact_evaluator_component_status"]["ssim"], "missing")
        self.assertTrue(provenance["fallback_used"])
        self.assertEqual(artifact, "b" * 64)
        self.assertIsNone(metrics["ssim"])

    def test_report_summary_digest_is_deterministic(self):
        with TemporaryDirectory() as temp:
            path = _svg(Path(temp))
            first = _call(path, _Report())[2]
            second = _call(path, _Report())[2]
        self.assertEqual(
            first["exact_evaluator_report_summary_sha256"],
            second["exact_evaluator_report_summary_sha256"],
        )

    def test_aggregate_retains_three_repeat_provenance_rows(self):
        with TemporaryDirectory() as temp:
            provenance = _call(_svg(Path(temp)), _Report())[2]
        provenance["artifact_sha256"] = "b" * 64
        merged = aggregate_repeats([_result(dict(provenance)) for _ in range(3)])
        aggregate = merged.metric_provenance
        self.assertIsNotNone(aggregate)
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["repeat_provenance_schema"], REPEAT_PROVENANCE_SCHEMA)
        rows = aggregate["repeat_provenance"]
        self.assertEqual([row["repeat_index"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["artifact_sha256"] == "b" * 64 for row in rows))
        self.assertTrue(all(row["exact_evaluator_report_status"] == "returned" for row in rows))
        self.assertTrue(all("repeat_provenance" not in row for row in rows))

    def test_repeat_evaluator_detail_drift_fails_closed(self):
        with TemporaryDirectory() as temp:
            provenance = _call(_svg(Path(temp)), _Report())[2]
        provenance["artifact_sha256"] = "b" * 64
        drifted = json.loads(json.dumps(provenance))
        drifted["exact_evaluator_reason_code"] = "different_reason"
        with self.assertRaisesRegex(ValueError, "non-deterministic metric provenance"):
            aggregate_repeats([_result(dict(provenance)), _result(drifted)])

    def test_repeat_provenance_artifact_mismatch_fails_closed(self):
        with TemporaryDirectory() as temp:
            provenance = _call(_svg(Path(temp)), _Report())[2]
        provenance["artifact_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "provenance artifact mismatch"):
            aggregate_repeats([_result(dict(provenance)) for _ in range(3)])

    def test_release_decision_and_rfv4_block_are_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        decision = json.loads(
            (root / "docs/real_world_fidelity/evidence/rfv3_quality_decision.json").read_text(encoding="utf-8")
        )
        roadmap = json.loads((root / "docs/real_world_fidelity_roadmap.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["release_decision"], "no_go")
        self.assertIs(decision["rfv4_allowed"], False)
        self.assertEqual(roadmap["phases"][3]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
