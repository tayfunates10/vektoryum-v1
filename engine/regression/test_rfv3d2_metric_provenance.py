"""RFV-3D2: exact-winner metric path provenance instrumentation tests.

Proves at runtime which path produced each benchmark metric row, that every
fallback is recorded (never silent), that missing metrics stay ``None`` and
that published evidence contains no raw filesystem paths. No production
quality algorithm, threshold or corpus change is exercised here.
"""
from __future__ import annotations

import hashlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS
from benchmark.pipeline_results import (
    PROVENANCE_SCHEMA,
    _exact_winner_metrics,
    _sanitize_failure_message,
    run_case,
)
from benchmark.pipeline_smoke import aggregate_repeats


class _FakeReport:
    def __init__(self, *, ssim=0.99, edge=0.98, de00=1.2, hard_codes=()):
        self.sha256 = "b" * 64
        self.hard_fail_codes = list(hard_codes)
        self.metrics = {
            "B_visual": {"ssim": ssim, "ms_ssim": None},
            "C_color": {"de00_mean": de00},
            "D_edge_geometry": {"edge_f1_1px": edge},
            "H_editability": {"path_count": 7},
            "G_gradient_alpha": {"alpha_iou": 0.997},
        }


def _source_rgba(size=8):
    return Image.new("RGBA", (size, size), (255, 0, 0, 255))


def _write_svg(directory: Path) -> Path:
    svg = directory / "winner.svg"
    svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><rect width="8" height="8" fill="#f00"/></svg>')
    return svg


def _winner(output_svg: Path | None, extra: dict | None = None) -> dict:
    output: dict = {"best": {}}
    if output_svg is not None:
        output["best"] = {"svg_path": str(output_svg), "fidelity_score": 99.0}
    if extra:
        output.update(extra)
    return output


def _call(output, **patches):
    with TemporaryDirectory() as tmp:
        del tmp
        with patch("benchmark.pipeline_results.evaluate_final_svg", **patches.get("evaluator", {"return_value": _FakeReport()})), \
             patch("benchmark.pipeline_results.render_svg_to_rgba", return_value=None):
            return _exact_winner_metrics(output, _source_rgba(), elapsed_ms=10.0, peak_rss_mb=100.0)


class RFV3D2ProvenanceTests(unittest.TestCase):
    def test_exact_path_success_records_completed_provenance(self):
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            metrics, exact_sha, prov = _call(_winner(svg))
        self.assertEqual(prov["schema"], PROVENANCE_SCHEMA)
        self.assertEqual(prov["metric_source"], "exact_final_artifact")
        self.assertTrue(prov["exact_evaluator_attempted"])
        self.assertTrue(prov["exact_evaluator_completed"])
        self.assertIsNone(prov["exact_evaluator_failure_class"])
        self.assertTrue(prov["selected_svg_path_present"])
        self.assertTrue(prov["selected_svg_file_present"])
        self.assertFalse(prov["fallback_used"])
        self.assertIsNone(prov["fallback_source"])
        self.assertEqual(exact_sha, "b" * 64)
        for name in ("ssim", "edge_f1", "delta_e00"):
            self.assertIsNotNone(metrics[name])
        self.assertEqual(set(metrics), set(REQUIRED_METRICS))

    def test_selected_svg_sha_matches_file_bytes(self):
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            expected = hashlib.sha256(svg.read_bytes()).hexdigest()
            _metrics, _sha, prov = _call(_winner(svg))
        self.assertEqual(prov["selected_svg_sha256"], expected)

    def test_missing_svg_path_falls_back_with_class(self):
        metrics, exact_sha, prov = _call(_winner(None))
        self.assertEqual(prov["exact_evaluator_failure_class"], "selected_svg_path_missing")
        self.assertFalse(prov["exact_evaluator_attempted"])
        self.assertFalse(prov["selected_svg_path_present"])
        self.assertTrue(prov["fallback_used"])
        self.assertEqual(prov["metric_source"], "partial_quality_report")
        self.assertIsNone(exact_sha)
        self.assertIsNone(metrics["ssim"])  # null korunur; uydurma yok
        self.assertIsNone(metrics["edge_f1"])

    def test_missing_svg_file_falls_back_with_class(self):
        with TemporaryDirectory() as tmp:
            ghost = Path(tmp) / "missing.svg"
            _metrics, _sha, prov = _call(_winner(ghost))
        self.assertEqual(prov["exact_evaluator_failure_class"], "selected_svg_file_missing")
        self.assertTrue(prov["selected_svg_path_present"])
        self.assertFalse(prov["selected_svg_file_present"])
        self.assertTrue(prov["fallback_used"])
        self.assertFalse(prov["exact_evaluator_completed"])

    def test_evaluator_exception_is_recorded_and_sanitized(self):
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            metrics, _sha, prov = _call(
                _winner(svg),
                evaluator={"side_effect": RuntimeError(f"boom at {svg} object 0x7fab12cd")},
            )
        self.assertEqual(prov["exact_evaluator_failure_class"], "evaluator_exception")
        self.assertTrue(prov["exact_evaluator_attempted"])
        self.assertFalse(prov["exact_evaluator_completed"])
        message = prov["exact_evaluator_failure_message_sanitized"]
        self.assertIn("<redacted-path>", message)
        self.assertIn("<addr>", message)
        self.assertNotIn(str(svg), message)
        self.assertTrue(prov["fallback_used"])
        self.assertIsNone(metrics["ssim"])

    def test_render_failure_class_from_hard_codes(self):
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            _metrics, _sha, prov = _call(
                _winner(svg),
                evaluator={"return_value": _FakeReport(ssim=None, edge=None, de00=None, hard_codes=("render_failed",))},
            )
        self.assertEqual(prov["exact_evaluator_failure_class"], "render_failure")
        self.assertFalse(prov["exact_evaluator_completed"])
        self.assertTrue(prov["fallback_used"])

    def test_partial_exact_metrics_fail_closed(self):
        # evaluator döndü ama edge_f1 yok → completed sayılmaz, sınıf açık
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            metrics, exact_sha, prov = _call(
                _winner(svg),
                evaluator={"return_value": _FakeReport(edge=None)},
            )
        self.assertEqual(prov["exact_evaluator_failure_class"], "exact_metrics_incomplete")
        self.assertFalse(prov["exact_evaluator_completed"])
        self.assertEqual(prov["metric_source"], "partial_quality_report")
        self.assertIn("edge_f1", prov["exact_evaluator_failure_message_sanitized"])
        self.assertIsNone(exact_sha)
        self.assertIsNone(metrics["edge_f1"])  # fallback boş output'tan → null

    def test_no_branch_completes_without_finite_components(self):
        # fail-closed: hangi dal olursa olsun completed=False iken source exact olamaz
        cases = [
            _call(_winner(None)),
        ]
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            cases.append(_call(_winner(svg), evaluator={"return_value": _FakeReport(ssim=None)}))
            cases.append(_call(_winner(svg), evaluator={"side_effect": ValueError("x")}))
        for _metrics, _sha, prov in cases:
            self.assertFalse(prov["exact_evaluator_completed"])
            self.assertNotEqual(prov["metric_source"], "exact_final_artifact")
            self.assertTrue(prov["fallback_used"])
            self.assertIsNotNone(prov["exact_evaluator_failure_class"])

    def test_sanitizer_redacts_paths_and_addresses(self):
        raw = "failed /home/runner/work/repo/job/file.svg and C:\\Users\\ci\\a\\b.svg at 0xDEADBEEF"
        clean = _sanitize_failure_message(raw)
        self.assertNotIn("/home/", clean)
        self.assertNotIn("\\Users\\", clean)
        self.assertNotIn("0xDEADBEEF", clean)
        self.assertLessEqual(len(clean), 200)

    def test_run_case_binds_provenance_to_artifact_sha(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus").mkdir()
            png = io.BytesIO()
            Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(png, "PNG")
            source = root / "corpus" / "case.png"
            source.write_bytes(png.getvalue())
            case = BenchmarkCase(
                case_id="rfv3d2-unit-1",
                category="logos",
                source_path="case.png",
                license_id="cc0",
                source_sha256=hashlib.sha256(png.getvalue()).hexdigest(),
            )
            svg = _write_svg(root)

            def fake_pipeline(image, src, mode, job_dir):
                return _winner(svg)

            with patch("benchmark.pipeline_results.evaluate_final_svg", return_value=_FakeReport()), \
                 patch("benchmark.pipeline_results.render_svg_to_rgba", return_value=None):
                result = run_case(
                    case,
                    corpus_root=root / "corpus",
                    work_root=root / "work",
                    pipeline=fake_pipeline,
                    engine_version="unit-test",
                )
        prov = result.metric_provenance
        self.assertIsNotNone(prov)
        self.assertEqual(prov["artifact_sha256"], result.artifact_sha256)
        self.assertEqual(result.artifact_sha256, "b" * 64)
        self.assertEqual(prov["metric_source"], "exact_final_artifact")
        # rapor şeması: to_dict deterministik serileşir ve provenance taşır
        row = result.to_dict()
        self.assertEqual(row["metric_provenance"]["schema"], PROVENANCE_SCHEMA)

    def test_provenance_decisions_are_deterministic(self):
        with TemporaryDirectory() as tmp:
            svg = _write_svg(Path(tmp))
            _m1, _s1, prov1 = _call(_winner(svg))
            _m2, _s2, prov2 = _call(_winner(svg))
        self.assertEqual(json.dumps(prov1, sort_keys=True), json.dumps(prov2, sort_keys=True))

    def test_backward_compatible_result_schema(self):
        legacy_row = {
            "case_id": "legacy",
            "engine_version": "v0",
            "metrics": {name: None for name in REQUIRED_METRICS},
            "artifact_sha256": None,
            "failure": None,
        }
        result = BenchmarkResult(**legacy_row)   # provenance alanı yok → geçerli
        self.assertIsNone(result.metric_provenance)
        self.assertIsNone(result.to_dict()["metric_provenance"])
        with self.assertRaises(ValueError):
            BenchmarkResult(
                case_id="bad", engine_version="v0",
                metrics={name: None for name in REQUIRED_METRICS},
                metric_provenance="not-a-dict",  # type: ignore[arg-type]
            ).validate()

    def test_aggregate_repeats_carries_and_gates_provenance(self):
        def _result(prov):
            return BenchmarkResult(
                case_id="agg", engine_version="v1",
                metrics={name: (1.0 if name != "path_count" else 3) for name in REQUIRED_METRICS},
                artifact_sha256="c" * 64,
                metric_provenance=prov,
            )

        base_prov = {
            "schema": PROVENANCE_SCHEMA,
            "metric_source": "exact_final_artifact",
            "exact_evaluator_attempted": True,
            "exact_evaluator_completed": True,
            "exact_evaluator_failure_class": None,
            "exact_evaluator_failure_message_sanitized": None,
            "selected_svg_path_present": True,
            "selected_svg_file_present": True,
            "selected_svg_sha256": "d" * 64,
            "fallback_used": False,
            "fallback_source": None,
            "artifact_sha256": "c" * 64,
        }
        merged = aggregate_repeats([_result(dict(base_prov)) for _ in range(3)])
        self.assertIsNotNone(merged.metric_provenance)
        self.assertEqual(merged.metric_provenance["repeat_count"], 3)
        self.assertEqual(merged.metric_provenance["metric_source"], "exact_final_artifact")

        drifted = dict(base_prov)
        drifted["metric_source"] = "partial_quality_report"
        with self.assertRaises(ValueError):
            aggregate_repeats([_result(dict(base_prov)), _result(drifted)])
        with self.assertRaises(ValueError):
            aggregate_repeats([_result(dict(base_prov)), _result(None)])

    def test_release_stays_no_go_and_rfv4_blocked(self):
        root = Path(__file__).resolve().parents[2]
        decision = json.loads((root / "docs/real_world_fidelity/evidence/rfv3_quality_decision.json").read_text(encoding="utf-8"))
        roadmap = json.loads((root / "docs/real_world_fidelity_roadmap.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["release_decision"], "no_go")
        self.assertIs(decision["rfv4_allowed"], False)
        self.assertEqual(roadmap["phases"][3]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
