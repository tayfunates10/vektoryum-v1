import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS
from engine.regression.rfv3_measurement_runner import (
    CATEGORY_MAP,
    EXPECTED_CASES_SHA256,
    MeasurementError,
    ROOT,
    load_qualification_cases,
    load_json,
    run_measurement,
    run_repeated_case,
    validate_policy,
)

POLICY_PATH = ROOT / "engine" / "regression" / "rfv3_measurement_policy.json"
QUALIFICATION_MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"


def metrics(value=0.99):
    return {
        "fidelity": value,
        "ssim": value,
        "edge_f1": value,
        "alpha_iou": value,
        "delta_e00": 1.0,
        "path_count": 10,
        "svg_bytes": 1000,
        "render_ms": 50.0,
        "peak_rss_mb": 100.0,
    }


def result(case_id, *, sha="a" * 64, value=0.99):
    return BenchmarkResult(
        case_id=case_id,
        engine_version="test-engine",
        metrics=metrics(value),
        artifact_sha256=sha,
    )


def create_extracted_bundle(root):
    manifest = load_json(QUALIFICATION_MANIFEST_PATH)
    (root / "objects").mkdir(parents=True)
    for case in manifest["cases"]:
        path = root / "objects" / case["storage_object_id"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"external-rfv2-object")
    (root / "qualification-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "bundle-index.json").write_text(
        json.dumps(
            {
                "schema": "vektoryum-rfv2-live-bundle-index-v1",
                "qualified_case_count": 24,
                "cases_sha256": EXPECTED_CASES_SHA256,
                "raw_assets_in_repository": False,
                "files": [],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest


class RFV3MeasurementRunnerTests(unittest.TestCase):
    def test_policy_is_exact_finite_and_preserves_unmeasured_values(self):
        policy = validate_policy(load_json(POLICY_PATH))
        self.assertEqual(policy["expected_case_count"], 24)
        self.assertEqual(policy["expected_cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(policy["repeat_count"], 3)
        self.assertEqual(policy["max_transient_retries_per_repeat"], 1)
        self.assertEqual(set(policy["required_metrics"]), set(REQUIRED_METRICS))
        self.assertEqual(policy["unmeasured_metric_policy"], "preserve_null_never_fabricate")
        self.assertFalse(policy["raw_assets_in_repository"])
        self.assertTrue(policy["phase_completion_requires_live_results"])

    def test_loads_exact_24_case_bundle_and_maps_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = create_extracted_bundle(root)
            with patch(
                "engine.regression.rfv3_measurement_runner.sha256_file",
                side_effect=lambda path: Path(path).stem,
            ):
                cases = load_qualification_cases(root)
        self.assertEqual(len(cases), 24)
        self.assertEqual([case.case_id for case in cases], sorted(case["case_id"] for case in manifest["cases"]))
        mapped = {case.case_id: case.category for case in cases}
        for case in manifest["cases"]:
            self.assertEqual(mapped[case["case_id"]], CATEGORY_MAP[case["category"]])
        self.assertTrue(all("rfv2" in case.tags for case in cases))

    def test_bundle_identity_digest_and_repository_boundary_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = create_extracted_bundle(root)
            with self.assertRaisesRegex(MeasurementError, "source digest mismatch"):
                load_qualification_cases(root)

            tampered = copy.deepcopy(manifest)
            tampered["cases"][0]["category"] = "flat_logo" if tampered["cases"][0]["category"] != "flat_logo" else "badge_seal"
            (root / "qualification-manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
            with patch("engine.regression.rfv3_measurement_runner.sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(MeasurementError, "case-set digest mismatch"):
                    load_qualification_cases(root)

        with self.assertRaisesRegex(MeasurementError, "outside the repository"):
            load_qualification_cases(ROOT)

    def test_transient_timeout_is_retried_once_and_audited(self):
        case = BenchmarkCase("case-1", "logos", "case.png", "cc0", "1" * 64)
        calls = []

        def fake_runner(case, **kwargs):
            calls.append(kwargs["work_root"])
            if len(calls) == 1:
                raise TimeoutError("transient timeout")
            return result(case.case_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregated, audit = run_repeated_case(
                case,
                corpus_root=root,
                work_root=root / "work",
                engine_version="test-engine",
                repeat_count=3,
                timeout_seconds=10,
                max_transient_retries=1,
                runner=fake_runner,
            )
        self.assertEqual(len(calls), 4)
        self.assertEqual(aggregated.case_id, "case-1")
        self.assertEqual(len(audit), 3)
        self.assertTrue(audit[0]["retried"])
        self.assertEqual(audit[0]["attempt_count"], 2)
        self.assertEqual(audit[0]["attempts"][0]["retry_class"], "TimeoutError")
        self.assertFalse(audit[1]["retried"])

    def test_non_retryable_and_exhausted_failures_stop_closed(self):
        case = BenchmarkCase("case-1", "logos", "case.png", "cc0", "1" * 64)
        non_retry_calls = []

        def non_retryable(case, **kwargs):
            non_retry_calls.append(1)
            raise ValueError("source digest mismatch")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(MeasurementError, "failed closed"):
                run_repeated_case(
                    case,
                    corpus_root=root,
                    work_root=root / "work",
                    engine_version="test-engine",
                    repeat_count=3,
                    timeout_seconds=10,
                    max_transient_retries=1,
                    runner=non_retryable,
                )
        self.assertEqual(len(non_retry_calls), 1)

        timeout_calls = []

        def exhausted(case, **kwargs):
            timeout_calls.append(1)
            raise TimeoutError("still unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(MeasurementError, "failed closed"):
                run_repeated_case(
                    case,
                    corpus_root=root,
                    work_root=root / "work",
                    engine_version="test-engine",
                    repeat_count=3,
                    timeout_seconds=10,
                    max_transient_retries=1,
                    runner=exhausted,
                )
        self.assertEqual(len(timeout_calls), 2)

    def test_non_deterministic_artifact_and_non_finite_metric_fail_closed(self):
        case = BenchmarkCase("case-1", "logos", "case.png", "cc0", "1" * 64)
        shas = iter(("a" * 64, "b" * 64, "a" * 64))

        def nondeterministic(case, **kwargs):
            return result(case.case_id, sha=next(shas))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(MeasurementError, "aggregation failed closed"):
                run_repeated_case(
                    case,
                    corpus_root=root,
                    work_root=root / "work",
                    engine_version="test-engine",
                    repeat_count=3,
                    timeout_seconds=10,
                    max_transient_retries=1,
                    runner=nondeterministic,
                )

        def invalid_metric(case, **kwargs):
            candidate = result(case.case_id)
            candidate.metrics["ssim"] = float("nan")
            return candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(MeasurementError, "non-finite"):
                run_repeated_case(
                    case,
                    corpus_root=root,
                    work_root=root / "work",
                    engine_version="test-engine",
                    repeat_count=3,
                    timeout_seconds=10,
                    max_transient_retries=1,
                    runner=invalid_metric,
                )

    def test_full_runner_writes_24_results_and_72_repeat_audits(self):
        cases = [
            BenchmarkCase(f"case-{index:02d}", "logos", f"case-{index:02d}.png", "cc0", f"{index + 1:064x}")
            for index in range(24)
        ]

        def fake_runner(case, **kwargs):
            candidate = result(case.case_id)
            candidate.metrics["alpha_iou"] = None
            return candidate

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            output = root / "output"
            corpus.mkdir()
            with patch(
                "engine.regression.rfv3_measurement_runner.load_qualification_cases",
                return_value=cases,
            ):
                results = run_measurement(
                    corpus_root=corpus,
                    output_dir=output,
                    engine_version="test-engine",
                    runner=fake_runner,
                )
            payload = json.loads((output / "pipeline-results.json").read_text(encoding="utf-8"))
            retry = json.loads((output / "retry-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(len(results), 24)
        self.assertEqual(payload["case_count"], 24)
        self.assertEqual(payload["measurement_method"]["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(len(retry["samples"]), 72)
        self.assertEqual(retry["completed_case_count"], 24)
        self.assertTrue(all(sample["retried"] is False for sample in retry["samples"]))
        self.assertTrue(all(item["metrics"]["alpha_iou"] is None for item in payload["results"]))

    def test_roadmap_remains_pending_until_live_measurement_exists(self):
        roadmap = load_json(ROADMAP_PATH)
        phases = roadmap["phases"]
        self.assertEqual([phase["id"] for phase in phases], ["RFV-1", "RFV-2", "RFV-3", "RFV-4"])
        self.assertEqual([phase["status"] for phase in phases], ["merged", "merged", "pending", "pending"])
        self.assertEqual(phases[2]["preparation_evidence"], "docs/real_world_fidelity/rfv-3a.md")
        self.assertEqual(phases[2]["measurement_policy"], "engine/regression/rfv3_measurement_policy.json")
        self.assertTrue((ROOT / phases[2]["preparation_evidence"]).is_file())
        self.assertTrue((ROOT / phases[2]["measurement_policy"]).is_file())


if __name__ == "__main__":
    unittest.main()
