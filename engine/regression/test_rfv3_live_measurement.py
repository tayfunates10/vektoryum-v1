import hashlib
import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS
from benchmark.pipeline_results import write_results
from engine.regression.rfv3_live_measurement import (
    DEFAULT_SHARD_COUNT,
    EXPECTED_CASE_COUNT,
    LiveMeasurementError,
    aggregate_shards,
    partition_cases,
    run_shard,
    safe_extract_bundle,
)
from engine.regression.rfv3_measurement_runner import (
    EXPECTED_CASES_SHA256,
    QUALIFICATION_MANIFEST_PATH,
    load_json,
)


def _metrics(value=0.99):
    return {
        "fidelity": value,
        "ssim": value,
        "edge_f1": value,
        "alpha_iou": value,
        "delta_e00": 1.0,
        "path_count": 12,
        "svg_bytes": 2048,
        "render_ms": 100.0,
        "peak_rss_mb": 200.0,
    }


def _result(case_id, engine_version="test-engine"):
    return BenchmarkResult(
        case_id=case_id,
        engine_version=engine_version,
        metrics=_metrics(),
        artifact_sha256=hashlib.sha256(case_id.encode("utf-8")).hexdigest(),
    )


def _retry_samples(case_ids):
    return [
        {
            "case_id": case_id,
            "repeat_index": repeat_index,
            "attempt_count": 1,
            "retried": False,
            "status": "success",
            "attempts": [
                {
                    "attempt": 1,
                    "status": "success",
                    "retry_class": None,
                    "error": None,
                }
            ],
        }
        for case_id in case_ids
        for repeat_index in range(1, 4)
    ]


def _case_ids():
    manifest = load_json(QUALIFICATION_MANIFEST_PATH)
    return sorted(case["case_id"] for case in manifest["cases"])


class RFV3LiveMeasurementTests(unittest.TestCase):
    def test_partition_is_complete_unique_and_balanced(self):
        cases = [
            BenchmarkCase(case_id, "logos", f"{case_id}.png", "cc0", f"{index + 1:064x}")
            for index, case_id in enumerate(_case_ids())
        ]
        shards = [partition_cases(cases, shard_index=index, shard_count=DEFAULT_SHARD_COUNT) for index in range(DEFAULT_SHARD_COUNT)]
        self.assertEqual([len(shard) for shard in shards], [4] * DEFAULT_SHARD_COUNT)
        flattened = [case.case_id for shard in shards for case in shard]
        self.assertEqual(set(flattened), set(_case_ids()))
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaisesRegex(LiveMeasurementError, "shard index"):
            partition_cases(cases, shard_index=DEFAULT_SHARD_COUNT, shard_count=DEFAULT_SHARD_COUNT)

    def test_safe_extract_rejects_digest_mismatch_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "bundle.tar.gz"
            with tarfile.open(bundle, "w:gz") as archive:
                payload = b"bad"
                info = tarfile.TarInfo("../escape")
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
            checksums = root / "checksums.json"
            checksums.write_text(
                json.dumps(
                    {
                        "schema": "vektoryum-rfv2-live-bundle-checksums-v1",
                        "qualified_case_count": EXPECTED_CASE_COUNT,
                        "cases_sha256": EXPECTED_CASES_SHA256,
                        "bundle_sha256": "0" * 64,
                        "raw_assets_in_repository": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LiveMeasurementError, "digest mismatch"):
                safe_extract_bundle(bundle=bundle, checksums=checksums, destination=root / "extract-a")
            payload = json.loads(checksums.read_text())
            payload["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
            checksums.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LiveMeasurementError, "unsafe"):
                safe_extract_bundle(bundle=bundle, checksums=checksums, destination=root / "extract-b")

    def test_shard_runner_writes_four_results_and_twelve_repeat_samples(self):
        cases = [
            BenchmarkCase(case_id, "logos", f"{case_id}.png", "cc0", f"{index + 1:064x}")
            for index, case_id in enumerate(_case_ids())
        ]

        def fake_runner(case, **kwargs):
            return _result(case.case_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            output = root / "output"
            corpus.mkdir()
            with patch("engine.regression.rfv3_live_measurement.load_qualification_cases", return_value=cases):
                results = run_shard(
                    corpus_root=corpus,
                    output_dir=output,
                    engine_version="test-engine",
                    shard_index=0,
                    shard_count=DEFAULT_SHARD_COUNT,
                    runner=fake_runner,
                )
            result_payload = json.loads((output / "pipeline-results.json").read_text())
            retry_payload = json.loads((output / "retry-audit.json").read_text())
            summary = json.loads((output / "shard-summary.json").read_text())
        self.assertEqual(len(results), 4)
        self.assertEqual(result_payload["case_count"], 4)
        self.assertEqual(len(retry_payload["samples"]), 12)
        self.assertEqual(summary["case_ids"], _case_ids()[0::DEFAULT_SHARD_COUNT])

    def test_aggregate_requires_exact_24_results_and_72_retry_samples(self):
        engine_version = "a" * 40
        case_ids = _case_ids()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output = root / "output"
            input_root.mkdir()
            for shard_index in range(DEFAULT_SHARD_COUNT):
                selected = case_ids[shard_index::DEFAULT_SHARD_COUNT]
                shard = input_root / f"shard-{shard_index}"
                shard.mkdir()
                write_results(
                    shard / "pipeline-results.json",
                    [_result(case_id, engine_version=engine_version) for case_id in selected],
                    measurement_method={
                        "schema": "vektoryum-rfv3-shard-measurement-v1",
                        "cases_sha256": EXPECTED_CASES_SHA256,
                        "expected_case_count": EXPECTED_CASE_COUNT,
                        "selected_case_count": len(selected),
                        "shard_index": shard_index,
                        "shard_count": DEFAULT_SHARD_COUNT,
                        "repeat_count": 3,
                    },
                )
                (shard / "retry-audit.json").write_text(
                    json.dumps(
                        {
                            "schema": "vektoryum-rfv3-shard-retry-audit-v1",
                            "expected_case_count": EXPECTED_CASE_COUNT,
                            "completed_case_count": len(selected),
                            "repeat_count": 3,
                            "max_transient_retries_per_repeat": 1,
                            "shard_index": shard_index,
                            "shard_count": DEFAULT_SHARD_COUNT,
                            "samples": _retry_samples(selected),
                        }
                    ),
                    encoding="utf-8",
                )
                (shard / "shard-summary.json").write_text(
                    json.dumps(
                        {
                            "schema": "vektoryum-rfv3-shard-summary-v1",
                            "engine_version": engine_version,
                            "cases_sha256": EXPECTED_CASES_SHA256,
                            "expected_case_count": EXPECTED_CASE_COUNT,
                            "shard_index": shard_index,
                            "shard_count": DEFAULT_SHARD_COUNT,
                            "case_ids": selected,
                            "result_count": len(selected),
                            "repeat_sample_count": len(selected) * 3,
                            "raw_assets_in_repository": False,
                        }
                    ),
                    encoding="utf-8",
                )
            envelope = aggregate_shards(
                input_root=input_root,
                output_dir=output,
                engine_version=engine_version,
                shard_count=DEFAULT_SHARD_COUNT,
            )
            combined = json.loads((output / "pipeline-results.json").read_text())
            retry = json.loads((output / "retry-audit.json").read_text())
            self.assertEqual(envelope["case_count"], EXPECTED_CASE_COUNT)
            self.assertEqual(combined["case_count"], EXPECTED_CASE_COUNT)
            self.assertEqual(retry["sample_count"], 72)
            self.assertEqual({item["case_id"] for item in combined["results"]}, set(case_ids))

            (input_root / "shard-5" / "shard-summary.json").unlink()
            with self.assertRaisesRegex(LiveMeasurementError, "artifact count"):
                aggregate_shards(
                    input_root=input_root,
                    output_dir=root / "rejected-output",
                    engine_version=engine_version,
                    shard_count=DEFAULT_SHARD_COUNT,
                )

    def test_required_metric_set_is_finite_or_null(self):
        self.assertEqual(set(_metrics()), set(REQUIRED_METRICS))


if __name__ == "__main__":
    unittest.main()
