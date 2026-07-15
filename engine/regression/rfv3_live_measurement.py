from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS
from benchmark.pipeline_results import write_results
from engine.regression.rfv3_measurement_runner import (
    EXPECTED_CASES_SHA256,
    MeasurementError,
    POLICY_PATH,
    QUALIFICATION_MANIFEST_PATH,
    ROOT,
    load_json,
    load_qualification_cases,
    require_external_directory,
    require_external_output,
    run_repeated_case,
    sha256_file,
    validate_policy,
    write_json_atomic,
)

EXPECTED_CASE_COUNT = 24
DEFAULT_SHARD_COUNT = 6


class LiveMeasurementError(RuntimeError):
    pass


def _external_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise LiveMeasurementError(f"{label} must resolve outside the repository")
    if not resolved.is_file() or resolved.is_symlink():
        raise LiveMeasurementError(f"{label} must be an existing non-symlink file")
    return resolved


def _safe_member_path(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise LiveMeasurementError("unsafe RFV-2 bundle path")
    resolved = (destination / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(destination.resolve())
    except ValueError as exc:
        raise LiveMeasurementError("RFV-2 bundle path escapes extraction root") from exc
    return resolved


def safe_extract_bundle(*, bundle: Path, checksums: Path, destination: Path) -> Path:
    bundle = _external_file(bundle, "RFV-2 corpus bundle")
    checksums = _external_file(checksums, "RFV-2 bundle checksums")
    destination = require_external_output(destination, "RFV-3 extracted corpus")
    if any(destination.iterdir()):
        raise LiveMeasurementError("RFV-3 extraction destination must be empty")

    checksum_payload = load_json(checksums)
    if checksum_payload.get("schema") != "vektoryum-rfv2-live-bundle-checksums-v1":
        raise LiveMeasurementError("RFV-2 bundle checksum schema mismatch")
    if checksum_payload.get("qualified_case_count") != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("RFV-2 bundle case count mismatch")
    if checksum_payload.get("cases_sha256") != EXPECTED_CASES_SHA256:
        raise LiveMeasurementError("RFV-2 bundle case-set digest mismatch")
    if checksum_payload.get("bundle_sha256") != sha256_file(bundle):
        raise LiveMeasurementError("RFV-2 bundle digest mismatch")
    if checksum_payload.get("raw_assets_in_repository") is not False:
        raise LiveMeasurementError("RFV-2 raw-asset boundary mismatch")

    seen: set[str] = set()
    try:
        with tarfile.open(bundle, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name in seen:
                    raise LiveMeasurementError("duplicate RFV-2 bundle member")
                seen.add(member.name)
                output = _safe_member_path(destination, member.name)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise LiveMeasurementError("non-regular RFV-2 bundle member")
                source = archive.extractfile(member)
                if source is None:
                    raise LiveMeasurementError("RFV-2 bundle member cannot be read")
                output.parent.mkdir(parents=True, exist_ok=True)
                with source, output.open("wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
    except (tarfile.TarError, OSError) as exc:
        raise LiveMeasurementError("RFV-2 bundle extraction failed") from exc

    cases = load_qualification_cases(destination)
    if len(cases) != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("extracted RFV-2 corpus is incomplete")
    return destination


def partition_cases(cases: list[BenchmarkCase], *, shard_index: int, shard_count: int) -> list[BenchmarkCase]:
    if shard_count < 1 or shard_count > EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("invalid RFV-3 shard count")
    if shard_index < 0 or shard_index >= shard_count:
        raise LiveMeasurementError("invalid RFV-3 shard index")
    ordered = sorted(cases, key=lambda item: item.case_id)
    selected = ordered[shard_index::shard_count]
    if not selected:
        raise LiveMeasurementError("RFV-3 shard is empty")
    return selected


def run_shard(
    *,
    corpus_root: Path,
    output_dir: Path,
    engine_version: str,
    shard_index: int,
    shard_count: int,
    runner: Callable[..., BenchmarkResult] | None = None,
) -> list[BenchmarkResult]:
    policy = validate_policy(load_json(POLICY_PATH))
    corpus_root = require_external_directory(corpus_root, "RFV-2 corpus root")
    output_dir = require_external_output(output_dir, "RFV-3 shard output")
    all_cases = load_qualification_cases(corpus_root)
    if len(all_cases) != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("RFV-3 source corpus count mismatch")
    cases = partition_cases(all_cases, shard_index=shard_index, shard_count=shard_count)

    results: list[BenchmarkResult] = []
    retry_samples: list[dict[str, Any]] = []
    for case in cases:
        kwargs: dict[str, Any] = {
            "corpus_root": corpus_root,
            "work_root": output_dir / "jobs" / case.case_id,
            "engine_version": engine_version,
            "repeat_count": policy["repeat_count"],
            "timeout_seconds": policy["repeat_timeout_seconds"],
            "max_transient_retries": policy["max_transient_retries_per_repeat"],
        }
        if runner is not None:
            kwargs["runner"] = runner
        result, audit = run_repeated_case(case, **kwargs)
        results.append(result)
        retry_samples.extend(audit)
        write_json_atomic(
            output_dir / "retry-audit.json",
            {
                "schema": "vektoryum-rfv3-shard-retry-audit-v1",
                "expected_case_count": EXPECTED_CASE_COUNT,
                "completed_case_count": len(results),
                "repeat_count": policy["repeat_count"],
                "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
                "shard_index": shard_index,
                "shard_count": shard_count,
                "samples": retry_samples,
            },
        )

    write_results(
        output_dir / "pipeline-results.json",
        results,
        measurement_method={
            "schema": "vektoryum-rfv3-shard-measurement-v1",
            "cases_sha256": EXPECTED_CASES_SHA256,
            "expected_case_count": EXPECTED_CASE_COUNT,
            "selected_case_count": len(results),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "repeat_count": policy["repeat_count"],
            "repeat_timeout_seconds": policy["repeat_timeout_seconds"],
            "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
            "performance_aggregation": policy["performance_aggregation"],
            "quality_aggregation": policy["quality_aggregation"],
            "artifact_sha_policy": policy["artifact_sha_policy"],
            "unmeasured_metric_policy": policy["unmeasured_metric_policy"],
        },
    )
    write_json_atomic(
        output_dir / "shard-summary.json",
        {
            "schema": "vektoryum-rfv3-shard-summary-v1",
            "engine_version": engine_version,
            "cases_sha256": EXPECTED_CASES_SHA256,
            "expected_case_count": EXPECTED_CASE_COUNT,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "case_ids": [item.case_id for item in results],
            "result_count": len(results),
            "repeat_sample_count": len(retry_samples),
            "raw_assets_in_repository": False,
        },
    )
    return results


def _validate_metric_payload(metrics: Any, case_id: str) -> None:
    if not isinstance(metrics, dict) or set(metrics) != set(REQUIRED_METRICS):
        raise LiveMeasurementError(f"RFV-3 metric contract mismatch: {case_id}")
    for name, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LiveMeasurementError(f"RFV-3 non-finite metric: {case_id}:{name}")


def _validate_retry_samples(
    samples: Any,
    *,
    case_ids: list[str],
    repeat_count: int,
    retryable_failures: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(samples, list):
        raise LiveMeasurementError("RFV-3 retry samples must be a list")
    expected = {(case_id, repeat_index) for case_id in case_ids for repeat_index in range(1, repeat_count + 1)}
    seen: set[tuple[str, int]] = set()
    validated: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise LiveMeasurementError("invalid RFV-3 retry sample")
        case_id = sample.get("case_id")
        repeat_index = sample.get("repeat_index")
        identity = (case_id, repeat_index)
        if identity not in expected or identity in seen:
            raise LiveMeasurementError("duplicate or unexpected RFV-3 retry identity")
        seen.add(identity)
        attempts = sample.get("attempts")
        attempt_count = sample.get("attempt_count")
        if sample.get("status") != "success" or not isinstance(attempts, list):
            raise LiveMeasurementError("RFV-3 retry sample is not successful")
        if attempt_count != len(attempts) or attempt_count not in (1, 2):
            raise LiveMeasurementError("RFV-3 retry attempt count mismatch")
        if sample.get("retried") is not (attempt_count == 2):
            raise LiveMeasurementError("RFV-3 retry flag mismatch")
        for index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, dict) or attempt.get("attempt") != index:
                raise LiveMeasurementError("RFV-3 retry attempt identity mismatch")
            if index == attempt_count:
                if attempt.get("status") != "success" or attempt.get("retry_class") is not None or attempt.get("error") is not None:
                    raise LiveMeasurementError("RFV-3 final retry attempt must succeed")
            else:
                if attempt.get("status") != "failure" or attempt.get("retry_class") not in retryable_failures:
                    raise LiveMeasurementError("RFV-3 retry used a non-transient failure")
                if not isinstance(attempt.get("error"), str) or not attempt["error"]:
                    raise LiveMeasurementError("RFV-3 retry failure is missing an error")
        validated.append(sample)
    if seen != expected:
        raise LiveMeasurementError("RFV-3 retry audit is incomplete")
    return validated


def _expected_case_ids() -> list[str]:
    manifest = load_json(QUALIFICATION_MANIFEST_PATH)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("committed RFV-2 manifest is incomplete")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != EXPECTED_CASE_COUNT or any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise LiveMeasurementError("committed RFV-2 case identity is invalid")
    if len(set(case_ids)) != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("committed RFV-2 case identity is duplicated")
    return sorted(case_ids)


def aggregate_shards(*, input_root: Path, output_dir: Path, engine_version: str, shard_count: int) -> dict[str, Any]:
    policy = validate_policy(load_json(POLICY_PATH))
    input_root = require_external_directory(input_root, "RFV-3 shard artifact root")
    output_dir = require_external_output(output_dir, "RFV-3 aggregate output")
    if shard_count < 1 or shard_count > EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("invalid RFV-3 aggregate shard count")

    expected_case_ids = _expected_case_ids()
    expected_by_shard = {
        index: [case_id for position, case_id in enumerate(expected_case_ids) if position % shard_count == index]
        for index in range(shard_count)
    }
    summaries = sorted(input_root.rglob("shard-summary.json"))
    if len(summaries) != shard_count:
        raise LiveMeasurementError("RFV-3 shard artifact count mismatch")

    combined_results: list[BenchmarkResult] = []
    combined_samples: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    seen_cases: set[str] = set()
    for summary_path in summaries:
        summary = load_json(summary_path)
        if summary.get("schema") != "vektoryum-rfv3-shard-summary-v1":
            raise LiveMeasurementError("RFV-3 shard summary schema mismatch")
        shard_index = summary.get("shard_index")
        if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index in seen_shards:
            raise LiveMeasurementError("RFV-3 shard identity mismatch")
        seen_shards.add(shard_index)
        if summary.get("shard_count") != shard_count or summary.get("engine_version") != engine_version:
            raise LiveMeasurementError("RFV-3 shard configuration mismatch")
        if summary.get("cases_sha256") != EXPECTED_CASES_SHA256 or summary.get("raw_assets_in_repository") is not False:
            raise LiveMeasurementError("RFV-3 shard evidence boundary mismatch")
        case_ids = summary.get("case_ids")
        if case_ids != expected_by_shard.get(shard_index):
            raise LiveMeasurementError("RFV-3 shard partition mismatch")

        result_payload = load_json(summary_path.parent / "pipeline-results.json")
        if result_payload.get("schema_version") != "benchmark-results-v1":
            raise LiveMeasurementError("RFV-3 shard result schema mismatch")
        method = result_payload.get("measurement_method")
        if not isinstance(method, dict) or method.get("schema") != "vektoryum-rfv3-shard-measurement-v1":
            raise LiveMeasurementError("RFV-3 shard measurement method mismatch")
        if method.get("cases_sha256") != EXPECTED_CASES_SHA256 or method.get("shard_index") != shard_index:
            raise LiveMeasurementError("RFV-3 shard result identity mismatch")
        raw_results = result_payload.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(case_ids):
            raise LiveMeasurementError("RFV-3 shard result count mismatch")
        shard_result_ids: list[str] = []
        for raw in raw_results:
            if not isinstance(raw, dict):
                raise LiveMeasurementError("invalid RFV-3 result record")
            result = BenchmarkResult(**raw)
            result.validate()
            if result.engine_version != engine_version or result.failure is not None or result.artifact_sha256 is None:
                raise LiveMeasurementError("RFV-3 result publication state mismatch")
            _validate_metric_payload(result.metrics, result.case_id)
            if result.case_id in seen_cases:
                raise LiveMeasurementError("duplicate RFV-3 result case")
            seen_cases.add(result.case_id)
            shard_result_ids.append(result.case_id)
            combined_results.append(result)
        if shard_result_ids != case_ids:
            raise LiveMeasurementError("RFV-3 shard result order mismatch")

        retry_payload = load_json(summary_path.parent / "retry-audit.json")
        if retry_payload.get("schema") != "vektoryum-rfv3-shard-retry-audit-v1":
            raise LiveMeasurementError("RFV-3 shard retry schema mismatch")
        if retry_payload.get("shard_index") != shard_index or retry_payload.get("shard_count") != shard_count:
            raise LiveMeasurementError("RFV-3 shard retry identity mismatch")
        if retry_payload.get("completed_case_count") != len(case_ids):
            raise LiveMeasurementError("RFV-3 shard retry count mismatch")
        combined_samples.extend(
            _validate_retry_samples(
                retry_payload.get("samples"),
                case_ids=case_ids,
                repeat_count=policy["repeat_count"],
                retryable_failures=set(policy["retryable_failures"]),
            )
        )

    if seen_shards != set(range(shard_count)) or seen_cases != set(expected_case_ids):
        raise LiveMeasurementError("RFV-3 aggregate corpus is incomplete")
    if len(combined_results) != EXPECTED_CASE_COUNT:
        raise LiveMeasurementError("RFV-3 aggregate result count mismatch")
    if len(combined_samples) != EXPECTED_CASE_COUNT * policy["repeat_count"]:
        raise LiveMeasurementError("RFV-3 aggregate retry audit count mismatch")

    results_path = output_dir / "pipeline-results.json"
    retry_path = output_dir / "retry-audit.json"
    write_results(
        results_path,
        combined_results,
        measurement_method={
            "schema": "vektoryum-rfv3-live-measurement-v1",
            "cases_sha256": EXPECTED_CASES_SHA256,
            "case_count": EXPECTED_CASE_COUNT,
            "shard_count": shard_count,
            "repeat_count": policy["repeat_count"],
            "repeat_timeout_seconds": policy["repeat_timeout_seconds"],
            "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
            "performance_aggregation": policy["performance_aggregation"],
            "quality_aggregation": policy["quality_aggregation"],
            "artifact_sha_policy": policy["artifact_sha_policy"],
            "unmeasured_metric_policy": policy["unmeasured_metric_policy"],
        },
    )
    write_json_atomic(
        retry_path,
        {
            "schema": "vektoryum-rfv3-live-retry-audit-v1",
            "expected_case_count": EXPECTED_CASE_COUNT,
            "completed_case_count": EXPECTED_CASE_COUNT,
            "repeat_count": policy["repeat_count"],
            "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
            "sample_count": len(combined_samples),
            "samples": sorted(combined_samples, key=lambda item: (item["case_id"], item["repeat_index"])),
        },
    )
    envelope = {
        "schema": "vektoryum-rfv3-live-measurement-envelope-v1",
        "engine_version": engine_version,
        "case_count": EXPECTED_CASE_COUNT,
        "repeat_sample_count": len(combined_samples),
        "shard_count": shard_count,
        "cases_sha256": EXPECTED_CASES_SHA256,
        "pipeline_results_sha256": sha256_file(results_path),
        "retry_audit_sha256": sha256_file(retry_path),
        "raw_assets_in_repository": False,
    }
    write_json_atomic(output_dir / "measurement-envelope.json", envelope)
    return envelope


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFV-3B live acquisition measurement orchestration")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--bundle", type=Path, required=True)
    extract.add_argument("--checksums", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)

    shard = commands.add_parser("shard")
    shard.add_argument("--corpus-root", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--engine-version", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--engine-version", required=True)
    aggregate.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "extract":
            safe_extract_bundle(bundle=args.bundle, checksums=args.checksums, destination=args.destination)
            payload = {"status": "extracted", "case_count": EXPECTED_CASE_COUNT}
        elif args.command == "shard":
            results = run_shard(
                corpus_root=args.corpus_root,
                output_dir=args.output,
                engine_version=args.engine_version,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            payload = {"status": "measured", "shard_index": args.shard_index, "case_count": len(results)}
        else:
            envelope = aggregate_shards(
                input_root=args.input_root,
                output_dir=args.output,
                engine_version=args.engine_version,
                shard_count=args.shard_count,
            )
            payload = {"status": "aggregated", **envelope}
    except (LiveMeasurementError, MeasurementError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
