from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from benchmark.manifest import BenchmarkCase, BenchmarkResult, REQUIRED_METRICS
from benchmark.pipeline_results import write_results
from benchmark.pipeline_smoke import _run_case_isolated, aggregate_repeats

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "engine" / "regression" / "rfv3_measurement_policy.json"
QUALIFICATION_MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
EXPECTED_CASES_SHA256 = "5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89"

CATEGORY_MAP = {
    "flat_logo": "logos",
    "badge_seal": "seals",
    "small_text": "multilingual",
    "monoline": "technical",
    "multicolor": "logos",
    "low_resolution_signage_photo": "low_resolution",
    "gradient_artwork": "gradients",
    "native_4k": "logos",
    "transparent_dark_background": "transparent",
    "complex_illustration": "logos",
}

CaseRunner = Callable[..., BenchmarkResult]


class MeasurementError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON root must be an object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise MeasurementError(f"{label} must resolve outside the repository")
    if not resolved.exists() or not resolved.is_dir() or resolved.is_symlink():
        raise MeasurementError(f"{label} must be an existing non-symlink directory")
    return resolved


def require_external_output(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise MeasurementError(f"{label} must resolve outside the repository")
    if resolved.exists() and resolved.is_symlink():
        raise MeasurementError(f"{label} symlinks are forbidden")
    resolved.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink():
        raise MeasurementError(f"{label} symlinks are forbidden")
    return resolved


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema") != "vektoryum-rfv3-measurement-policy-v1":
        raise MeasurementError("RFV-3 measurement policy schema mismatch")
    if policy.get("expected_case_count") != 24:
        raise MeasurementError("RFV-3 finite case count drift")
    if policy.get("expected_cases_sha256") != EXPECTED_CASES_SHA256:
        raise MeasurementError("RFV-3 case-set digest drift")
    repeat_count = policy.get("repeat_count")
    if not isinstance(repeat_count, int) or isinstance(repeat_count, bool) or repeat_count < 1 or repeat_count % 2 == 0:
        raise MeasurementError("repeat_count must be a positive odd integer")
    timeout = policy.get("repeat_timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise MeasurementError("repeat timeout must be positive")
    retries = policy.get("max_transient_retries_per_repeat")
    if retries != 1:
        raise MeasurementError("RFV-3 permits exactly one transient retry per repeat")
    if set(policy.get("required_metrics") or []) != set(REQUIRED_METRICS):
        raise MeasurementError("RFV-3 required metric contract drift")
    if policy.get("unmeasured_metric_policy") != "preserve_null_never_fabricate":
        raise MeasurementError("unmeasured metric policy drift")
    if policy.get("raw_assets_in_repository") is not False:
        raise MeasurementError("raw assets must remain outside the repository")
    if policy.get("phase_completion_requires_live_results") is not True:
        raise MeasurementError("RFV-3 cannot complete without live results")
    return policy


def _validate_case_record(case: dict[str, Any]) -> None:
    required = {
        "case_id",
        "category",
        "source_sha256",
        "license",
        "source_format",
        "storage_object_id",
        "source_verified",
        "consent_verified",
        "object_immutable",
        "decode_verified",
        "privacy_review",
        "contains_public_pii",
    }
    if not required.issubset(case):
        raise MeasurementError("qualification case is missing measurement fields")
    if case["category"] not in CATEGORY_MAP:
        raise MeasurementError(f"unsupported RFV category: {case['category']}")
    if case["privacy_review"] != "approved" or case["contains_public_pii"] is not False:
        raise MeasurementError("qualification privacy evidence failed")
    for key in ("source_verified", "consent_verified", "object_immutable", "decode_verified"):
        if case[key] is not True:
            raise MeasurementError(f"qualification verification failed: {key}")


def load_qualification_cases(corpus_root: Path) -> list[BenchmarkCase]:
    corpus_root = require_external_directory(corpus_root, "RFV-2 corpus root")
    bundle_manifest_path = corpus_root / "qualification-manifest.json"
    bundle_index_path = corpus_root / "bundle-index.json"
    if not bundle_manifest_path.is_file() or bundle_manifest_path.is_symlink():
        raise MeasurementError("qualification manifest is missing from the extracted bundle")
    if not bundle_index_path.is_file() or bundle_index_path.is_symlink():
        raise MeasurementError("bundle index is missing from the extracted bundle")

    bundle_manifest = load_json(bundle_manifest_path)
    repository_manifest = load_json(QUALIFICATION_MANIFEST_PATH)
    bundle_index = load_json(bundle_index_path)
    if bundle_manifest.get("schema") != "vektoryum-rfv2-qualification-manifest-v1":
        raise MeasurementError("qualification manifest schema mismatch")
    if bundle_manifest.get("status") != "qualified" or bundle_manifest.get("qualified_case_count") != 24:
        raise MeasurementError("qualification corpus is not complete")
    cases = bundle_manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 24 or any(not isinstance(case, dict) for case in cases):
        raise MeasurementError("qualification corpus must contain exactly 24 cases")
    if canonical_sha256(cases) != EXPECTED_CASES_SHA256:
        raise MeasurementError("qualification case-set digest mismatch")
    if bundle_manifest.get("cases_sha256") != EXPECTED_CASES_SHA256:
        raise MeasurementError("qualification manifest digest mismatch")
    if repository_manifest.get("cases") != cases or repository_manifest.get("cases_sha256") != EXPECTED_CASES_SHA256:
        raise MeasurementError("extracted corpus does not match committed RFV-2 evidence")
    if bundle_index.get("schema") != "vektoryum-rfv2-live-bundle-index-v1":
        raise MeasurementError("bundle index schema mismatch")
    if bundle_index.get("qualified_case_count") != 24 or bundle_index.get("cases_sha256") != EXPECTED_CASES_SHA256:
        raise MeasurementError("bundle index identity mismatch")
    if bundle_index.get("raw_assets_in_repository") is not False:
        raise MeasurementError("bundle repository boundary mismatch")

    benchmark_cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for case in cases:
        _validate_case_record(case)
        case_id = case["case_id"]
        source_sha256 = case["source_sha256"]
        if case_id in seen_ids or source_sha256 in seen_sources:
            raise MeasurementError("duplicate qualification identity")
        seen_ids.add(case_id)
        seen_sources.add(source_sha256)

        source_path = Path("objects") / case["storage_object_id"]
        resolved = (corpus_root / source_path).resolve()
        if not _is_inside(resolved, corpus_root):
            raise MeasurementError("qualification object escapes corpus root")
        if not resolved.is_file() or resolved.is_symlink():
            raise MeasurementError(f"qualification object missing: {case_id}")
        if sha256_file(resolved) != source_sha256:
            raise MeasurementError(f"qualification source digest mismatch: {case_id}")
        benchmark_case = BenchmarkCase(
            case_id=case_id,
            category=CATEGORY_MAP[case["category"]],
            source_path=source_path.as_posix(),
            license_id=case["license"],
            source_sha256=source_sha256,
            tags=("rfv2", f"rfv_category:{case['category']}", f"source_format:{case['source_format']}"),
        )
        benchmark_case.validate()
        benchmark_cases.append(benchmark_case)
    return sorted(benchmark_cases, key=lambda item: item.case_id)


def _retry_class(exc: BaseException) -> str | None:
    if isinstance(exc, TimeoutError):
        return "TimeoutError"
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if "exited without a result" in message:
            return "isolated_worker_no_result"
        if "isolated benchmark repeat failed" in message:
            return "isolated_worker_exit"
    return None


def _validate_result(result: BenchmarkResult, case: BenchmarkCase) -> None:
    result.validate()
    if result.case_id != case.case_id:
        raise MeasurementError("measurement result case identity mismatch")
    if result.failure is not None:
        raise MeasurementError(f"measurement result contains failure: {case.case_id}")
    if result.artifact_sha256 is None:
        raise MeasurementError(f"measurement result is missing artifact digest: {case.case_id}")
    if set(result.metrics) != set(REQUIRED_METRICS):
        raise MeasurementError(f"measurement metric set mismatch: {case.case_id}")
    for name, value in result.metrics.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise MeasurementError(f"non-finite measurement metric: {case.case_id}:{name}")


def run_repeated_case(
    case: BenchmarkCase,
    *,
    corpus_root: Path,
    work_root: Path,
    engine_version: str,
    repeat_count: int,
    timeout_seconds: int,
    max_transient_retries: int,
    runner: CaseRunner = _run_case_isolated,
) -> tuple[BenchmarkResult, list[dict[str, Any]]]:
    if repeat_count < 1 or repeat_count % 2 == 0:
        raise MeasurementError("repeat_count must be a positive odd integer")
    if max_transient_retries != 1:
        raise MeasurementError("exactly one transient retry is permitted")

    repeats: list[BenchmarkResult] = []
    audit: list[dict[str, Any]] = []
    for repeat_index in range(1, repeat_count + 1):
        attempts: list[dict[str, Any]] = []
        result: BenchmarkResult | None = None
        for attempt_index in range(1, max_transient_retries + 2):
            try:
                candidate = runner(
                    case,
                    corpus_root=corpus_root,
                    work_root=work_root / f"repeat-{repeat_index}" / f"attempt-{attempt_index}",
                    engine_version=engine_version,
                    timeout_seconds=timeout_seconds,
                )
                _validate_result(candidate, case)
                attempts.append({"attempt": attempt_index, "status": "success", "retry_class": None, "error": None})
                result = candidate
                break
            except BaseException as exc:
                retry_class = _retry_class(exc)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "status": "failure",
                        "retry_class": retry_class,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if retry_class is None or attempt_index > max_transient_retries:
                    raise MeasurementError(
                        f"measurement repeat failed closed: {case.case_id}: repeat={repeat_index}: {type(exc).__name__}: {exc}"
                    ) from exc
        if result is None:
            raise MeasurementError(f"measurement repeat produced no result: {case.case_id}")
        repeats.append(result)
        audit.append(
            {
                "case_id": case.case_id,
                "repeat_index": repeat_index,
                "attempt_count": len(attempts),
                "retried": len(attempts) > 1,
                "status": "success",
                "attempts": attempts,
            }
        )
    try:
        aggregated = aggregate_repeats(repeats)
    except (ValueError, TypeError) as exc:
        raise MeasurementError(f"repeat aggregation failed closed: {case.case_id}: {exc}") from exc
    _validate_result(aggregated, case)
    return aggregated, audit


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".rfv3-", delete=False) as temp:
            temp_name = temp.name
            json.dump(value, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def run_measurement(
    *,
    corpus_root: Path,
    output_dir: Path,
    engine_version: str,
    runner: CaseRunner = _run_case_isolated,
) -> list[BenchmarkResult]:
    policy = validate_policy(load_json(POLICY_PATH))
    corpus_root = require_external_directory(corpus_root, "RFV-2 corpus root")
    output_dir = require_external_output(output_dir, "RFV-3 output directory")
    cases = load_qualification_cases(corpus_root)
    if len(cases) != policy["expected_case_count"]:
        raise MeasurementError("RFV-3 case count mismatch")

    results: list[BenchmarkResult] = []
    retry_samples: list[dict[str, Any]] = []
    work_root = output_dir / "jobs"
    for case in cases:
        result, audit = run_repeated_case(
            case,
            corpus_root=corpus_root,
            work_root=work_root / case.case_id,
            engine_version=engine_version,
            repeat_count=policy["repeat_count"],
            timeout_seconds=policy["repeat_timeout_seconds"],
            max_transient_retries=policy["max_transient_retries_per_repeat"],
            runner=runner,
        )
        results.append(result)
        retry_samples.extend(audit)
        write_json_atomic(
            output_dir / "retry-audit.json",
            {
                "schema": "vektoryum-rfv3-retry-audit-v1",
                "expected_case_count": policy["expected_case_count"],
                "completed_case_count": len(results),
                "repeat_count": policy["repeat_count"],
                "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
                "samples": retry_samples,
            },
        )

    write_results(
        output_dir / "pipeline-results.json",
        results,
        measurement_method={
            "schema": "vektoryum-rfv3-measurement-method-v1",
            "cases_sha256": EXPECTED_CASES_SHA256,
            "case_count": len(results),
            "repeat_count": policy["repeat_count"],
            "repeat_timeout_seconds": policy["repeat_timeout_seconds"],
            "max_transient_retries_per_repeat": policy["max_transient_retries_per_repeat"],
            "performance_aggregation": policy["performance_aggregation"],
            "quality_aggregation": policy["quality_aggregation"],
            "artifact_sha_policy": policy["artifact_sha_policy"],
            "unmeasured_metric_policy": policy["unmeasured_metric_policy"],
        },
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the qualified RFV-2 corpus through the real production pipeline.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-version", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = run_measurement(
            corpus_root=args.corpus_root,
            output_dir=args.output,
            engine_version=args.engine_version,
        )
    except MeasurementError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "measured", "case_count": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
