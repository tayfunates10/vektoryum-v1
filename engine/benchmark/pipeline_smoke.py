"""Run the deterministic eight-category real-pipeline benchmark seed set."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
from pathlib import Path
from statistics import median
from typing import Any

from app.pipeline_entry import run_pipeline
from benchmark.manifest import BenchmarkCase, BenchmarkResult
from benchmark.pipeline_results import run_case, write_results
from benchmark.seed_runner import CATEGORIES, generate_seed_corpus

BENCHMARK_CATEGORIES = frozenset(CATEGORIES)
REPEAT_COUNT = 3
REPEAT_TIMEOUT_SECONDS = 1800
MEASUREMENT_METHOD_VERSION = "median-performance-v3-repeat-samples"
REPEAT_PROVENANCE_SCHEMA = "rfv3e-repeat-metric-provenance-v1"
_HIGHER_IS_BETTER = {"fidelity", "ssim", "edge_f1", "alpha_iou"}
_LOWER_IS_BETTER = {"delta_e00", "path_count", "svg_bytes"}
_PERFORMANCE_METRICS = {"render_ms", "peak_rss_mb"}
_PROVENANCE_AGGREGATE_KEYS = {"repeat_count", "repeat_provenance_schema", "repeat_provenance"}
_PROVENANCE_VOLATILE_KEYS = {"exact_evaluator_failure_message_sanitized"} | _PROVENANCE_AGGREGATE_KEYS


def _peak_rss_mb() -> float:
    # Each repeat runs in a fresh spawned process, so ru_maxrss belongs only to that repeat.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _isolated_worker(
    queue: Any,
    case_payload: dict[str, Any],
    corpus_root: str,
    work_root: str,
    engine_version: str,
) -> None:
    try:
        case = BenchmarkCase(**case_payload)
        result = run_case(
            case,
            corpus_root=Path(corpus_root),
            work_root=Path(work_root),
            pipeline=run_pipeline,
            engine_version=engine_version,
            trace_mode="auto",
            peak_rss_mb=None,
        )
        result.metrics["peak_rss_mb"] = round(_peak_rss_mb(), 6)
        queue.put({"ok": True, "result": result.to_dict()})
    except BaseException as exc:  # fail closed across the process boundary
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_case_isolated(
    case: BenchmarkCase,
    *,
    corpus_root: Path,
    work_root: Path,
    engine_version: str,
    timeout_seconds: int = REPEAT_TIMEOUT_SECONDS,
) -> BenchmarkResult:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_isolated_worker,
        args=(queue, case.to_dict(), str(corpus_root), str(work_root), engine_version),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"isolated benchmark repeat timed out: {case.case_id}")
    if queue.empty():
        raise RuntimeError(f"isolated benchmark repeat exited without a result: {case.case_id} ({process.exitcode})")
    payload = queue.get()
    if not payload.get("ok") or process.exitcode != 0:
        raise RuntimeError(f"isolated benchmark repeat failed: {case.case_id}: {payload.get('error')}")
    return BenchmarkResult(**payload["result"])


def _measured(values: list[float | int | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _provenance_decision_view(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        key: provenance.get(key)
        for key in sorted(set(provenance) - _PROVENANCE_VOLATILE_KEYS)
    }


def _repeat_provenance_snapshot(
    provenance: dict[str, Any], *, repeat_index: int, artifact_sha256: str | None
) -> dict[str, Any]:
    snapshot = {
        key: value
        for key, value in provenance.items()
        if key not in _PROVENANCE_AGGREGATE_KEYS
    }
    snapshot["repeat_index"] = repeat_index
    snapshot["artifact_sha256"] = artifact_sha256
    return snapshot


def aggregate_repeats(repeats: list[BenchmarkResult]) -> BenchmarkResult:
    if not repeats:
        raise ValueError("at least one benchmark repeat is required")
    first = repeats[0]
    if any(item.case_id != first.case_id or item.engine_version != first.engine_version for item in repeats):
        raise ValueError("benchmark repeats must describe the same case and engine")

    artifact_shas = {item.artifact_sha256 for item in repeats}
    if len(artifact_shas) != 1:
        raise ValueError(f"non-deterministic artifact sha256: {first.case_id}")
    failures = {item.failure for item in repeats}
    if failures != {None}:
        raise ValueError(f"benchmark repeat failed: {first.case_id}: {sorted(str(item) for item in failures)}")

    metrics: dict[str, float | int | None] = {}
    metric_names = set().union(*(item.metrics.keys() for item in repeats))
    for metric in sorted(metric_names):
        values = _measured([item.metrics.get(metric) for item in repeats])
        if not values:
            metrics[metric] = None
        elif metric in _PERFORMANCE_METRICS:
            metrics[metric] = round(float(median(values)), 6)
        elif metric in _HIGHER_IS_BETTER:
            metrics[metric] = min(values)
        elif metric in _LOWER_IS_BETTER:
            metrics[metric] = max(values)
        else:
            raise ValueError(f"unsupported repeated benchmark metric: {metric}")

    # RFV-3D2/RFV-3E: repeats must agree on the metric-path decision, while the
    # full sanitized per-repeat provenance is retained for live diagnostics.
    provenances = [item.metric_provenance for item in repeats]
    provenance: dict[str, Any] | None = None
    if any(item is not None for item in provenances):
        if any(item is None for item in provenances):
            raise ValueError(f"mixed metric provenance coverage: {first.case_id}")
        typed = [item for item in provenances if item is not None]
        base_keys = set(typed[0]) - _PROVENANCE_VOLATILE_KEYS
        base = _provenance_decision_view(typed[0])
        for item in typed[1:]:
            if set(item) - _PROVENANCE_VOLATILE_KEYS != base_keys:
                raise ValueError(f"metric provenance field drift: {first.case_id}")
            if _provenance_decision_view(item) != base:
                raise ValueError(f"non-deterministic metric provenance: {first.case_id}")
        for repeat, item in zip(repeats, typed):
            if item.get("artifact_sha256") not in (None, repeat.artifact_sha256):
                raise ValueError(f"metric provenance artifact mismatch: {first.case_id}")
        provenance = {
            key: value
            for key, value in typed[0].items()
            if key not in _PROVENANCE_AGGREGATE_KEYS
        }
        provenance["repeat_count"] = len(repeats)
        provenance["repeat_provenance_schema"] = REPEAT_PROVENANCE_SCHEMA
        provenance["repeat_provenance"] = [
            _repeat_provenance_snapshot(
                item,
                repeat_index=index,
                artifact_sha256=repeat.artifact_sha256,
            )
            for index, (repeat, item) in enumerate(zip(repeats, typed), start=1)
        ]

    return BenchmarkResult(
        case_id=first.case_id,
        engine_version=first.engine_version,
        metrics=metrics,
        artifact_sha256=first.artifact_sha256,
        metric_provenance=provenance,
    )


def _write_repeat_samples(path: Path, samples: list[dict[str, Any]], *, repeat_count: int, timeout_seconds: int) -> None:
    payload = {
        "schema_version": "benchmark-repeat-samples-v1",
        "measurement_method": MEASUREMENT_METHOD_VERSION,
        "repeat_count": repeat_count,
        "timeout_seconds": timeout_seconds,
        "sample_count": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_smoke(
    output_dir: Path,
    *,
    engine_version: str,
    repeat_count: int = REPEAT_COUNT,
    timeout_seconds: int = REPEAT_TIMEOUT_SECONDS,
) -> list[BenchmarkResult]:
    if repeat_count < 1 or repeat_count % 2 == 0:
        raise ValueError("repeat_count must be a positive odd integer")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = output_dir / "corpus"
    work_root = output_dir / "jobs"
    samples_path = output_dir / "repeat_samples.json"
    cases = [
        case
        for case in generate_seed_corpus(corpus_root)
        if case.category in BENCHMARK_CATEGORIES
    ]
    results: list[BenchmarkResult] = []
    samples: list[dict[str, Any]] = []
    for case in cases:
        repeats = []
        for repeat_index in range(repeat_count):
            sample = {
                "case_id": case.case_id,
                "repeat_index": repeat_index + 1,
                "status": "running",
                "result": None,
                "error": None,
            }
            try:
                result = _run_case_isolated(
                    case,
                    corpus_root=corpus_root,
                    work_root=work_root / f"repeat-{repeat_index + 1}",
                    engine_version=engine_version,
                    timeout_seconds=timeout_seconds,
                )
            except BaseException as exc:
                sample["status"] = "failure"
                sample["error"] = f"{type(exc).__name__}: {exc}"
                samples.append(sample)
                _write_repeat_samples(samples_path, samples, repeat_count=repeat_count, timeout_seconds=timeout_seconds)
                raise
            sample["status"] = "success"
            sample["result"] = result.to_dict()
            samples.append(sample)
            _write_repeat_samples(samples_path, samples, repeat_count=repeat_count, timeout_seconds=timeout_seconds)
            repeats.append(result)
        results.append(aggregate_repeats(repeats))
    write_results(
        output_dir / "pipeline_results.json",
        results,
        measurement_method={
            "version": MEASUREMENT_METHOD_VERSION,
            "repeat_count": repeat_count,
            "performance_aggregation": "median",
            "quality_aggregation": "conservative_worst_case",
            "artifact_sha_policy": "all_repeats_must_match",
            "rss_scope": "fresh_spawned_process_per_repeat",
            "repeat_samples": "repeat_samples.json",
            "timeout_seconds": timeout_seconds,
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--engine-version", default="unknown")
    parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
    parser.add_argument("--repeat-timeout-seconds", type=int, default=REPEAT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    results = run_smoke(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        timeout_seconds=args.repeat_timeout_seconds,
    )
    print(json.dumps({"status": "ok", "case_count": len(results), "repeat_count": args.repeat_count}, sort_keys=True))


if __name__ == "__main__":
    main()
