"""Run the deterministic eight-category real-pipeline benchmark seed set."""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
from statistics import median

from app.pipeline_entry import run_pipeline
from benchmark.manifest import BenchmarkResult
from benchmark.pipeline_results import run_case, write_results
from benchmark.seed_runner import CATEGORIES, generate_seed_corpus

BENCHMARK_CATEGORIES = frozenset(CATEGORIES)
REPEAT_COUNT = 3
_HIGHER_IS_BETTER = {"fidelity", "ssim", "edge_f1", "alpha_iou"}
_LOWER_IS_BETTER = {"delta_e00", "path_count", "svg_bytes"}
_PERFORMANCE_METRICS = {"render_ms", "peak_rss_mb"}


def _peak_rss_mb() -> float:
    # GitHub Actions uses Linux, where ru_maxrss is reported in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _measured(values: list[float | int | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


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

    return BenchmarkResult(
        case_id=first.case_id,
        engine_version=first.engine_version,
        metrics=metrics,
        artifact_sha256=first.artifact_sha256,
    )


def run_smoke(output_dir: Path, *, engine_version: str, repeat_count: int = REPEAT_COUNT) -> list[BenchmarkResult]:
    if repeat_count < 1 or repeat_count % 2 == 0:
        raise ValueError("repeat_count must be a positive odd integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = output_dir / "corpus"
    work_root = output_dir / "jobs"
    cases = [
        case
        for case in generate_seed_corpus(corpus_root)
        if case.category in BENCHMARK_CATEGORIES
    ]
    results: list[BenchmarkResult] = []
    for case in cases:
        repeats = []
        for repeat_index in range(repeat_count):
            repeats.append(
                run_case(
                    case,
                    corpus_root=corpus_root,
                    work_root=work_root / f"repeat-{repeat_index + 1}",
                    pipeline=run_pipeline,
                    engine_version=engine_version,
                    trace_mode="auto",
                    peak_rss_mb=_peak_rss_mb(),
                )
            )
        results.append(aggregate_repeats(repeats))
    write_results(output_dir / "pipeline_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--engine-version", default="unknown")
    parser.add_argument("--repeat-count", type=int, default=REPEAT_COUNT)
    args = parser.parse_args()
    results = run_smoke(args.output, engine_version=args.engine_version, repeat_count=args.repeat_count)
    print(json.dumps({"status": "ok", "case_count": len(results), "repeat_count": args.repeat_count}, sort_keys=True))


if __name__ == "__main__":
    main()
