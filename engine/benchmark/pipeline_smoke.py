"""Run a tiny deterministic real-pipeline benchmark subset."""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

from app.pipeline_entry import run_pipeline
from benchmark.pipeline_results import run_case, write_results
from benchmark.seed_runner import generate_seed_corpus

SMOKE_CATEGORIES = {"logos", "transparent"}


def _peak_rss_mb() -> float:
    # GitHub Actions uses Linux, where ru_maxrss is reported in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def run_smoke(output_dir: Path, *, engine_version: str) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = output_dir / "corpus"
    work_root = output_dir / "jobs"
    cases = [case for case in generate_seed_corpus(corpus_root) if case.category in SMOKE_CATEGORIES]
    results = []
    for case in cases:
        results.append(
            run_case(
                case,
                corpus_root=corpus_root,
                work_root=work_root,
                pipeline=run_pipeline,
                engine_version=engine_version,
                trace_mode="auto",
                peak_rss_mb=_peak_rss_mb(),
            )
        )
    write_results(output_dir / "pipeline_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_artifacts"))
    parser.add_argument("--engine-version", default="unknown")
    args = parser.parse_args()
    results = run_smoke(args.output, engine_version=args.engine_version)
    print(json.dumps({"status": "ok", "case_count": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
