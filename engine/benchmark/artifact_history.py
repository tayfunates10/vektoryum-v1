"""Compare scheduled benchmark seed manifests without inventing quality metrics.

This layer is intentionally limited to corpus/artifact integrity. True quality metric
deltas are handled by benchmark.compare once real pipeline result files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_seed_manifests(baseline: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    current_cases = {item["case_id"]: item for item in current.get("cases", [])}
    if baseline is None:
        return {
            "schema_version": "benchmark-artifact-history-v1",
            "status": "bootstrap",
            "baseline_available": False,
            "case_count": len(current_cases),
            "added": sorted(current_cases),
            "removed": [],
            "changed": [],
        }

    baseline_cases = {item["case_id"]: item for item in baseline.get("cases", [])}
    added = sorted(set(current_cases) - set(baseline_cases))
    removed = sorted(set(baseline_cases) - set(current_cases))
    changed = sorted(
        case_id
        for case_id in set(current_cases) & set(baseline_cases)
        if current_cases[case_id].get("source_sha256") != baseline_cases[case_id].get("source_sha256")
    )
    status = "changed" if added or removed or changed else "stable"
    return {
        "schema_version": "benchmark-artifact-history-v1",
        "status": status,
        "baseline_available": True,
        "case_count": len(current_cases),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_delta.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = "".join(
        f"<tr><td>{kind}</td><td>{', '.join(report[kind]) or '-'}</td></tr>"
        for kind in ("added", "removed", "changed")
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Benchmark Artifact Delta</title>"
        "</head><body><h1>Benchmark Artifact History</h1>"
        f"<p>Status: {report['status']}</p><p>Baseline available: {report['baseline_available']}</p>"
        f"<table><tbody>{rows}</tbody></table></body></html>"
    )
    (output_dir / "artifact_delta.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = json.loads(args.current.read_text(encoding="utf-8"))
    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = compare_seed_manifests(baseline, current)
    write_reports(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
