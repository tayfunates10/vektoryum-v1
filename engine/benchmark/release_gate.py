"""Fail-closed release gate for benchmark result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.compare import build_delta_report
from benchmark.manifest import REQUIRED_METRICS


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != "benchmark-results-v1":
        raise ValueError("unsupported benchmark results schema")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark results list is required")
    return results


def evaluate_release_gate(
    current_payload: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    current = _results(current_payload)
    unmeasured: list[dict[str, str]] = []
    for case in current:
        metrics = case.get("metrics") or {}
        for metric in sorted(REQUIRED_METRICS):
            if metric not in metrics or metrics[metric] is None:
                unmeasured.append({"case_id": str(case.get("case_id", "")), "metric": metric})

    if baseline_payload is None:
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "bootstrap" if not unmeasured else "fail",
            "reason": "baseline_missing" if not unmeasured else "unmeasured_required_metrics",
            "unmeasured": unmeasured,
            "delta": None,
        }

    baseline = _results(baseline_payload)
    if unmeasured:
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "fail",
            "reason": "unmeasured_required_metrics",
            "unmeasured": unmeasured,
            "delta": None,
        }

    delta = build_delta_report(baseline, current)
    return {
        "schema_version": "benchmark-release-gate-v1",
        "status": "pass" if delta["status"] == "pass" else "fail",
        "reason": "within_tolerance" if delta["status"] == "pass" else "metric_regression",
        "unmeasured": [],
        "delta": delta,
    }


def write_gate_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    current = json.loads(args.current.read_text(encoding="utf-8"))
    baseline = None
    if args.baseline is not None and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    report = evaluate_release_gate(current, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gate_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"pass", "bootstrap"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
