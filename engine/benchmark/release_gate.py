"""Fail-closed release gate for benchmark result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.compare import build_delta_report
from benchmark.manifest import REQUIRED_METRICS

_ALPHA_RELEVANT_CATEGORIES = {"transparent"}


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != "benchmark-results-v1":
        raise ValueError("unsupported benchmark results schema")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark results list is required")
    return results


def _category(case_id: str) -> str:
    parts = str(case_id).split("-", 2)
    return parts[2] if len(parts) == 3 else str(case_id)


def _required_metrics(case_id: str) -> set[str]:
    required = set(REQUIRED_METRICS)
    if _category(case_id) not in _ALPHA_RELEVANT_CATEGORIES:
        required.discard("alpha_iou")
    return required


def evaluate_release_gate(current_payload: dict[str, Any], baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    current = _results(current_payload)
    unmeasured: list[dict[str, str]] = []
    for case in current:
        case_id = str(case.get("case_id", ""))
        metrics = case.get("metrics") or {}
        for metric in sorted(_required_metrics(case_id)):
            if metric not in metrics or metrics[metric] is None:
                unmeasured.append({"case_id": case_id, "metric": metric})

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

    current_ids = {str(item.get("case_id", "")) for item in current}
    baseline_ids = {str(item.get("case_id", "")) for item in baseline}
    if "" in current_ids or "" in baseline_ids or len(current_ids) != len(current) or len(baseline_ids) != len(baseline):
        raise ValueError("benchmark case ids must be non-empty and unique")
    if baseline_ids != current_ids:
        added = sorted(current_ids - baseline_ids)
        removed = sorted(baseline_ids - current_ids)
        if baseline_ids < current_ids:
            return {
                "schema_version": "benchmark-release-gate-v1",
                "status": "bootstrap",
                "reason": "case_set_expanded",
                "unmeasured": [],
                "delta": None,
                "case_set": {"added": added, "removed": []},
            }
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "fail",
            "reason": "case_set_mismatch",
            "unmeasured": [],
            "delta": None,
            "case_set": {"added": added, "removed": removed},
        }

    current_method = current_payload.get("measurement_method")
    baseline_method = baseline_payload.get("measurement_method")
    if current_method != baseline_method:
        if not isinstance(current_method, dict) or not str(current_method.get("version", "")).strip():
            return {
                "schema_version": "benchmark-release-gate-v1",
                "status": "fail",
                "reason": "invalid_measurement_method",
                "unmeasured": [],
                "delta": None,
            }
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "bootstrap",
            "reason": "measurement_method_changed",
            "unmeasured": [],
            "delta": None,
            "measurement_method": {
                "baseline": baseline_method,
                "current": current_method,
            },
        }

    exclusions = {
        case_id: {"alpha_iou"}
        for case_id in current_ids
        if _category(case_id) not in _ALPHA_RELEVANT_CATEGORIES
    }
    delta = build_delta_report(baseline, current, excluded_metrics_by_case=exclusions)
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
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline is not None and args.baseline.exists() else None
    report = evaluate_release_gate(current, baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gate_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"pass", "bootstrap"} else 1


if __name__ == "__main__":
    raise SystemExit(main())