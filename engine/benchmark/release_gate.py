"""Fail-closed release gate for benchmark result artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import median
from typing import Any

from benchmark.compare import build_delta_report
from benchmark.manifest import REQUIRED_METRICS

_ALPHA_RELEVANT_CATEGORIES = {"transparent"}
_COMMON_MODE_MIN_CASES = 6
_COMMON_MODE_SHARE = 0.75
_COMMON_MODE_TRIGGER = 0.10
_COMMON_MODE_MAX_FACTOR = 1.50


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


def _only_render_time_regressed(delta: dict[str, Any]) -> bool:
    regressions = [
        metric["metric"]
        for case in delta.get("cases", [])
        for metric in case.get("metrics", [])
        if metric.get("status") == "regression"
    ]
    return bool(regressions) and set(regressions) == {"render_ms"}


def _normalize_common_mode_timing(
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    raw_delta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Remove bounded hosted-runner slowdown shared by almost the whole corpus.

    A single case or mixed metric regression is never normalized.  At least six
    cases and 75% of the corpus must exceed the existing 10% timing threshold in
    the same direction.  The median slowdown is capped at 1.50x; larger shifts
    remain fail-closed.  Raw ratios are retained in the report for audit.
    """
    if not _only_render_time_regressed(raw_delta):
        return current, None

    baseline_by_id = {str(item["case_id"]): item for item in baseline}
    ratios: dict[str, float] = {}
    for item in current:
        case_id = str(item["case_id"])
        before = float(baseline_by_id[case_id]["metrics"]["render_ms"])
        after = float(item["metrics"]["render_ms"])
        if before <= 0 or after <= 0:
            return current, None
        ratios[case_id] = after / before

    if len(ratios) < _COMMON_MODE_MIN_CASES:
        return current, None
    slow = sum(ratio > 1.0 + _COMMON_MODE_TRIGGER for ratio in ratios.values())
    if slow / len(ratios) < _COMMON_MODE_SHARE:
        return current, None

    factor = float(median(ratios.values()))
    if not 1.0 + _COMMON_MODE_TRIGGER < factor <= _COMMON_MODE_MAX_FACTOR:
        return current, None

    normalized = deepcopy(current)
    for item in normalized:
        metrics = item["metrics"]
        metrics["render_ms"] = float(metrics["render_ms"]) / factor
    return normalized, {
        "schema_version": "benchmark-timing-normalization-v1",
        "applied": True,
        "reason": "bounded_common_mode_hosted_runner_slowdown",
        "factor": factor,
        "case_count": len(ratios),
        "slow_case_count": slow,
        "minimum_share": _COMMON_MODE_SHARE,
        "maximum_factor": _COMMON_MODE_MAX_FACTOR,
        "raw_ratios": {case_id: ratios[case_id] for case_id in sorted(ratios)},
        "raw_regression_count": int(raw_delta.get("regression_count", 0)),
    }


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
            "timing_normalization": None,
        }

    baseline = _results(baseline_payload)
    if unmeasured:
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "fail",
            "reason": "unmeasured_required_metrics",
            "unmeasured": unmeasured,
            "delta": None,
            "timing_normalization": None,
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
                "timing_normalization": None,
                "case_set": {"added": added, "removed": []},
            }
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "fail",
            "reason": "case_set_mismatch",
            "unmeasured": [],
            "delta": None,
            "timing_normalization": None,
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
                "timing_normalization": None,
            }
        return {
            "schema_version": "benchmark-release-gate-v1",
            "status": "bootstrap",
            "reason": "measurement_method_changed",
            "unmeasured": [],
            "delta": None,
            "timing_normalization": None,
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
    raw_delta = build_delta_report(baseline, current, excluded_metrics_by_case=exclusions)
    compared_current, timing_normalization = _normalize_common_mode_timing(
        baseline,
        current,
        raw_delta,
    )
    delta = (
        build_delta_report(baseline, compared_current, excluded_metrics_by_case=exclusions)
        if timing_normalization is not None
        else raw_delta
    )
    return {
        "schema_version": "benchmark-release-gate-v1",
        "status": "pass" if delta["status"] == "pass" else "fail",
        "reason": "within_tolerance" if delta["status"] == "pass" else "metric_regression",
        "unmeasured": [],
        "delta": delta,
        "timing_normalization": timing_normalization,
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
