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
_RELEASE_TOLERANCES = {"render_ms": 0.25}
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


def _case_ids(results: list[dict[str, Any]]) -> set[str]:
    ids = {str(item.get("case_id", "")) for item in results}
    if "" in ids or len(ids) != len(results):
        raise ValueError("benchmark case ids must be non-empty and unique")
    return ids


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
    """Remove bounded hosted-runner slowdown for non-PR/scheduled comparisons.

    PR runs should supply a same-run base-SHA timing baseline instead. This
    historical fallback remains fail-closed above a 1.50x median shift.
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
        item["metrics"]["render_ms"] = float(item["metrics"]["render_ms"]) / factor
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


def _apply_same_runner_timing_baseline(
    historical_baseline: list[dict[str, Any]],
    timing_payload: dict[str, Any],
    current_payload: dict[str, Any],
    current_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace only historical render times with base-SHA times from this runner."""
    timing_results = _results(timing_payload)
    timing_ids = _case_ids(timing_results)
    if timing_ids != current_ids:
        raise ValueError("same-run timing baseline case set mismatch")
    if timing_payload.get("measurement_method") != current_payload.get("measurement_method"):
        raise ValueError("same-run timing baseline measurement method mismatch")

    timing_by_id = {str(item["case_id"]): item for item in timing_results}
    combined = deepcopy(historical_baseline)
    ratios_ready = 0
    for item in combined:
        case_id = str(item["case_id"])
        value = (timing_by_id[case_id].get("metrics") or {}).get("render_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"same-run timing baseline is unmeasured: {case_id}")
        item["metrics"]["render_ms"] = float(value)
        ratios_ready += 1
    return combined, {
        "schema_version": "benchmark-same-run-timing-baseline-v1",
        "source": "pull_request_base_sha_same_runner",
        "case_count": ratios_ready,
        "measurement_method": timing_payload.get("measurement_method"),
    }


def _report(
    *,
    status: str,
    reason: str,
    unmeasured: list[dict[str, str]],
    delta: dict[str, Any] | None,
    timing_normalization: dict[str, Any] | None = None,
    timing_baseline: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "benchmark-release-gate-v1",
        "status": status,
        "reason": reason,
        "unmeasured": unmeasured,
        "delta": delta,
        "timing_normalization": timing_normalization,
        "timing_baseline": timing_baseline,
        **extra,
    }


def evaluate_release_gate(
    current_payload: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    timing_baseline_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _results(current_payload)
    unmeasured: list[dict[str, str]] = []
    for case in current:
        case_id = str(case.get("case_id", ""))
        metrics = case.get("metrics") or {}
        for metric in sorted(_required_metrics(case_id)):
            if metric not in metrics or metrics[metric] is None:
                unmeasured.append({"case_id": case_id, "metric": metric})

    if baseline_payload is None:
        return _report(
            status="bootstrap" if not unmeasured else "fail",
            reason="baseline_missing" if not unmeasured else "unmeasured_required_metrics",
            unmeasured=unmeasured,
            delta=None,
        )

    baseline = _results(baseline_payload)
    if unmeasured:
        return _report(
            status="fail",
            reason="unmeasured_required_metrics",
            unmeasured=unmeasured,
            delta=None,
        )

    current_ids = _case_ids(current)
    baseline_ids = _case_ids(baseline)
    if baseline_ids != current_ids:
        added = sorted(current_ids - baseline_ids)
        removed = sorted(baseline_ids - current_ids)
        if baseline_ids < current_ids:
            return _report(
                status="bootstrap",
                reason="case_set_expanded",
                unmeasured=[],
                delta=None,
                case_set={"added": added, "removed": []},
            )
        return _report(
            status="fail",
            reason="case_set_mismatch",
            unmeasured=[],
            delta=None,
            case_set={"added": added, "removed": removed},
        )

    current_method = current_payload.get("measurement_method")
    baseline_method = baseline_payload.get("measurement_method")
    if current_method != baseline_method:
        if not isinstance(current_method, dict) or not str(current_method.get("version", "")).strip():
            return _report(
                status="fail",
                reason="invalid_measurement_method",
                unmeasured=[],
                delta=None,
            )
        return _report(
            status="bootstrap",
            reason="measurement_method_changed",
            unmeasured=[],
            delta=None,
            measurement_method={"baseline": baseline_method, "current": current_method},
        )

    comparison_baseline = baseline
    timing_baseline: dict[str, Any] | None = None
    if timing_baseline_payload is not None:
        try:
            comparison_baseline, timing_baseline = _apply_same_runner_timing_baseline(
                baseline,
                timing_baseline_payload,
                current_payload,
                current_ids,
            )
        except ValueError as exc:
            return _report(
                status="fail",
                reason="invalid_same_runner_timing_baseline",
                unmeasured=[],
                delta=None,
                timing_baseline={"error": str(exc)},
            )

    exclusions = {
        case_id: {"alpha_iou"}
        for case_id in current_ids
        if _category(case_id) not in _ALPHA_RELEVANT_CATEGORIES
    }
    raw_delta = build_delta_report(
        comparison_baseline,
        current,
        excluded_metrics_by_case=exclusions,
        tolerances=_RELEASE_TOLERANCES,
    )

    timing_normalization = None
    compared_current = current
    if timing_baseline is None:
        compared_current, timing_normalization = _normalize_common_mode_timing(
            comparison_baseline,
            current,
            raw_delta,
        )
    delta = (
        build_delta_report(
            comparison_baseline,
            compared_current,
            excluded_metrics_by_case=exclusions,
            tolerances=_RELEASE_TOLERANCES,
        )
        if timing_normalization is not None
        else raw_delta
    )
    return _report(
        status="pass" if delta["status"] == "pass" else "fail",
        reason="within_tolerance" if delta["status"] == "pass" else "metric_regression",
        unmeasured=[],
        delta=delta,
        timing_normalization=timing_normalization,
        timing_baseline=timing_baseline,
    )


def write_gate_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--timing-baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    current = json.loads(args.current.read_text(encoding="utf-8"))
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline is not None and args.baseline.exists()
        else None
    )
    timing_baseline = (
        json.loads(args.timing_baseline.read_text(encoding="utf-8"))
        if args.timing_baseline is not None and args.timing_baseline.exists()
        else None
    )
    report = evaluate_release_gate(current, baseline, timing_baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gate_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"pass", "bootstrap"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
