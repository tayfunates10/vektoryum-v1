"""Benchmark baseline comparison and regression classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.manifest import REQUIRED_METRICS

HIGHER_IS_BETTER = {"fidelity", "ssim", "edge_f1", "alpha_iou"}
LOWER_IS_BETTER = {"delta_e00", "path_count", "svg_bytes", "render_ms", "peak_rss_mb"}
DEFAULT_TOLERANCES = {
    "fidelity": 0.002,
    "ssim": 0.002,
    "edge_f1": 0.003,
    "alpha_iou": 0.003,
    "delta_e00": 0.25,
    "path_count": 0.05,
    "svg_bytes": 0.05,
    "render_ms": 0.10,
    "peak_rss_mb": 0.10,
}


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float
    relative_delta: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "status": self.status,
        }


def _relative_delta(baseline: float, current: float) -> float | None:
    if baseline == 0:
        return None
    return (current - baseline) / abs(baseline)


def compare_metrics(
    baseline: dict[str, float | int | None],
    current: dict[str, float | int | None],
    *,
    tolerances: dict[str, float] | None = None,
    metric_names: set[str] | frozenset[str] | None = None,
) -> list[MetricDelta]:
    selected = set(REQUIRED_METRICS if metric_names is None else metric_names)
    unsupported = selected.difference(REQUIRED_METRICS)
    if unsupported:
        raise ValueError(f"unsupported benchmark metrics: {sorted(unsupported)}")
    missing = selected.difference(baseline) | selected.difference(current)
    if missing:
        raise ValueError(f"missing benchmark metrics: {sorted(missing)}")
    limits = dict(DEFAULT_TOLERANCES)
    if tolerances:
        limits.update(tolerances)

    deltas: list[MetricDelta] = []
    for metric in sorted(selected):
        before = baseline[metric]
        after = current[metric]
        if before is None or after is None:
            raise ValueError(f"metric {metric} is not measurable")
        before_f = float(before)
        after_f = float(after)
        delta = after_f - before_f
        relative = _relative_delta(before_f, after_f)
        tolerance = float(limits[metric])

        if metric in HIGHER_IS_BETTER:
            if delta < -tolerance:
                status = "regression"
            elif delta > tolerance:
                status = "improvement"
            else:
                status = "stable"
        elif metric in LOWER_IS_BETTER:
            comparison = relative if relative is not None else delta
            if comparison > tolerance:
                status = "regression"
            elif comparison < -tolerance:
                status = "improvement"
            else:
                status = "stable"
        else:
            raise ValueError(f"unsupported benchmark metric: {metric}")

        deltas.append(MetricDelta(metric, before_f, after_f, delta, relative, status))
    return deltas


def build_delta_report(
    baseline_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
    *,
    excluded_metrics_by_case: dict[str, set[str] | frozenset[str]] | None = None,
) -> dict[str, Any]:
    baseline_by_id = {item["case_id"]: item for item in baseline_results}
    current_by_id = {item["case_id"]: item for item in current_results}
    if set(baseline_by_id) != set(current_by_id):
        missing = sorted(set(baseline_by_id) ^ set(current_by_id))
        raise ValueError(f"benchmark case mismatch: {missing}")

    exclusions = excluded_metrics_by_case or {}
    cases = []
    regression_count = 0
    for case_id in sorted(baseline_by_id):
        excluded = set(exclusions.get(case_id, set()))
        unsupported = excluded.difference(REQUIRED_METRICS)
        if unsupported:
            raise ValueError(f"unsupported excluded metrics for {case_id}: {sorted(unsupported)}")
        selected = set(REQUIRED_METRICS).difference(excluded)
        deltas = compare_metrics(
            baseline_by_id[case_id]["metrics"],
            current_by_id[case_id]["metrics"],
            metric_names=selected,
        )
        status = "regression" if any(item.status == "regression" for item in deltas) else "pass"
        regression_count += status == "regression"
        cases.append({
            "case_id": case_id,
            "status": status,
            "excluded_metrics": sorted(excluded),
            "metrics": [d.to_dict() for d in deltas],
        })

    return {
        "schema_version": "benchmark-delta-v1",
        "case_count": len(cases),
        "regression_count": regression_count,
        "status": "fail" if regression_count else "pass",
        "cases": cases,
    }
