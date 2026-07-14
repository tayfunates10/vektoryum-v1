import pytest

from benchmark.release_gate import evaluate_release_gate
from benchmark.manifest import REQUIRED_METRICS


def _payload(value: float = 1.0, *, missing: str | None = None, case_ids=("case-1",)):
    results = []
    for case_id in case_ids:
        metrics = {name: value for name in REQUIRED_METRICS}
        if missing:
            metrics[missing] = None
        results.append({"case_id": case_id, "engine_version": "v1", "metrics": metrics})
    return {
        "schema_version": "benchmark-results-v1",
        "case_count": len(results),
        "results": results,
    }


def test_bootstrap_passes_only_when_all_required_metrics_are_measured():
    assert evaluate_release_gate(_payload(), None)["status"] == "bootstrap"
    report = evaluate_release_gate(_payload(missing="ssim"), None)
    assert report["status"] == "fail"
    assert report["reason"] == "unmeasured_required_metrics"


def test_stable_results_pass():
    report = evaluate_release_gate(_payload(), _payload())
    assert report["status"] == "pass"
    assert report["reason"] == "within_tolerance"
    assert report["timing_normalization"] is None


def test_regression_fails():
    baseline = _payload()
    current = _payload()
    current["results"][0]["metrics"]["fidelity"] = 0.0
    report = evaluate_release_gate(current, baseline)
    assert report["status"] == "fail"
    assert report["reason"] == "metric_regression"


def test_unmeasured_metric_fails_before_delta_comparison():
    report = evaluate_release_gate(_payload(missing="ssim"), _payload())
    assert report["status"] == "fail"
    assert report["delta"] is None


def test_opaque_case_does_not_require_or_compare_alpha_iou():
    baseline = _payload(case_ids=("seed-01-logos",))
    current = _payload(case_ids=("seed-01-logos",))
    baseline["results"][0]["metrics"]["alpha_iou"] = None
    current["results"][0]["metrics"]["alpha_iou"] = 0.0
    report = evaluate_release_gate(current, baseline)
    assert report["status"] == "pass"
    case = report["delta"]["cases"][0]
    assert case["excluded_metrics"] == ["alpha_iou"]
    assert "alpha_iou" not in {item["metric"] for item in case["metrics"]}


def test_transparent_case_still_requires_alpha_iou():
    report = evaluate_release_gate(
        _payload(missing="alpha_iou", case_ids=("seed-02-transparent",)),
        None,
    )
    assert report["status"] == "fail"
    assert report["unmeasured"] == [{"case_id": "seed-02-transparent", "metric": "alpha_iou"}]


def test_measured_case_set_expansion_bootstraps_once():
    report = evaluate_release_gate(
        _payload(case_ids=("case-1", "case-2")),
        _payload(case_ids=("case-1",)),
    )
    assert report["status"] == "bootstrap"
    assert report["reason"] == "case_set_expanded"
    assert report["case_set"] == {"added": ["case-2"], "removed": []}


def test_case_removal_remains_fail_closed():
    report = evaluate_release_gate(
        _payload(case_ids=("case-1",)),
        _payload(case_ids=("case-1", "case-2")),
    )
    assert report["status"] == "fail"
    assert report["reason"] == "case_set_mismatch"
    assert report["case_set"]["removed"] == ["case-2"]


def _eight_cases() -> tuple[str, ...]:
    return tuple(f"seed-{index:02d}-category{index}" for index in range(1, 9))


def test_bounded_common_mode_hosted_runner_slowdown_is_normalized():
    case_ids = _eight_cases()
    baseline = _payload(case_ids=case_ids)
    current = _payload(case_ids=case_ids)
    for item in current["results"]:
        item["metrics"]["render_ms"] = 1.30

    report = evaluate_release_gate(current, baseline)

    assert report["status"] == "pass"
    normalization = report["timing_normalization"]
    assert normalization["applied"] is True
    assert normalization["factor"] == pytest.approx(1.30)
    assert normalization["slow_case_count"] == 8
    assert normalization["raw_regression_count"] == 8


def test_isolated_timing_regression_remains_fail_closed():
    case_ids = _eight_cases()
    baseline = _payload(case_ids=case_ids)
    current = _payload(case_ids=case_ids)
    current["results"][0]["metrics"]["render_ms"] = 1.30

    report = evaluate_release_gate(current, baseline)

    assert report["status"] == "fail"
    assert report["timing_normalization"] is None


def test_quality_regression_is_never_hidden_by_timing_normalization():
    case_ids = _eight_cases()
    baseline = _payload(case_ids=case_ids)
    current = _payload(case_ids=case_ids)
    for item in current["results"]:
        item["metrics"]["render_ms"] = 1.30
    current["results"][0]["metrics"]["fidelity"] = 0.0

    report = evaluate_release_gate(current, baseline)

    assert report["status"] == "fail"
    assert report["timing_normalization"] is None


def test_extreme_common_mode_slowdown_exceeds_safety_cap():
    case_ids = _eight_cases()
    baseline = _payload(case_ids=case_ids)
    current = _payload(case_ids=case_ids)
    for item in current["results"]:
        item["metrics"]["render_ms"] = 1.60

    report = evaluate_release_gate(current, baseline)

    assert report["status"] == "fail"
    assert report["timing_normalization"] is None
