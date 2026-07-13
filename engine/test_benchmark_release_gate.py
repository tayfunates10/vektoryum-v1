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


def test_regression_fails():
    baseline = _payload()
    current = _payload()
    current["results"][0]["metrics"]["fidelity"] = 0.0
    report = evaluate_release_gate(current, baseline)
    assert report["status"] == "fail"
    assert report["reason"] == "metric_regression"


def test_unmeasured_metric_fails_before_delta_comparison():
    report = evaluate_release_gate(_payload(missing="alpha_iou"), _payload())
    assert report["status"] == "fail"
    assert report["delta"] is None


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