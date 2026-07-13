from benchmark.release_gate import evaluate_release_gate
from benchmark.manifest import REQUIRED_METRICS


def _payload(value: float = 1.0, *, missing: str | None = None):
    metrics = {name: value for name in REQUIRED_METRICS}
    if missing:
        metrics[missing] = None
    return {
        "schema_version": "benchmark-results-v1",
        "case_count": 1,
        "results": [{"case_id": "case-1", "engine_version": "v1", "metrics": metrics}],
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
