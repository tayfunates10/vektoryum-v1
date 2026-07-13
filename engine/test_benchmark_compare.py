import pytest

from benchmark.compare import build_delta_report, compare_metrics
from benchmark.manifest import REQUIRED_METRICS


def _metrics(**overrides):
    values = {
        "fidelity": 0.98,
        "ssim": 0.97,
        "edge_f1": 0.96,
        "alpha_iou": 0.99,
        "delta_e00": 1.0,
        "path_count": 100,
        "svg_bytes": 10000,
        "render_ms": 1000,
        "peak_rss_mb": 500,
    }
    values.update(overrides)
    return values


def test_compare_metrics_classifies_quality_and_cost_directions():
    deltas = {item.metric: item for item in compare_metrics(
        _metrics(),
        _metrics(fidelity=0.99, delta_e00=0.5, path_count=120),
    )}
    assert deltas["fidelity"].status == "improvement"
    assert deltas["delta_e00"].status == "improvement"
    assert deltas["path_count"].status == "regression"


def test_compare_metrics_is_fail_closed_for_missing_or_unmeasurable_values():
    current = _metrics()
    current.pop("ssim")
    with pytest.raises(ValueError, match="missing benchmark metrics"):
        compare_metrics(_metrics(), current)

    current = _metrics(ssim=None)
    with pytest.raises(ValueError, match="not measurable"):
        compare_metrics(_metrics(), current)


def test_delta_report_is_deterministic_and_fails_on_any_case_regression():
    baseline = [
        {"case_id": "b", "metrics": _metrics()},
        {"case_id": "a", "metrics": _metrics()},
    ]
    current = [
        {"case_id": "a", "metrics": _metrics()},
        {"case_id": "b", "metrics": _metrics(render_ms=1200)},
    ]
    report = build_delta_report(baseline, current)
    assert report["schema_version"] == "benchmark-delta-v1"
    assert report["status"] == "fail"
    assert report["regression_count"] == 1
    assert [case["case_id"] for case in report["cases"]] == ["a", "b"]


def test_delta_report_rejects_case_set_mismatch():
    with pytest.raises(ValueError, match="case mismatch"):
        build_delta_report(
            [{"case_id": "a", "metrics": _metrics()}],
            [{"case_id": "b", "metrics": _metrics()}],
        )


def test_metric_contract_stays_fully_covered():
    assert set(REQUIRED_METRICS) == {
        "fidelity", "ssim", "edge_f1", "alpha_iou", "delta_e00",
        "path_count", "svg_bytes", "render_ms", "peak_rss_mb",
    }
