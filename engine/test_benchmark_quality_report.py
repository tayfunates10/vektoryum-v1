import json
from pathlib import Path

import pytest

from benchmark.quality_report import build_quality_summary, write_reports
from benchmark.manifest import REQUIRED_METRICS


def _payload():
    base = {name: 1.0 for name in REQUIRED_METRICS}
    transparent = dict(base)
    transparent.update({"alpha_iou": 0.5, "render_ms": 20000.0, "fidelity": 90.0})
    opaque = dict(base)
    opaque.update({"alpha_iou": 0.1, "render_ms": 1000.0, "fidelity": 99.0})
    return {
        "schema_version": "benchmark-results-v1",
        "case_count": 2,
        "results": [
            {"case_id": "seed-02-transparent", "metrics": transparent, "failure": None},
            {"case_id": "seed-01-logos", "metrics": opaque, "failure": None},
        ],
    }


def test_summary_is_deterministic_and_surfaces_weak_cases():
    summary = build_quality_summary(_payload())
    assert [row["case_id"] for row in summary["rows"]] == ["seed-01-logos", "seed-02-transparent"]
    assert summary["case_count"] == 2
    assert summary["worst_cases"]["fidelity"][0]["case_id"] == "seed-02-transparent"
    assert {a["metric"] for a in summary["alerts"]} == {"alpha_iou", "render_ms"}


def test_alpha_alerts_and_ranking_only_use_transparent_cases():
    summary = build_quality_summary(_payload())
    rows = {row["case_id"]: row for row in summary["rows"]}
    assert rows["seed-01-logos"]["alpha_applicable"] is False
    assert rows["seed-02-transparent"]["alpha_applicable"] is True
    assert summary["worst_cases"]["alpha_iou"] == [
        {"case_id": "seed-02-transparent", "value": 0.5}
    ]
    assert not any(
        alert["case_id"] == "seed-01-logos" and alert["metric"] == "alpha_iou"
        for alert in summary["alerts"]
    )


def test_report_files_are_written(tmp_path: Path):
    summary = build_quality_summary(_payload())
    write_reports(tmp_path, summary)
    payload = json.loads((tmp_path / "quality_summary.json").read_text())
    assert payload["schema_version"] == "benchmark-quality-summary-v1"
    html = (tmp_path / "quality_summary.html").read_text()
    assert "Benchmark Quality Summary" in html
    assert "n/a" in html


def test_invalid_schema_fails_closed():
    with pytest.raises(ValueError):
        build_quality_summary({"schema_version": "wrong", "results": []})
