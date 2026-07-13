import json
from pathlib import Path

import pytest

from benchmark.quality_report import build_quality_summary, write_reports
from benchmark.manifest import REQUIRED_METRICS


def _payload():
    base = {name: 1.0 for name in REQUIRED_METRICS}
    slow = dict(base)
    slow.update({"alpha_iou": 0.5, "render_ms": 20000.0, "fidelity": 90.0})
    good = dict(base)
    good.update({"alpha_iou": 0.99, "render_ms": 1000.0, "fidelity": 99.0})
    return {
        "schema_version": "benchmark-results-v1",
        "case_count": 2,
        "results": [
            {"case_id": "seed-02-transparent", "metrics": slow, "failure": None},
            {"case_id": "seed-01-logos", "metrics": good, "failure": None},
        ],
    }


def test_summary_is_deterministic_and_surfaces_weak_cases():
    summary = build_quality_summary(_payload())
    assert [row["case_id"] for row in summary["rows"]] == ["seed-01-logos", "seed-02-transparent"]
    assert summary["case_count"] == 2
    assert summary["worst_cases"]["fidelity"][0]["case_id"] == "seed-02-transparent"
    assert {a["metric"] for a in summary["alerts"]} == {"alpha_iou", "render_ms"}


def test_report_files_are_written(tmp_path: Path):
    summary = build_quality_summary(_payload())
    write_reports(tmp_path, summary)
    payload = json.loads((tmp_path / "quality_summary.json").read_text())
    assert payload["schema_version"] == "benchmark-quality-summary-v1"
    assert "Benchmark Quality Summary" in (tmp_path / "quality_summary.html").read_text()


def test_invalid_schema_fails_closed():
    with pytest.raises(ValueError):
        build_quality_summary({"schema_version": "wrong", "results": []})
