import json
from pathlib import Path

import pytest

from benchmark import pipeline_smoke
from benchmark.manifest import BenchmarkResult, REQUIRED_METRICS


def _result(case_id: str) -> BenchmarkResult:
    metrics = {name: 1.0 for name in REQUIRED_METRICS}
    metrics.update({"render_ms": 2.0, "peak_rss_mb": 12.0})
    return BenchmarkResult(
        case_id=case_id,
        engine_version="test",
        metrics=metrics,
        artifact_sha256="a" * 64,
    )


def test_repeat_samples_are_written_for_successes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        pipeline_smoke,
        "_run_case_isolated",
        lambda case, **kwargs: _result(case.case_id),
    )
    pipeline_smoke.run_smoke(
        tmp_path,
        engine_version="test",
        repeat_count=3,
        timeout_seconds=45,
    )
    payload = json.loads((tmp_path / "repeat_samples.json").read_text())
    assert payload["schema_version"] == "benchmark-repeat-samples-v1"
    assert payload["sample_count"] == 24
    assert payload["repeat_count"] == 3
    assert payload["timeout_seconds"] == 45
    assert all(sample["status"] == "success" for sample in payload["samples"])


def test_repeat_failure_writes_partial_diagnostics(tmp_path: Path, monkeypatch):
    calls = 0

    def fake(case, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("synthetic timeout")
        return _result(case.case_id)

    monkeypatch.setattr(pipeline_smoke, "_run_case_isolated", fake)
    with pytest.raises(TimeoutError, match="synthetic timeout"):
        pipeline_smoke.run_smoke(
            tmp_path,
            engine_version="test",
            repeat_count=3,
            timeout_seconds=5,
        )
    payload = json.loads((tmp_path / "repeat_samples.json").read_text())
    assert payload["sample_count"] == 2
    assert payload["samples"][0]["status"] == "success"
    assert payload["samples"][1]["status"] == "failure"
    assert "TimeoutError" in payload["samples"][1]["error"]


def test_timeout_must_be_positive(tmp_path: Path):
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        pipeline_smoke.run_smoke(tmp_path, engine_version="test", timeout_seconds=0)
