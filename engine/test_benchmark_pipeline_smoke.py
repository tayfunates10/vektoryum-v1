import json
from pathlib import Path

import pytest

from benchmark import pipeline_smoke
from benchmark.manifest import BenchmarkResult, REQUIRED_METRICS
from benchmark.seed_runner import CATEGORIES


def _result(case_id: str, *, render_ms: float, peak_rss_mb: float, fidelity: float = 99.0, sha: str | None = None):
    metrics = {name: 1.0 for name in REQUIRED_METRICS}
    metrics.update({
        "fidelity": fidelity,
        "ssim": 0.99,
        "edge_f1": 0.98,
        "alpha_iou": 0.97,
        "delta_e00": 1.0,
        "path_count": 4,
        "svg_bytes": 100,
        "render_ms": render_ms,
        "peak_rss_mb": peak_rss_mb,
    })
    return BenchmarkResult(case_id=case_id, engine_version="test", metrics=metrics, artifact_sha256=sha)


def test_smoke_uses_all_deterministic_categories(tmp_path: Path, monkeypatch):
    seen = []

    def fake_isolated(case, **kwargs):
        seen.append(case.category)
        return _result(case.case_id, render_ms=1.0, peak_rss_mb=10.0)

    monkeypatch.setattr(pipeline_smoke, "_run_case_isolated", fake_isolated)
    results = pipeline_smoke.run_smoke(tmp_path, engine_version="test", repeat_count=3)
    assert seen == [category for category in CATEGORIES for _ in range(3)]
    assert len(results) == len(CATEGORIES) == 8
    payload = json.loads((tmp_path / "pipeline_results.json").read_text())
    assert payload["case_count"] == 8
    assert payload["measurement_method"]["version"] == "median-performance-v2-isolated-rss"
    assert payload["measurement_method"]["rss_scope"] == "fresh_spawned_process_per_repeat"


def test_repeats_use_median_performance_and_conservative_quality():
    repeats = [
        _result("seed-01-logos", render_ms=100.0, peak_rss_mb=50.0, fidelity=99.0),
        _result("seed-01-logos", render_ms=500.0, peak_rss_mb=70.0, fidelity=98.0),
        _result("seed-01-logos", render_ms=110.0, peak_rss_mb=55.0, fidelity=99.5),
    ]
    result = pipeline_smoke.aggregate_repeats(repeats)
    assert result.metrics["render_ms"] == 110.0
    assert result.metrics["peak_rss_mb"] == 55.0
    assert result.metrics["fidelity"] == 98.0


def test_non_deterministic_artifact_sha_fails_closed():
    with pytest.raises(ValueError, match="non-deterministic artifact"):
        pipeline_smoke.aggregate_repeats([
            _result("seed-01-logos", render_ms=1.0, peak_rss_mb=1.0, sha="a" * 64),
            _result("seed-01-logos", render_ms=1.0, peak_rss_mb=1.0, sha="b" * 64),
            _result("seed-01-logos", render_ms=1.0, peak_rss_mb=1.0, sha="a" * 64),
        ])


def test_repeat_count_must_be_positive_and_odd(tmp_path: Path):
    with pytest.raises(ValueError, match="positive odd"):
        pipeline_smoke.run_smoke(tmp_path, engine_version="test", repeat_count=2)


def test_isolated_repeat_without_payload_fails_closed(monkeypatch, tmp_path: Path):
    class FakeProcess:
        exitcode = 3
        def start(self): pass
        def join(self, timeout=None): pass
        def is_alive(self): return False

    class FakeQueue:
        def empty(self): return True

    class FakeContext:
        def Queue(self, maxsize=1): return FakeQueue()
        def Process(self, target, args): return FakeProcess()

    monkeypatch.setattr(pipeline_smoke.multiprocessing, "get_context", lambda mode: FakeContext())
    case = type("Case", (), {"case_id": "seed-01-logos", "to_dict": lambda self: {}})()
    with pytest.raises(RuntimeError, match="exited without a result"):
        pipeline_smoke._run_case_isolated(
            case,
            corpus_root=tmp_path,
            work_root=tmp_path / "work",
            engine_version="test",
        )
