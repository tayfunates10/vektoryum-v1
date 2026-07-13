import json
from pathlib import Path

from benchmark import pipeline_smoke
from benchmark.seed_runner import CATEGORIES


def test_smoke_uses_all_deterministic_categories(tmp_path: Path, monkeypatch):
    seen = []

    def fake_run_case(case, **kwargs):
        seen.append(case.category)
        return type("R", (), {"case_id": case.case_id, "to_dict": lambda self: {
            "case_id": case.case_id,
            "engine_version": "test",
            "metrics": {
                "fidelity": None, "ssim": None, "edge_f1": None, "alpha_iou": None,
                "delta_e00": None, "path_count": 1, "svg_bytes": 10,
                "render_ms": 1.0, "peak_rss_mb": None,
            },
            "artifact_sha256": None, "failure": None,
        }})()

    monkeypatch.setattr(pipeline_smoke, "run_case", fake_run_case)
    results = pipeline_smoke.run_smoke(tmp_path, engine_version="test")
    assert seen == list(CATEGORIES)
    assert len(results) == len(CATEGORIES) == 8
    payload = json.loads((tmp_path / "pipeline_results.json").read_text())
    assert payload["case_count"] == 8
