from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _payload(value: float = 1.0) -> dict:
    metrics = {
        "fidelity": value,
        "ssim": value,
        "edge_f1": value,
        "alpha_iou": value,
        "delta_e00": 0.0,
        "path_count": 1,
        "svg_bytes": 100,
        "render_ms": 10.0,
        "peak_rss_mb": 50.0,
    }
    return {
        "schema_version": "benchmark-results-v1",
        "case_count": 1,
        "results": [{"case_id": "case-1", "engine_version": "test", "metrics": metrics, "artifact_sha256": "a" * 64}],
    }


def test_cli_bootstrap_and_pass(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "gate.json"
    current.write_text(json.dumps(_payload()), encoding="utf-8")
    baseline.write_text(json.dumps(_payload()), encoding="utf-8")

    bootstrap = subprocess.run(
        [sys.executable, "-m", "benchmark.release_gate", "--current", str(current), "--output", str(output)],
        check=False,
        cwd=Path(__file__).parent,
    )
    assert bootstrap.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "bootstrap"

    passed = subprocess.run(
        [sys.executable, "-m", "benchmark.release_gate", "--current", str(current), "--baseline", str(baseline), "--output", str(output)],
        check=False,
        cwd=Path(__file__).parent,
    )
    assert passed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_cli_fails_for_unmeasured_metric(tmp_path: Path) -> None:
    payload = _payload()
    payload["results"][0]["metrics"]["ssim"] = None
    current = tmp_path / "current.json"
    output = tmp_path / "gate.json"
    current.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "benchmark.release_gate", "--current", str(current), "--output", str(output)],
        check=False,
        cwd=Path(__file__).parent,
    )
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"
