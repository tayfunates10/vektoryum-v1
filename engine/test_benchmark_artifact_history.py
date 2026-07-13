import json
from pathlib import Path

from benchmark.artifact_history import compare_seed_manifests, write_reports


def _manifest(*pairs: tuple[str, str]) -> dict:
    return {
        "schema_version": "benchmark-seed-v1",
        "cases": [
            {"case_id": case_id, "source_sha256": digest, "category": "logos"}
            for case_id, digest in pairs
        ],
    }


def test_bootstrap_report_when_no_baseline() -> None:
    report = compare_seed_manifests(None, _manifest(("a", "1" * 64)))
    assert report["status"] == "bootstrap"
    assert report["baseline_available"] is False
    assert report["added"] == ["a"]


def test_stable_and_changed_reports_are_deterministic() -> None:
    baseline = _manifest(("b", "2" * 64), ("a", "1" * 64))
    stable = compare_seed_manifests(baseline, _manifest(("a", "1" * 64), ("b", "2" * 64)))
    assert stable["status"] == "stable"

    changed = compare_seed_manifests(
        baseline,
        _manifest(("a", "9" * 64), ("c", "3" * 64)),
    )
    assert changed["status"] == "changed"
    assert changed["added"] == ["c"]
    assert changed["removed"] == ["b"]
    assert changed["changed"] == ["a"]


def test_json_and_html_reports_are_written(tmp_path: Path) -> None:
    report = compare_seed_manifests(None, _manifest(("a", "1" * 64)))
    write_reports(tmp_path, report)
    payload = json.loads((tmp_path / "artifact_delta.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "benchmark-artifact-history-v1"
    assert "Benchmark Artifact History" in (tmp_path / "artifact_delta.html").read_text(encoding="utf-8")
