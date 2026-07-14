from __future__ import annotations

import hashlib
from pathlib import Path

import analyzer_release_runner as runner
from analyzer_release_contract import AUTO_MODES, CASE_KINDS, REPEAT_COUNT


def _digest(mode: str, name: str) -> str:
    return hashlib.sha256(f"{mode}:{name}".encode("utf-8")).hexdigest()


def test_generated_corpus_is_deterministic_and_complete(tmp_path) -> None:
    first = runner.generate_corpus(tmp_path / "first")
    second = runner.generate_corpus(tmp_path / "second")
    assert len(first) == len(AUTO_MODES) * len(CASE_KINDS)
    assert {(case["label"], case["kind"]) for case in first} == {
        (mode, kind) for mode in AUTO_MODES for kind in CASE_KINDS
    }
    for left, right in zip(first, second):
        assert left["case_id"] == right["case_id"]
        assert left["source_sha256"] == right["source_sha256"]
        assert Path(left["source_path"]).read_bytes() == Path(right["source_path"]).read_bytes()
        assert left["environment"] == "no_hed"


def test_release_runner_writes_valid_report_with_stubbed_samples(monkeypatch, tmp_path) -> None:
    def fake_sample(case, repeat_index, timeout_seconds):
        mode = case["label"]
        return {
            "repeat_index": repeat_index,
            "status": "success",
            "contract_status": "valid",
            "source_pixel_sha256": _digest(mode, "source"),
            "feature_digest": _digest(mode, "feature"),
            "recommendation_digest": _digest(mode, "recommendation"),
            "recommended_mode": mode,
            "decision_status": "accepted",
            "execution_mode": mode,
            "fallback_applied": False,
            "confidence": 0.8,
            "runner_up_mode": "logo_color" if mode != "logo_color" else "minimal_ai",
            "runner_up_margin": 0.2,
            "reason_codes": ["verified_recommendation"],
            "hed_status": "unavailable",
        }

    monkeypatch.setattr(runner, "run_sample", fake_sample)
    report = runner.run_release(tmp_path / "release", repeat_count=REPEAT_COUNT, timeout_seconds=1)
    assert report["verdict"] == "release_ready"
    assert report["errors"] == []
    assert (tmp_path / "release" / "analyzer_release_report.json").is_file()
    assert all(case["deterministic"] for case in report["cases"])
    assert report["metrics"]["accepted_wrong_mode_count"] == 0


def test_release_runner_requires_exact_repeat_count(tmp_path) -> None:
    try:
        runner.run_release(tmp_path, repeat_count=1)
    except ValueError as exc:
        assert "exactly" in str(exc)
    else:
        raise AssertionError("repeat count must be rejected")
