from __future__ import annotations

from pathlib import Path

from PIL import Image

import app.pipeline_entry as entry


def _result() -> dict:
    return {
        "best": {"name": "prod", "svg_path": Path("prod.svg")},
        "scored": [],
        "mode_used": "logo_color",
        "selection_reason": "highest_fidelity",
    }


def test_flag_off_returns_exact_core_object(monkeypatch, tmp_path):
    core_result = _result()
    monkeypatch.delenv("VEKTORYUM_SHADOW_SELECTOR", raising=False)
    monkeypatch.setattr(entry, "_run_pipeline_core", lambda *a, **k: core_result)

    actual = entry.run_pipeline(
        Image.new("RGB", (2, 2)), tmp_path / "source.png", "auto", tmp_path
    )

    assert actual is core_result
    assert "shadow_telemetry" not in actual


def test_flag_on_preserves_production_winner(monkeypatch, tmp_path):
    core_result = _result()
    winner = core_result["best"]
    monkeypatch.setenv("VEKTORYUM_SHADOW_SELECTOR", "on")
    monkeypatch.setattr(entry, "_run_pipeline_core", lambda *a, **k: core_result)
    monkeypatch.setattr(
        entry,
        "maybe_attach_shadow_telemetry",
        lambda result, audit_path=None: {**result, "shadow_telemetry": {"decision": "agreement"}},
    )

    actual = entry.run_pipeline(
        Image.new("RGB", (2, 2)), tmp_path / "source.png", "auto", tmp_path
    )

    assert actual["best"] is winner
    assert actual["best"]["svg_path"] == Path("prod.svg")
    assert actual["shadow_telemetry"]["decision"] == "agreement"


def test_wrapper_forwards_pipeline_options(monkeypatch, tmp_path):
    seen = {}

    def fake_core(image, original_path, trace_mode, job_dir, refine=True, edge_cleanup=True):
        seen.update(
            original_path=original_path,
            trace_mode=trace_mode,
            job_dir=job_dir,
            refine=refine,
            edge_cleanup=edge_cleanup,
        )
        return _result()

    monkeypatch.delenv("VEKTORYUM_SHADOW_SELECTOR", raising=False)
    monkeypatch.setattr(entry, "_run_pipeline_core", fake_core)
    entry.run_pipeline(
        Image.new("RGB", (2, 2)),
        tmp_path / "source.png",
        "minimal_ai",
        tmp_path,
        refine=False,
        edge_cleanup=False,
    )

    assert seen == {
        "original_path": tmp_path / "source.png",
        "trace_mode": "minimal_ai",
        "job_dir": tmp_path,
        "refine": False,
        "edge_cleanup": False,
    }


def test_job_audit_path_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("VEKTORYUM_SHADOW_AUDIT", raising=False)
    assert entry._audit_path(tmp_path) is None

    monkeypatch.setenv("VEKTORYUM_SHADOW_AUDIT", "job")
    assert entry._audit_path(tmp_path) == tmp_path / "shadow_telemetry.jsonl"

    custom = tmp_path / "global.jsonl"
    monkeypatch.setenv("VEKTORYUM_SHADOW_AUDIT", str(custom))
    assert entry._audit_path(tmp_path) == custom
