from __future__ import annotations

from pathlib import Path

from PIL import Image

import app.pipeline_entry as entry


def test_pipeline_entry_registers_final_attached_result(monkeypatch, tmp_path) -> None:
    core = {
        "best": {"name": "legacy", "svg_path": Path("legacy.svg")},
        "scored": [],
    }
    attached = {**core, "canonical_svg_candidate": object()}
    seen = {}

    monkeypatch.setattr(entry, "_run_pipeline_core", lambda *a, **k: core)
    monkeypatch.setattr(entry, "maybe_attach_shadow_telemetry", lambda result, audit_path=None: result)
    monkeypatch.setattr(entry, "maybe_attach_canonical_svg_candidate", lambda result, image: attached)

    def register(job_dir, result):
        seen.update(job_dir=job_dir, result=result)
        return True

    monkeypatch.setattr(entry, "register_pipeline_canonical_report", register)
    image = Image.new("RGB", (3, 3), "white")
    actual = entry.run_pipeline(image, tmp_path / "source.png", "auto", tmp_path)

    assert actual is attached
    assert seen["job_dir"] == tmp_path
    assert seen["result"] is attached
