from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import app.pipeline_canonical_report as canonical
import app.pipeline_entry as entry


def _pipeline_result() -> dict:
    return {
        "best": {"name": "legacy", "svg_path": Path("legacy.svg")},
        "selection_reason": "highest_fidelity",
        "scored": [],
    }


def _ready_candidate():
    return SimpleNamespace(
        valid=True,
        document=SimpleNamespace(document_sha256="a" * 64, face_count=2),
        promotion=SimpleNamespace(ready=True),
        palette_size=3,
        errors=(),
    )


def test_disabled_gate_returns_exact_production_object(monkeypatch) -> None:
    original = _pipeline_result()

    def should_not_run(*args, **kwargs):
        raise AssertionError("canonical builder must not run while disabled")

    monkeypatch.setattr(canonical, "build_canonical_svg_candidate", should_not_run)
    actual = canonical.maybe_attach_canonical_svg_candidate(
        original,
        Image.new("RGB", (4, 4), "white"),
        env={},
    )

    assert actual is original
    assert "canonical_svg_candidate" not in actual


def test_enabled_gate_attaches_ready_report_without_replacing_winner(monkeypatch) -> None:
    original = _pipeline_result()
    winner = original["best"]
    seen = {}

    def fake_builder(image, **kwargs):
        seen.update(size=image.size, kwargs=kwargs)
        return _ready_candidate()

    monkeypatch.setattr(canonical, "build_canonical_svg_candidate", fake_builder)
    actual = canonical.maybe_attach_canonical_svg_candidate(
        original,
        Image.new("RGB", (6, 5), "white"),
        env={"VEKTORYUM_CANONICAL_CANDIDATE_ENABLED": "on"},
    )

    report = actual["canonical_svg_candidate"]
    assert actual is not original
    assert actual["best"] is winner
    assert actual["best"]["svg_path"] == Path("legacy.svg")
    assert report.ready is True
    assert report.status == "ready"
    assert report.document_sha256 == "a" * 64
    assert report.face_count == 2
    assert report.palette_size == 3
    assert seen == {
        "size": (6, 5),
        "kwargs": {"max_colors": 32, "repeat_runs": 3, "max_pixels": 16_000_000},
    }


def test_invalid_configuration_fails_closed_before_builder(monkeypatch) -> None:
    called = False

    def fake_builder(*args, **kwargs):
        nonlocal called
        called = True
        return _ready_candidate()

    monkeypatch.setattr(canonical, "build_canonical_svg_candidate", fake_builder)
    report = canonical.build_pipeline_canonical_svg_report(
        Image.new("RGB", (4, 4), "white"),
        env={
            "VEKTORYUM_CANONICAL_CANDIDATE_ENABLED": "on",
            "VEKTORYUM_CANONICAL_CANDIDATE_MAX_COLORS": "1",
        },
    )

    assert called is False
    assert report.ready is False
    assert report.status == "configuration_error"
    assert report.candidate is None
    assert report.document_sha256 == ""
    assert report.errors == (
        "VEKTORYUM_CANONICAL_CANDIDATE_MAX_COLORS must be between 2 and 64",
    )


def test_builder_exception_is_isolated_from_production(monkeypatch) -> None:
    original = _pipeline_result()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(canonical, "build_canonical_svg_candidate", explode)
    actual = canonical.maybe_attach_canonical_svg_candidate(
        original,
        Image.new("RGB", (4, 4), "white"),
        env={"VEKTORYUM_CANONICAL_CANDIDATE_ENABLED": "true"},
    )

    report = actual["canonical_svg_candidate"]
    assert actual["best"] is original["best"]
    assert report.ready is False
    assert report.status == "builder_error"
    assert report.candidate is None
    assert report.errors == ("canonical candidate builder failed: RuntimeError",)


def test_non_promotable_candidate_cannot_be_marked_ready(monkeypatch) -> None:
    invalid = SimpleNamespace(
        valid=False,
        document=None,
        promotion=SimpleNamespace(ready=False),
        palette_size=4,
        errors=("graph invariant failed",),
    )
    monkeypatch.setattr(canonical, "build_canonical_svg_candidate", lambda *a, **k: invalid)

    report = canonical.build_pipeline_canonical_svg_report(
        Image.new("RGB", (4, 4), "white"),
        env={"VEKTORYUM_CANONICAL_CANDIDATE_ENABLED": "enabled"},
    )

    assert report.ready is False
    assert report.status == "invalid"
    assert report.document_sha256 == ""
    assert report.face_count == 0
    assert report.errors == ("graph invariant failed",)


def test_public_pipeline_facade_invokes_canonical_gate_after_core(monkeypatch, tmp_path) -> None:
    core_result = _pipeline_result()
    image = Image.new("RGB", (3, 2), "white")
    seen = {}

    monkeypatch.delenv("VEKTORYUM_SHADOW_SELECTOR", raising=False)
    monkeypatch.setattr(entry, "_run_pipeline_core", lambda *a, **k: core_result)

    def fake_attach(result, received_image):
        seen.update(result=result, image=received_image)
        return {**result, "canonical_svg_candidate": "attached"}

    monkeypatch.setattr(entry, "maybe_attach_canonical_svg_candidate", fake_attach)
    actual = entry.run_pipeline(image, tmp_path / "source.png", "auto", tmp_path)

    assert seen["result"] is core_result
    assert seen["image"] is image
    assert actual["best"] is core_result["best"]
    assert actual["canonical_svg_candidate"] == "attached"
