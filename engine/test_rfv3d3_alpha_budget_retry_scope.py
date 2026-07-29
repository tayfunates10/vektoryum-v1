"""Accepted-representation, painter-handoff and probe-budget RFV-3D3 contracts."""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import alpha_budget_retry as retry
from app import alpha_budget_retry_scope as scope
from app import painter_measurement_reuse as reuse


def setup_function() -> None:
    retry._PENDING.set({})
    reuse._clear_scope()


def test_any_applied_accepted_alpha_representation_registers_context() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = root / "candidate.svg"
        source = root / "source.png"
        candidate.write_text("<svg/>", encoding="utf-8")
        source.write_bytes(b"source")
        report = {
            "status": "accepted",
            "applied": True,
            "mask_encoding": "rect",
        }

        def original(svg_path, source_path, mode):
            assert Path(svg_path) == candidate
            assert Path(source_path) == source
            assert mode == "logo_color"
            return report

        wrapped = scope._wrap_register_accepted_alpha(original)
        assert wrapped(candidate, source, "logo_color") is report
        pending = retry._take(candidate)
        assert pending is not None
        assert pending.source_path == source
        assert pending.mode == "logo_color"
        assert pending.report is report


def test_non_applied_or_rejected_report_does_not_register() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = root / "candidate.svg"
        source = root / "source.png"
        candidate.write_text("<svg/>", encoding="utf-8")
        source.write_bytes(b"source")

        for report in (
            {"status": "accepted", "applied": False},
            {"status": "rejected", "applied": True},
        ):
            wrapped = scope._wrap_register_accepted_alpha(
                lambda *_args, _report=report: _report
            )
            assert wrapped(candidate, source, "logo_color") is report
            assert retry._take(candidate) is None


def test_scope_wrapper_is_idempotent() -> None:
    def original(*_args):
        return {"status": "accepted", "applied": True}

    once = scope._wrap_register_accepted_alpha(original)
    twice = scope._wrap_register_accepted_alpha(once)
    assert twice is once


def test_direct_child_listing_is_not_a_false_processing_budget() -> None:
    root = ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + "".join('<path fill="#000" d="M0 0h1v1Z"/>' for _ in range(858))
        + "</svg>"
    )
    indexes = scope._direct_renderable_indexes(root)
    assert len(indexes) == 858
    assert indexes[0] == 0
    assert indexes[-1] == 857


def test_direct_probe_budget_counts_only_measured_children() -> None:
    measured = scope._wrap_direct_alpha_mode(lambda *_args, **_kwargs: 255)

    def within_budget():
        return [measured(None, None) for _ in range(512)]

    wrapped_within = scope._wrap_direct_candidate_build(within_budget)
    assert len(wrapped_within()) == 512
    assert scope._DIRECT_PROBE_COUNT.get() is None

    def over_budget():
        return [measured(None, None) for _ in range(513)]

    wrapped_over = scope._wrap_direct_candidate_build(over_budget)
    with pytest.raises(
        RuntimeError,
        match=r"source_alpha_direct_probe_budget_exceeded:513>512",
    ):
        wrapped_over()
    assert scope._DIRECT_PROBE_COUNT.get() is None


def _cache(parent: Path, candidate: Path, source: np.ndarray) -> dict:
    namespace = reuse._namespace(source, 512, set())
    result = {}
    for path, seconds in ((parent, 2.0), (candidate, 3.0)):
        sha = reuse._sha_path(path)
        key = reuse._metric_key(namespace, sha)
        result[key] = {"sha256": sha}
        result[f"{key}\x00seconds"] = seconds
    return result


def _journal(source: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        source_rgb=source.copy(),
        image_class="clean_logo",
        required_metrics=set(),
        max_side=512,
        _verified_source_alpha_contract=lambda stage, report: (
            stage == "source_alpha_vector_mask"
            and report.get("status") == "accepted"
            and report.get("applied") is True
        ),
    )


def test_painter_measurement_handoff_survives_report_assembly_and_is_one_shot() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "parent.svg"
        candidate = root / "candidate.svg"
        parent.write_bytes(b"<svg><path/></svg>")
        candidate.write_bytes(b"<svg><path/><mask/></svg>")
        source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        report = {
            "schema": "rfv3d2-candidate-painter-reconstruction-v1",
            "status": "accepted",
            "applied": True,
            "candidate_geometry_preserved": True,
        }
        cache = _cache(parent, candidate, source)

        scope._register_painter_handoff(
            parent,
            candidate,
            source,
            "clean_logo",
            set(),
            512,
            cache,
            report,
        )
        handoff = report.get("journal_measurement_handoff_id")
        assert isinstance(handoff, str) and len(handoff) == 64

        # The real painter assembles a larger final report around the verified
        # assessment dictionary. Benign added telemetry must not break lineage.
        final_report = {
            "before_byte_size": parent.stat().st_size,
            "after_byte_size": candidate.stat().st_size,
            "painter_attempt_count": 7,
            **report,
        }
        proof = scope._take_painter_handoff(
            _journal(source), parent, candidate, final_report
        )
        assert proof is not None
        assert proof.cache is cache
        assert proof.measured_seconds == 5.0

        # Consumption is one-shot; no later request may inherit the cache.
        assert scope._take_painter_handoff(
            _journal(source), parent, candidate, final_report
        ) is None


def test_painter_handoff_rejects_identity_or_lineage_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "parent.svg"
        candidate = root / "candidate.svg"
        parent.write_bytes(b"parent")
        candidate.write_bytes(b"candidate")
        source = np.zeros((4, 4, 3), dtype=np.uint8)
        report = {"status": "accepted", "applied": True}
        scope._register_painter_handoff(
            parent,
            candidate,
            source,
            "clean_logo",
            set(),
            512,
            _cache(parent, candidate, source),
            report,
        )
        report["journal_measurement_handoff_id"] = "0" * 64
        assert scope._take_painter_handoff(
            _journal(source), parent, candidate, report
        ) is None

        reuse._clear_scope()
        report = {"status": "accepted", "applied": True}
        scope._register_painter_handoff(
            parent,
            candidate,
            source,
            "clean_logo",
            set(),
            512,
            _cache(parent, candidate, source),
            report,
        )
        candidate.write_bytes(b"changed")
        assert scope._take_painter_handoff(
            _journal(source), parent, candidate, report
        ) is None
