"""Accepted-representation scope for RFV-3D3 alpha budget retry."""
from __future__ import annotations

import tempfile
from pathlib import Path

from app import alpha_budget_retry as retry
from app import alpha_budget_retry_scope as scope


def setup_function() -> None:
    retry._PENDING.set({})


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
