"""Contracts for the painter journal compatibility capture binding."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import painter_measurement_capture as capture


class _FakeJournal:
    accept = True

    def __init__(self, baseline, source_rgb, *, image_class, required_metrics):
        self.baseline_path = Path(baseline)
        self.source_rgb = source_rgb
        self.image_class = image_class
        self.required_metrics = required_metrics
        self.max_side = 512
        self._cache = {}

    def consider_candidate(
        self,
        stage_id,
        parent_path,
        candidate_path,
        *,
        transform_report=None,
    ):
        assert stage_id == "source_alpha_painter_candidate"
        assert Path(parent_path) == self.baseline_path
        if self.accept:
            return Path(candidate_path), {"status": "accepted", "reason_codes": []}
        return Path(parent_path), {"status": "rolled_back", "reason_codes": ["seam_regression"]}


def test_passing_capture_registers_exact_proof(tmp_path: Path) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_text("<svg/>", encoding="utf-8")
    candidate.write_text("<svg><path/></svg>", encoding="utf-8")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    report = {"schema": "proof", "status": "accepted"}
    external_cache = {}
    _FakeJournal.accept = True

    with (
        patch("app.transform_journal.TransformJournal", _FakeJournal),
        patch.object(capture._reuse, "_register") as register,
    ):
        passed, reasons = capture.run_painter_geometry_journal_with_capture(
            parent,
            candidate,
            source,
            "clean_logo",
            report,
            measurement_cache=external_cache,
        )

    assert passed is True
    assert reasons == []
    register.assert_called_once()
    args = register.call_args.args
    assert args[0] == parent
    assert args[1] == candidate
    assert args[6] is external_cache
    assert args[7] is report


def test_rejected_capture_never_registers(tmp_path: Path) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_text("<svg/>", encoding="utf-8")
    candidate.write_text("<svg><path/></svg>", encoding="utf-8")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    _FakeJournal.accept = False

    with (
        patch("app.transform_journal.TransformJournal", _FakeJournal),
        patch.object(capture._reuse, "_register") as register,
    ):
        passed, reasons = capture.run_painter_geometry_journal_with_capture(
            parent,
            candidate,
            source,
            "clean_logo",
            {"schema": "proof"},
        )

    assert passed is False
    assert reasons == ["seam_regression"]
    register.assert_not_called()
