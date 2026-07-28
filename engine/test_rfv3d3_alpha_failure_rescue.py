"""Contracts for one-shot source-alpha failure rescue orchestration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.alpha_failure_rescue import (
    alpha_failure_rescue_enabled,
    wrap_run_pipeline_with_alpha_failure_rescue,
)


def _call(wrapped, tmp_path: Path):
    return wrapped(
        object(),
        tmp_path / "source.png",
        "auto",
        tmp_path,
        refine=True,
        edge_cleanup=True,
    )


def test_successful_pipeline_is_not_retried(tmp_path: Path) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append(alpha_failure_rescue_enabled())
        return {"status": "ok"}

    result = _call(wrap_run_pipeline_with_alpha_failure_rescue(original), tmp_path)

    assert result == {"status": "ok"}
    assert calls == [False]


def test_source_alpha_failure_gets_one_context_scoped_rescue(tmp_path: Path) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append(alpha_failure_rescue_enabled())
        if len(calls) == 1:
            raise RuntimeError("source_alpha_mask_transform_gate_rejected:seam_regression")
        return {"status": "ok"}

    wrapped = wrap_run_pipeline_with_alpha_failure_rescue(original)
    with patch("app.alpha_pipeline_retry._has_meaningful_transparency", return_value=True):
        result = _call(wrapped, tmp_path)

    assert calls == [False, True]
    assert result["status"] == "ok"
    report = result["alpha_failure_rescue_report"]
    assert report["status"] == "engine_diverse_remediation_succeeded"
    assert report["attempt_count"] == 1
    assert report["thresholds_changed"] is False
    assert alpha_failure_rescue_enabled() is False


def test_non_alpha_error_never_retries(tmp_path: Path) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append(alpha_failure_rescue_enabled())
        raise RuntimeError("ordinary_pipeline_failure")

    wrapped = wrap_run_pipeline_with_alpha_failure_rescue(original)
    with pytest.raises(RuntimeError, match="ordinary_pipeline_failure"):
        _call(wrapped, tmp_path)

    assert calls == [False]


def test_rescue_failure_propagates_without_third_attempt(tmp_path: Path) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append(alpha_failure_rescue_enabled())
        raise RuntimeError("source_alpha_candidate_knockout_parent_not_collapsed")

    wrapped = wrap_run_pipeline_with_alpha_failure_rescue(original)
    with (
        patch("app.alpha_pipeline_retry._has_meaningful_transparency", return_value=True),
        pytest.raises(RuntimeError, match="parent_not_collapsed"),
    ):
        _call(wrapped, tmp_path)

    assert calls == [False, True]
    assert alpha_failure_rescue_enabled() is False
