from __future__ import annotations

from app.alpha_pipeline_retry import _retryable_alpha_failure


def test_evaluation_budget_exhaustion_is_terminal() -> None:
    error = RuntimeError(
        "source_alpha_mask_transform_gate_rejected:evaluation_budget_exhausted"
    )
    assert _retryable_alpha_failure(error) is False


def test_other_source_alpha_failure_remains_retryable() -> None:
    error = RuntimeError("source_alpha_mask_transform_gate_rejected:seam_regression")
    assert _retryable_alpha_failure(error) is True
