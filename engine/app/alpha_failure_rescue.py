"""One-shot engine-diverse rescue after a proven source-alpha pipeline failure.

Ordinary production retains the legacy bounded parent shortlist. Only when the
complete legacy-first + remediation pipeline still raises a fail-closed
``source_alpha_*`` error does this module authorize one additional remediation
pass. That pass skips the already-proven failing legacy pass and includes one
highest-fidelity alternate-engine parent through the existing alpha, evaluator,
byte and TransformJournal gates. No threshold or retry budget is increased.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

_RESCUE_ENABLED: ContextVar[bool] = ContextVar(
    "vektoryum_alpha_failure_rescue_enabled",
    default=False,
)
_INSTALLED = False


def alpha_failure_rescue_enabled() -> bool:
    return bool(_RESCUE_ENABLED.get())


@contextmanager
def alpha_failure_rescue_context(enabled: bool = True) -> Iterator[None]:
    token = _RESCUE_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _RESCUE_ENABLED.reset(token)


def _retryable(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and str(error).startswith("source_alpha_")


def install_alpha_failure_rescue_policy() -> None:
    """Bind ContextVar-backed rescue policy to existing selection/retry helpers."""
    global _INSTALLED
    if _INSTALLED:
        return

    from app import alpha_parent_selection, alpha_pipeline_retry

    original_diversity = alpha_parent_selection._diagnostic_engine_diversity_enabled
    if not getattr(original_diversity, "__vektoryum_rescue_context__", False):
        @wraps(original_diversity)
        def diversity_enabled() -> bool:
            return alpha_failure_rescue_enabled() or bool(original_diversity())

        diversity_enabled.__vektoryum_rescue_context__ = True
        alpha_parent_selection._diagnostic_engine_diversity_enabled = diversity_enabled

    original_dense = alpha_pipeline_retry._has_exact_dense_partial_alpha
    if not getattr(original_dense, "__vektoryum_rescue_context__", False):
        @wraps(original_dense)
        def direct_remediation(source_path: Path, trace_mode: str) -> bool:
            if alpha_failure_rescue_enabled():
                return True
            return bool(original_dense(source_path, trace_mode))

        direct_remediation.__vektoryum_rescue_context__ = True
        alpha_pipeline_retry._has_exact_dense_partial_alpha = direct_remediation

    _INSTALLED = True


def wrap_run_pipeline_with_alpha_failure_rescue(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Retry one direct engine-diverse remediation pass after final alpha failure."""
    if getattr(original, "__vektoryum_alpha_failure_rescue__", False):
        return original

    @wraps(original)
    def rescued(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ) -> dict[str, Any]:
        trigger: str | None = None
        try:
            return original(
                image,
                original_path,
                trace_mode,
                job_dir,
                refine=refine,
                edge_cleanup=edge_cleanup,
            )
        except RuntimeError as error:
            if alpha_failure_rescue_enabled() or not _retryable(error):
                raise
            from app.alpha_pipeline_retry import _has_meaningful_transparency

            if not _has_meaningful_transparency(Path(original_path)):
                raise
            trigger = str(error)

        with alpha_failure_rescue_context(True):
            result = original(
                image,
                original_path,
                trace_mode,
                job_dir,
                refine=refine,
                edge_cleanup=edge_cleanup,
            )
        if isinstance(result, dict):
            result["alpha_failure_rescue_report"] = {
                "status": "engine_diverse_remediation_succeeded",
                "attempt_count": 1,
                "trigger": trigger,
                "shortlist_policy": "legacy_plus_highest_fidelity_alternate_engine",
                "thresholds_changed": False,
            }
        return result

    rescued.__vektoryum_alpha_failure_rescue__ = True
    return rescued
