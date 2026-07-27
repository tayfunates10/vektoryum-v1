"""Extend the compact alpha-budget retry to every accepted alpha representation.

The original RFV-3D3 hook registered retry context only when the accepted source-alpha
builder already reported ``candidate_geometry_knockout``. Live RFV-3B evidence showed
that an alternate accepted representation can still reach the exact same
``source_alpha_vector_mask`` journal budget rejection. This adapter records that
accepted context as well; the existing compact retry remains authoritative and must
still pass byte, artwork identity, alpha, final-evaluator and unchanged journal gates.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

from app import alpha_budget_retry as _retry
from app import alpha_svg_mask

_INSTALLED = False


def _wrap_register_accepted_alpha(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Register any applied source-alpha result for a later budget-only retry."""
    if getattr(original, "__vektoryum_alpha_budget_retry_all_accepted__", False):
        return original

    @wraps(original)
    def wrapped(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        report = original(Path(svg_path), Path(source_path), mode)
        if (
            isinstance(report, dict)
            and report.get("status") == "accepted"
            and report.get("applied") is True
        ):
            _retry._register(
                Path(svg_path),
                _retry._PendingKnockout(Path(source_path), str(mode), report),
            )
        return report

    wrapped.__vektoryum_alpha_budget_retry_all_accepted__ = True
    return wrapped


def install_alpha_budget_retry_scope() -> None:
    """Install the idempotent accepted-alpha context hook after the base retry."""
    global _INSTALLED
    if _INSTALLED:
        return
    alpha_svg_mask.apply_source_alpha_mask = _wrap_register_accepted_alpha(
        alpha_svg_mask.apply_source_alpha_mask
    )
    _INSTALLED = True
