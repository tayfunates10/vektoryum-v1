"""Legacy-first orchestration for source-alpha remediation.

Transparent inputs must not change the established trace candidate unless the
selected legacy artifact actually fails the source-alpha finalization contract.
The first production pass therefore bypasses alpha-specific preprocessing and
parent reselection. A single bounded retry enables those remediation stages only
for a fail-closed ``source_alpha_*`` rejection on a meaningfully transparent
source.

A narrowly proven dense partial-alpha source is the only exception: when the
source plane itself can be represented by the exact, budgeted hybrid-polygon
encoder, the historical first pass would deterministically be discarded and the
entire vectorization repeated. That source-only preflight authorizes one enabled
pass directly; every quality, topology, byte, path and node gate remains active.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

_ALPHA_REMEDIATION_ENABLED: ContextVar[bool] = ContextVar(
    "vektoryum_alpha_remediation_enabled",
    default=True,
)


def alpha_remediation_enabled() -> bool:
    """Return whether alpha-specific preprocessing/selection may run."""
    return bool(_ALPHA_REMEDIATION_ENABLED.get())


@contextmanager
def alpha_remediation_context(enabled: bool) -> Iterator[None]:
    """Temporarily select legacy or remediation behavior for one pipeline pass."""
    token = _ALPHA_REMEDIATION_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ALPHA_REMEDIATION_ENABLED.reset(token)


def _historical_callable(original: Callable[..., Any]) -> Callable[..., Any]:
    historical = getattr(original, "__wrapped__", None)
    return historical if callable(historical) else original


def wrap_preprocess_with_alpha_context(
    original: Callable[..., tuple[Path, dict[str, Any]]],
) -> Callable[..., tuple[Path, dict[str, Any]]]:
    """Use the historical preprocess during the legacy-first production pass."""
    if getattr(original, "__vektoryum_alpha_contextual__", False):
        return original
    historical = _historical_callable(original)

    @wraps(original)
    def contextual(*args, **kwargs):
        target = original if alpha_remediation_enabled() else historical
        return target(*args, **kwargs)

    contextual.__vektoryum_alpha_contextual__ = True
    contextual.__vektoryum_alpha_historical__ = historical
    return contextual


def wrap_gradient_with_alpha_context(
    original: Callable[..., None],
) -> Callable[..., None]:
    """Preserve the historical gradient input until remediation is required."""
    if getattr(original, "__vektoryum_alpha_contextual__", False):
        return original
    historical = _historical_callable(original)

    @wraps(original)
    def contextual(*args, **kwargs):
        target = original if alpha_remediation_enabled() else historical
        return target(*args, **kwargs)

    contextual.__vektoryum_alpha_contextual__ = True
    contextual.__vektoryum_alpha_historical__ = historical
    return contextual


def _has_meaningful_transparency(source_path: Path) -> bool:
    try:
        with Image.open(Path(source_path)) as source:
            low, high = source.convert("RGBA").getchannel("A").getextrema()
    except (OSError, ValueError):
        return False
    return int(low) < 255 and int(high) > 0


def _has_exact_dense_partial_alpha(source_path: Path, trace_mode: str) -> bool:
    """Return true only when the existing exact hybrid-polygon plan is available.

    This is not a filename, mode-quality or heuristic shortcut. The same source
    alpha plane, quantizer and geometry planner used by the publishing encoder are
    executed before tracing. Any parse, size, level or polygon rejection preserves
    the normal legacy-first behavior. ``auto`` is eligible because its inner
    analyzer still resolves to one of the existing alpha-mask execution modes
    before finalization; all publishing gates remain authoritative there.
    """
    try:
        from app.alpha_mask_adaptive import _dense_polygon_plan  # noqa: PLC0415
        from app.alpha_svg_mask import _ALPHA_MASK_MODES, _quantize_alpha  # noqa: PLC0415

        if trace_mode != "auto" and trace_mode not in _ALPHA_MASK_MODES:
            return False
        with Image.open(Path(source_path)) as source:
            alpha = np.asarray(
                source.convert("RGBA").getchannel("A"),
                dtype=np.uint8,
            ).copy()
        if not np.any(alpha > 0) or bool(np.all(alpha == 255)):
            return False
        quantized, opacity_by_level = _quantize_alpha(alpha)
        return _dense_polygon_plan(quantized, opacity_by_level) is not None
    except (OSError, ValueError, RuntimeError, TypeError, ImportError):
        return False


def _retryable_alpha_failure(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and str(error).startswith("source_alpha_")


def wrap_run_pipeline_with_alpha_retry(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Run legacy bytes first and retry alpha remediation at most once.

    The first result is returned untouched when it succeeds. This is the key
    non-regression property: no alpha staging, alternate-parent selection or
    second trace can alter an already-valid artifact. Only a source-alpha
    fail-closed rejection on a transparent source authorizes the second pass.

    Dense partial alpha whose exact hybrid-polygon plan is already proven runs one
    remediation-enabled pass directly. The output is the same guarded path that a
    deterministic legacy rejection would select, without repeating vectorization.
    """
    if getattr(original, "__vektoryum_alpha_retry_wrapped__", False):
        return original

    @wraps(original)
    def legacy_first(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ) -> dict[str, Any]:
        source_path = Path(original_path)
        if _has_exact_dense_partial_alpha(source_path, trace_mode):
            with alpha_remediation_context(True):
                result = original(
                    image,
                    original_path,
                    trace_mode,
                    job_dir,
                    refine=refine,
                    edge_cleanup=edge_cleanup,
                )
            if isinstance(result, dict):
                result["alpha_retry_report"] = {
                    "status": "dense_preflight_remediation_succeeded",
                    "attempt_count": 1,
                    "first_pass": "exact_dense_alpha_preflight",
                    "first_rejection": None,
                    "retry": "not_required",
                }
            return result

        first_error: RuntimeError | None = None
        with alpha_remediation_context(False):
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
                if not _retryable_alpha_failure(error):
                    raise
                if not _has_meaningful_transparency(source_path):
                    raise
                first_error = error

        try:
            with alpha_remediation_context(True):
                result = original(
                    image,
                    original_path,
                    trace_mode,
                    job_dir,
                    refine=refine,
                    edge_cleanup=edge_cleanup,
                )
        except Exception as retry_error:
            raise retry_error from first_error

        if isinstance(result, dict):
            result["alpha_retry_report"] = {
                "status": "remediation_succeeded",
                "attempt_count": 2,
                "first_pass": "legacy_alpha_preprocess_bypassed",
                "first_rejection": str(first_error),
                "retry": "alpha_remediation_enabled",
            }
        return result

    legacy_first.__vektoryum_alpha_retry_wrapped__ = True
    return legacy_first


def _install_soft_ellipse_direct_factory() -> None:
    """Bind compact alpha specializations before ``app.__init__`` captures them."""
    from app import alpha_candidate_direct  # noqa: PLC0415
    from app.alpha_soft_ellipse import make_soft_ellipse_alpha_first  # noqa: PLC0415
    from app.alpha_visible_paint import make_visible_alpha_paint_repair  # noqa: PLC0415

    current = alpha_candidate_direct.make_direct_element_alpha_first
    if getattr(current, "__vektoryum_soft_ellipse_factory__", False):
        return

    @wraps(current)
    def soft_ellipse_factory(guarded_builder):
        assembled = make_soft_ellipse_alpha_first(current(guarded_builder))
        return make_visible_alpha_paint_repair(assembled)

    soft_ellipse_factory.__vektoryum_soft_ellipse_factory__ = True
    alpha_candidate_direct.make_direct_element_alpha_first = soft_ellipse_factory


_install_soft_ellipse_direct_factory()
