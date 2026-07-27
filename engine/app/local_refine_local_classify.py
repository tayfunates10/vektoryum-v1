"""Bit-identical crop-only palette classification for critical-component refit.

``local_refine.refine_critical_components`` reads palette labels only through
rectangular component crops, but the legacy cache classified the complete source or
render for every read. This adapter returns the same lazy crop surface already proven
for counter-merge, so geometry, rounds, thresholds and acceptance decisions stay
unchanged while only the pixels actually compared are classified.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

from app.counter_merge_local_classify import _LazyLocalLabels

_INSTALLED = False


class _LocalRefineClassificationCache:
    """Expose local-refine's cache surface without materializing full label maps."""

    __vektoryum_local_refine_classify__ = True

    def __init__(self, delegate: Any, source_rgb: np.ndarray) -> None:
        self._delegate = delegate
        self._source_rgb = np.asarray(source_rgb)

    def _classify_crop(
        self,
        crop: np.ndarray,
        fills_rgb: np.ndarray,
    ) -> np.ndarray:
        if self._delegate is not None:
            return self._delegate.classify(crop, fills_rgb)
        from app.palette_ops import classify_rgb  # noqa: PLC0415

        return classify_rgb(crop, fills_rgb)

    def classify(
        self,
        rgb: np.ndarray,
        fills_rgb: np.ndarray,
    ) -> _LazyLocalLabels:
        return _LazyLocalLabels(rgb, fills_rgb, self._classify_crop)

    def classify_source(self, fills_rgb: np.ndarray) -> _LazyLocalLabels:
        return _LazyLocalLabels(
            self._source_rgb,
            fills_rgb,
            self._classify_crop,
        )

    def __getattr__(self, name: str) -> Any:
        if self._delegate is None:
            raise AttributeError(name)
        return getattr(self._delegate, name)


def _wrap_refine_critical_components(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_local_refine_classify__", False):
        return original

    @wraps(original)
    def wrapped(
        svg_path: Any,
        source_rgb: np.ndarray,
        width: int,
        height: int,
        render_fn: Any,
        cache: Any = None,
    ) -> dict[str, Any]:
        local_cache = (
            cache
            if getattr(cache, "__vektoryum_local_refine_classify__", False)
            else _LocalRefineClassificationCache(cache, source_rgb)
        )
        return original(
            svg_path,
            source_rgb,
            width,
            height,
            render_fn,
            cache=local_cache,
        )

    wrapped.__vektoryum_local_refine_classify__ = True
    return wrapped


def install_local_refine_local_classify() -> None:
    """Bind the crop-only adapter without modifying local-refine production code."""
    global _INSTALLED
    if _INSTALLED:
        return
    from app import local_refine

    local_refine.refine_critical_components = _wrap_refine_critical_components(
        local_refine.refine_critical_components
    )
    _INSTALLED = True
