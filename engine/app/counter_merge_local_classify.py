"""Counter-merge palet kapısı için bit-birebir yerel sınıflandırma adaptörü.

``counter_merge.merge_counters`` sınıflandırma haritalarını yalnız aday örtmenin
``[ya:yb, xa:xb]`` kutusunda okur. Eski yol buna rağmen kaynak, önceki render ve
aday render'ın TAM tuvalini yüzlerce dolgu merkezine sınıflandırıyordu. RFV-3D3
profili public-11/public-12'de saatin çoğunun bu NumPy çağrılarında geçtiğini,
render ve tracer sürelerinin ise saniye mertebesinde kaldığını ölçtü.

Sınıflandırma piksel-bağımsızdır. Bu adaptör, aynı ``classify_rgb`` işlemini ancak
ilk dilim istendiğinde o dilimin RGB piksellerine uygular. Dolayısıyla:

* çıktı ``classify_rgb(full, fills)[crop]`` ile bit-birebir aynıdır;
* sayaç aday sırası, geometri, eşikler ve kabul bilançosu değişmez;
* mevcut ``RefinementCache`` varsa küçük kırpım onun üzerinden memoize edilir;
* counter-merge dışındaki çağrılar ve production varsayılanları etkilenmez.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_INSTALLED = False


def _slice_key(key: Any, shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Counter-merge'in iki boyutlu dikdörtgen dilimini doğrula ve normalize et."""
    if not isinstance(key, tuple) or len(key) < 2:
        raise TypeError("yerel sınıflandırma iki boyutlu dilim bekler")
    y_key, x_key = key[0], key[1]
    if not isinstance(y_key, slice) or not isinstance(x_key, slice):
        raise TypeError("yerel sınıflandırma yalnız dikdörtgen dilim destekler")
    y0, y1, y_step = y_key.indices(shape[0])
    x0, x1, x_step = x_key.indices(shape[1])
    if y_step != 1 or x_step != 1:
        raise ValueError("yerel sınıflandırma adımı 1 olmalı")
    return slice(y0, y1), slice(x0, x1)


class _LazyLocalLabels:
    """Tam label haritası yerine istenen dikdörtgeni sınıflandırır."""

    def __init__(
        self,
        rgb: np.ndarray,
        fills_rgb: np.ndarray,
        classify_crop: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self._rgb = np.asarray(rgb)
        self._fills_rgb = np.asarray(fills_rgb)
        self._classify_crop = classify_crop
        self._local_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self._rgb.shape[:2]

    def __getitem__(self, key: Any) -> np.ndarray:
        ys, xs = _slice_key(key, self._rgb.shape)
        cache_key = (int(ys.start), int(ys.stop), int(xs.start), int(xs.stop))
        cached = self._local_cache.get(cache_key)
        if cached is not None:
            return cached
        crop = np.ascontiguousarray(self._rgb[ys, xs])
        labels = self._classify_crop(crop, self._fills_rgb)
        self._local_cache[cache_key] = labels
        return labels


class _LocalClassificationCache:
    """Counter-merge'in beklediği ``cache.classify`` yüzeyini yerelleştirir."""

    __vektoryum_local_counter_classify__ = True

    def __init__(self, delegate: Any = None) -> None:
        self._delegate = delegate

    def _classify_crop(
        self,
        crop: np.ndarray,
        fills_rgb: np.ndarray,
    ) -> np.ndarray:
        if self._delegate is not None:
            return self._delegate.classify(crop, fills_rgb)
        from app.palette_ops import classify_rgb  # noqa: PLC0415

        return classify_rgb(crop, fills_rgb)

    def classify(self, rgb: np.ndarray, fills_rgb: np.ndarray) -> _LazyLocalLabels:
        return _LazyLocalLabels(rgb, fills_rgb, self._classify_crop)


def _wrap_merge_counters(original: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_local_counter_classify__", False):
        return original

    @wraps(original)
    def wrapped(
        svg_path: Any,
        width: int,
        height: int,
        render_fn: Any,
        source_rgb: np.ndarray | None = None,
        max_merges: int = 8,
        cache: Any = None,
    ) -> dict[str, Any]:
        local_cache = (
            cache
            if getattr(cache, "__vektoryum_local_counter_classify__", False)
            else _LocalClassificationCache(cache)
        )
        return original(
            svg_path,
            width,
            height,
            render_fn,
            source_rgb=source_rgb,
            max_merges=max_merges,
            cache=local_cache,
        )

    wrapped.__vektoryum_local_counter_classify__ = True
    return wrapped


def install_counter_merge_local_classify() -> None:
    """Counter-merge modülünü idempotent biçimde yerel sınıflandırmaya bağla."""
    global _INSTALLED
    if _INSTALLED:
        return
    from app import counter_merge

    counter_merge.merge_counters = _wrap_merge_counters(counter_merge.merge_counters)
    _INSTALLED = True
