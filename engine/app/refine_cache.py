"""İstek-kapsamlı render / sınıflandırma önbelleği (9.85–9.89 aşamaları).

Kalite aşamaları aynı SVG'yi ve aynı sınıflandırma sonucunu tekrar tekrar
üretiyordu (ölçüldü: 3840² fixture'da ~10 render, ~14 sınıflandırma; büyük
kısmı yinelenen). resvg render'ı 3840²'de saniyeler sürer — asıl darboğaz
budur. Bu modül TEK İSTEK boyunca yaşayan, istek bitince serbest bırakılan
bir önbellek sağlar.

Güvenlik ilkeleri:
* Anahtar SVG İÇERİK HASH'idir (blake2b), geometry_version DEĞİL: yanlışlıkla
  aynı versiyonla farklı içerik gelirse hash bunu yakalar, stale sonuç dönmez.
* Global/paylaşımlı değildir: her istek kendi örneğini kurar; kullanıcılar
  arası sızıntı imkânsız.
* LRU + bellek bütçesi: en çok N tam render tutulur (3840² RGB = ~44 MB);
  aşımda en eski atılır. İstek bitince ``close()`` tüm referansları bırakır.
* Sınıflandırma önbelleği render içerik-hash'i + palet hash'iyle anahtarlanır;
  aynı raster + aynı palet ikinci kez sınıflandırılmaz.

``render`` metodu, aşamaların beklediği ``Callable[[Path,int,int], ndarray]``
imzasına uyar: mevcut ``render_fn`` yerine doğrudan geçirilebilir.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 3840² RGB ~ 44 MB; 2 giriş ~ 88 MB. Sınıflandırma label haritası uint8
# ~14.7 MB; 6 giriş ~ 88 MB. Toplam önbellek tepe ~180 MB — bellek bütçesi
# (≤1800 MB) içinde; 3 render tepeyi 1800'ün üstüne çıkarıyordu (ölçüldü).
_DEFAULT_MAX_RENDERS = 2
_DEFAULT_MAX_CLS = 6


def _hash_bytes(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()


class RefinementCache:
    """Tek istek için render + sınıflandırma memoizasyonu (LRU, bellek sınırlı)."""

    def __init__(self, source_rgb: np.ndarray, max_renders: int = _DEFAULT_MAX_RENDERS,
                 max_cls: int = _DEFAULT_MAX_CLS) -> None:
        self.source_rgb = source_rgb
        self._max_renders = max(1, max_renders)
        self._max_cls = max(1, max_cls)
        self._render_lru: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._render_hash: dict[tuple, str] = {}
        self._cls_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._src_cls_cache: dict[str, np.ndarray] = {}
        self.cls_evictions = 0
        # metrikler
        self.render_calls = 0
        self.render_hits = 0
        self.render_misses = 0
        self.render_evictions = 0
        self.cls_calls = 0
        self.cls_hits = 0
        self.cls_misses = 0
        self.src_cls_calls = 0
        self.src_cls_hits = 0
        self._real_render = None  # geç bağlanır (döngüsel import kaçınma)

    # -- render ------------------------------------------------------------
    def render(self, svg_path: Path, width: int, height: int) -> np.ndarray | None:
        """SVG içeriğini render eder; aynı içerik+boyut için önbellekten döner."""
        self.render_calls += 1
        if self._real_render is None:
            from app.fidelity import render_svg_to_rgb  # noqa: PLC0415

            self._real_render = render_svg_to_rgb
        try:
            content = Path(svg_path).read_bytes()
        except OSError:
            return self._real_render(svg_path, width, height)
        h = _hash_bytes(content)
        key = (h, int(width), int(height))
        cached = self._render_lru.get(key)
        if cached is not None:
            self._render_lru.move_to_end(key)
            self.render_hits += 1
            return cached
        self.render_misses += 1
        arr = self._real_render(svg_path, width, height)
        if arr is not None:
            self._render_lru[key] = arr
            self._render_hash[key] = h
            while len(self._render_lru) > self._max_renders:
                self._render_lru.popitem(last=False)
                self.render_evictions += 1
        return arr

    # -- sınıflandırma -----------------------------------------------------
    def classify(self, rgb: np.ndarray, fills_rgb: np.ndarray) -> np.ndarray:
        """RGB'yi en yakın dolguya sınıflar; render önbelleğiyle memoize eder.

        Anahtar: rgb'nin içerik-hash'i + palet hash'i. Büyük dizi hash'i
        pahalı olduğundan, önce dizinin önbellekteki bir render OLUP olmadığı
        kimlikle (is) aranır; değilse tam hash hesaplanır.
        """
        self.cls_calls += 1
        from app.palette_ops import classify_rgb  # noqa: PLC0415

        rid = None
        for key, arr in self._render_lru.items():
            if arr is rgb:
                rid = self._render_hash.get(key)
                break
        if rid is None:
            rid = _hash_bytes(np.ascontiguousarray(rgb).tobytes())
        fkey = _hash_bytes(np.ascontiguousarray(fills_rgb).tobytes())
        ck = (rid, fkey)
        cached = self._cls_cache.get(ck)
        if cached is not None:
            self._cls_cache.move_to_end(ck)
            self.cls_hits += 1
            return cached
        self.cls_misses += 1
        labels = classify_rgb(rgb, fills_rgb)
        self._cls_cache[ck] = labels
        while len(self._cls_cache) > self._max_cls:
            self._cls_cache.popitem(last=False)
            self.cls_evictions += 1
        return labels

    def classify_source(self, fills_rgb: np.ndarray) -> np.ndarray:
        """Kaynak görüntünün sınıflandırması: istek boyunca bir kez hesaplanır."""
        self.src_cls_calls += 1
        fkey = _hash_bytes(np.ascontiguousarray(fills_rgb).tobytes())
        cached = self._src_cls_cache.get(fkey)
        if cached is not None:
            self.src_cls_hits += 1
            return cached
        from app.palette_ops import classify_rgb  # noqa: PLC0415

        labels = classify_rgb(self.source_rgb, fills_rgb)
        self._src_cls_cache[fkey] = labels
        return labels

    # -- yaşam döngüsü -----------------------------------------------------
    def close(self) -> None:
        self._render_lru.clear()
        self._render_hash.clear()
        self._cls_cache.clear()
        self._src_cls_cache.clear()

    def __enter__(self) -> RefinementCache:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        return {
            "render_calls": self.render_calls,
            "render_hits": self.render_hits,
            "render_misses": self.render_misses,
            "render_evictions": self.render_evictions,
            "cls_calls": self.cls_calls,
            "cls_hits": self.cls_hits,
            "cls_misses": self.cls_misses,
            "cls_evictions": self.cls_evictions,
            "src_cls_calls": self.src_cls_calls,
            "src_cls_hits": self.src_cls_hits,
            "render_cache_size": len(self._render_lru),
            "render_cache_bytes": sum(a.nbytes for a in self._render_lru.values()),
        }
