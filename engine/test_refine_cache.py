"""RefinementCache birim regresyonları: izolasyon, stale güvenliği, hit, bellek.

Çalıştırma::  .venv/bin/python test_refine_cache.py   (~5 sn, pipeline'sız)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _svg(color: str, size: int = 64) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}"><rect width="{size}" height="{size}" '
            f'fill="{color}"/></svg>')


def test_render_hit() -> None:
    print("== Render cache hit (aynı içerik tek render) ==")
    from app.refine_cache import RefinementCache

    src = np.zeros((64, 64, 3), np.uint8)
    c = RefinementCache(src)
    calls = {"n": 0}
    from app import fidelity
    orig = fidelity.render_svg_to_rgb

    def counting(p, w, h):
        calls["n"] += 1
        return orig(p, w, h)

    fidelity.render_svg_to_rgb = counting
    try:
        f = Path(tempfile.mkstemp(suffix=".svg")[1])
        f.write_text(_svg("#e3000b"))
        a = c.render(f, 64, 64)
        b = c.render(f, 64, 64)
        d = c.render(f, 64, 64)
        check(a is not None, "render üretildi")
        check(calls["n"] == 1, f"gerçek render yalnız 1 kez çağrıldı ({calls['n']})")
        check(b is a and d is a, "sonraki çağrılar önbellekten (aynı nesne)")
        check(c.render_hits == 2 and c.render_misses == 1, "hit/miss sayaçları doğru")
        f.unlink(missing_ok=True)
    finally:
        fidelity.render_svg_to_rgb = orig


def test_stale_safety() -> None:
    print("== Stale güvenliği (içerik değişince yeni render) ==")
    from app.refine_cache import RefinementCache

    src = np.zeros((64, 64, 3), np.uint8)
    c = RefinementCache(src)
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(_svg("#e3000b"))
    a = c.render(f, 64, 64)
    f.write_text(_svg("#00a651"))  # AYNI dosya yolu, farklı içerik
    b = c.render(f, 64, 64)
    check(a is not None and b is not None, "iki render de üretildi")
    check(not np.array_equal(a, b), "içerik değişti: render farklı (stale döndürülmedi)")
    check(c.render_misses == 2, "iki miss (hash farkı stale hit'i engelledi)")
    f.unlink(missing_ok=True)


def test_isolation() -> None:
    print("== İstek izolasyonu (iki bağlam karışmaz) ==")
    from app.refine_cache import RefinementCache

    src1 = np.zeros((64, 64, 3), np.uint8)
    src2 = np.full((64, 64, 3), 255, np.uint8)
    c1 = RefinementCache(src1)
    c2 = RefinementCache(src2)
    fills = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32)
    l1 = c1.classify_source(fills)
    l2 = c2.classify_source(fills)
    check(int(l1.sum()) == 0, "bağlam1 kaynağı sınıf 0")
    check(int(l2.sum()) == l2.size, "bağlam2 kaynağı sınıf 1 (karışma yok)")
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(_svg("#000000"))
    c1.render(f, 64, 64)
    check(len(c2._render_lru) == 0, "bağlam2 render önbelleği bağlam1'den etkilenmedi")
    f.unlink(missing_ok=True)


def test_classify_source_once() -> None:
    print("== Kaynak sınıflandırması istekte bir kez ==")
    from app import palette_ops
    from app.refine_cache import RefinementCache

    src = np.random.RandomState(3).randint(0, 256, (128, 128, 3), np.uint8)
    c = RefinementCache(src)
    fills = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255]], dtype=np.float32)
    calls = {"n": 0}
    orig = palette_ops.classify_rgb

    def counting(img, f):
        calls["n"] += 1
        return orig(img, f)

    palette_ops.classify_rgb = counting
    try:
        a = c.classify_source(fills)
        b = c.classify_source(fills)
        c.classify_source(fills)
        check(calls["n"] == 1, f"kaynak sınıflandırması 1 kez ({calls['n']})")
        check(b is a, "aynı label dizisi döndü")
        check(c.src_cls_hits == 2, "src hit sayacı")
    finally:
        palette_ops.classify_rgb = orig


def test_memory_lru() -> None:
    print("== LRU bellek sınırı (eviction) ==")
    from app.refine_cache import RefinementCache

    src = np.zeros((64, 64, 3), np.uint8)
    c = RefinementCache(src, max_renders=2)
    for i, col in enumerate(("#e3000b", "#00a651", "#0000ff", "#ffff00")):
        f = Path(tempfile.mkstemp(suffix=".svg")[1])
        f.write_text(_svg(col))
        c.render(f, 64, 64)
        f.unlink(missing_ok=True)
    check(len(c._render_lru) == 2, f"LRU en çok 2 giriş tutar ({len(c._render_lru)})")
    check(c.render_evictions == 2, f"2 eviction ({c.render_evictions})")
    c.close()
    check(len(c._render_lru) == 0, "close() önbelleği boşalttı")


def main() -> int:
    test_render_hit()
    test_stale_safety()
    test_isolation()
    test_classify_source_once()
    test_memory_lru()
    print("=" * 60)
    if FAILS:
        print(f"SONUC: {len(FAILS)} KONTROL BASARISIZ")
        for m in FAILS:
            print(" -", m)
        return 1
    print("SONUC: tum kontroller gecti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
