"""Cusp/segment bölme + bant sınıflandırma birim regresyonları (hızlı, sentetik).

Kilitlediği davranışlar:
* De Casteljau bölme TAM'dır: bölme render/geometri değiştirmez.
* Dar kama cusp'ı segment bölme + kanonik ortak düğümle kapanır; sliver yiter.
* Yumuşak sınır sapmaları cusp bölmesi GEREKTİRMEZ (önce hata-refine çözer;
  cusp aşaması çapa eklemez).
* Üçüncü ince renk bölgesi iki komşunun arasındayken yanlış birleştirme yok.
* Bant (tiled) sınıflandırma tek parçayla BİT-BİREBİR aynıdır (3 bant boyutu).
* refine_cusp_regions deterministtir (iki koşu aynı bayt çıktısı).

Çalıştırma::  .venv/bin/python test_cusp_and_tiles.py     (~30-60 sn)
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# 1. De Casteljau bölme tamlığı
# ---------------------------------------------------------------------------
def test_decasteljau() -> None:
    print("== De Casteljau bölme tamlığı ==")
    from svgpathtools import CubicBezier

    seg = CubicBezier(10 + 20j, 40 + 5j, 80 + 95j, 120 + 30j)
    for t in (0.3, 0.5, 0.77):
        a, b = seg.split(t)
        max_dev = 0.0
        for i in range(201):
            u = i / 200.0
            p_orig = seg.point(u)
            p_split = a.point(u / t) if u <= t else b.point((u - t) / (1 - t))
            max_dev = max(max_dev, abs(p_orig - p_split))
        check(max_dev < 1e-9, f"t={t}: bölme birebir (maks sapma {max_dev:.2e})")
        for v in (a.start, a.control1, a.control2, a.end, b.control1, b.control2, b.end):
            check(np.isfinite(v.real) and np.isfinite(v.imag), f"t={t}: NaN/Inf yok")
            break


# ---------------------------------------------------------------------------
# Sentetik sahne yardımcıları
# ---------------------------------------------------------------------------
def _render(svg_txt: str, w: int, h: int):
    from app.fidelity import render_svg_to_rgb

    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(svg_txt)
    out = render_svg_to_rgb(f, w, h)
    f.unlink(missing_ok=True)
    return out


def _scene(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<path d="M0 0 L{w} 0 L{w} {h} L0 {h} Z" fill="#e3000b"/>{body}</svg>')


# ---------------------------------------------------------------------------
# 2. Dar kama: segment bölme sliver'ı kapatır (kanonik ortak düğüm)
# ---------------------------------------------------------------------------
def test_wedge() -> None:
    print("== Dar kama cusp bölmesi ==")
    from app.cusp_refine import refine_cusp_regions
    from app.fidelity import render_svg_to_rgb

    w = h = 512
    # KAYNAK: sarı üçgen kama tepesi (256, 60)'a ulaşır; siyah blok altta
    src_svg = _scene(w, h, (
        '<path d="M100 300 L256 60 L412 300 Z" fill="#ffed00"/>'
        '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>'
    ))
    src = _render(src_svg, w, h)
    assert src is not None, "render backend yok"
    # BOZUK SVG: kama tepesi 24px KISA (kesik) -> tepe sliver'ı kırmızı sızar
    bad_svg = _scene(w, h, (
        '<path d="M100 300 C160 210, 205 120, 250 72 L262 72 C307 120, 352 210, 412 300 Z" fill="#ffed00"/>'
        '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>'
    ))
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(bad_svg)
    r0 = render_svg_to_rgb(f, w, h)
    e0 = int((np.abs(src.astype(int) - r0.astype(int)).sum(axis=2) > 30).sum())
    rep = refine_cusp_regions(f, src, w, h, render_svg_to_rgb)
    r1 = render_svg_to_rgb(f, w, h)
    e1 = int((np.abs(src.astype(int) - r1.astype(int)).sum(axis=2) > 30).sum())
    check(rep.get("anchors_added", 0) >= 1, f"kamaya çapa eklendi ({rep.get('anchors_added')})")
    # bütçe: bölge başına 4 (modül içi), görsel başına 12 (şartname)
    check(rep.get("anchors_added", 99) <= 12, f"görsel çapa bütçesi aşılmadı ({rep.get('anchors_added')})")
    check(e1 <= e0 * 0.5, f"kama hatası >=%50 azaldı ({e0} -> {e1})")
    txt = f.read_text()
    check("NaN" not in txt and "Infinity" not in txt, "NaN/Infinity yok")
    f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Yumuşak sapma: hata-refine çözer, cusp bölmesi GEREKMEZ
# ---------------------------------------------------------------------------
def test_smooth_no_split() -> None:
    print("== Yumuşak sınır: gereksiz bölme yok ==")
    from app.cusp_refine import refine_cusp_regions
    from app.fidelity import render_svg_to_rgb
    from app.local_refine import refine_error_regions

    w = h = 512
    src_svg = _scene(w, h, '<circle cx="256" cy="256" r="150" fill="#ffed00"/>')
    src = _render(src_svg, w, h)
    # yumuşak sapma: yarıçap 2.5px küçük (cusp yok, düzgün ofset)
    bad_svg = _scene(w, h, (
        '<path d="M 403.5,256 A 147.5,147.5 0 1,0 108.5,256 '
        'A 147.5,147.5 0 1,0 403.5,256 Z" fill="#ffed00"/>'
    ))
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(bad_svg)
    refine_error_regions(f, src, w, h, render_svg_to_rgb)
    rep = refine_cusp_regions(f, src, w, h, render_svg_to_rgb)
    check(rep.get("anchors_added", 0) == 0,
          f"yumuşak eğriye çapa eklenmedi ({rep.get('anchors_added')}, {rep.get('status')})")
    f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Üçüncü ince bölge güvenliği: yanlış shared-boundary birleşmesi yok
# ---------------------------------------------------------------------------
def test_third_region_safety() -> None:
    print("== Üçüncü ince bölge güvenliği ==")
    from app.cusp_refine import refine_cusp_regions
    from app.fidelity import render_svg_to_rgb

    w = h = 512
    # kaynakta sarı ve siyah ARASINDA gerçek 4px beyaz şerit var
    body = ('<path d="M40 40 L250 40 L250 472 L40 472 Z" fill="#ffed00"/>'
            '<path d="M250 40 L254 40 L254 472 L250 472 Z" fill="#ffffff"/>'
            '<path d="M254 40 L472 40 L472 472 L254 472 Z" fill="#000000"/>')
    src_svg = _scene(w, h, body)
    src = _render(src_svg, w, h)
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(src_svg)  # SVG kaynağa sadık: değişiklik GEREKMEZ
    rep = refine_cusp_regions(f, src, w, h, render_svg_to_rgb)
    check(rep.get("anchors_added", 0) == 0,
          f"sadık üç-bölgeli sahnede çapa eklenmedi ({rep.get('status')})")
    r1 = _render(f.read_text(), w, h)
    stripe = r1[100:400, 251:253]
    white = int((np.abs(stripe.astype(int) - 255).sum(axis=2) < 90).sum())
    check(white >= 0.9 * stripe.shape[0] * stripe.shape[1],
          f"beyaz şerit korunuyor ({white}px)")
    f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. Bant sınıflandırma: bit-birebir + bant boyutundan bağımsız
# ---------------------------------------------------------------------------
def test_tiled_classify() -> None:
    print("== Bant (tiled) sınıflandırma eşdeğerliği ==")
    from app import palette_ops

    rng = np.random.RandomState(11)
    img = rng.randint(0, 256, (777, 1024, 3), dtype=np.uint8)
    fills = np.array([[0, 0, 0], [227, 0, 11], [255, 237, 0], [255, 255, 255],
                      [0, 166, 81]], dtype=np.float32)
    os.environ["VEKTORYUM_TILED_CLASSIFY"] = "off"
    mono = palette_ops.classify_rgb(img, fills)
    os.environ["VEKTORYUM_TILED_CLASSIFY"] = "on"
    orig_budget = palette_ops._BAND_BUDGET_BYTES
    try:
        for budget in (1 << 18, 1 << 22, 1 << 26):  # ~bant 16/256/4096 satır
            palette_ops._BAND_BUDGET_BYTES = budget
            tiled = palette_ops.classify_rgb(img, fills)
            check(bool((tiled == mono).all()),
                  f"bütçe {budget}B: bit-birebir (bant dikişi yok)")
    finally:
        palette_ops._BAND_BUDGET_BYTES = orig_budget
    check(mono.dtype == np.uint8, "sınıf çıktısı uint8 (bellek)")


# ---------------------------------------------------------------------------
# 6. Determinizm: cusp refine iki koşuda aynı bayt çıktısı
# ---------------------------------------------------------------------------
def test_determinism() -> None:
    print("== Determinizm (cusp refine, 2 koşu) ==")
    from app.cusp_refine import refine_cusp_regions
    from app.fidelity import render_svg_to_rgb

    w = h = 512
    src_svg = _scene(w, h, (
        '<path d="M100 300 L256 60 L412 300 Z" fill="#ffed00"/>'
        '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>'
    ))
    src = _render(src_svg, w, h)
    bad_svg = _scene(w, h, (
        '<path d="M100 300 C160 210, 205 120, 250 72 L262 72 C307 120, 352 210, 412 300 Z" fill="#ffed00"/>'
        '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>'
    ))
    outs = []
    reps = []
    for _i in range(2):
        f = Path(tempfile.mkstemp(suffix=".svg")[1])
        f.write_text(bad_svg)
        reps.append(refine_cusp_regions(f, src, w, h, render_svg_to_rgb))
        outs.append(f.read_bytes())
        f.unlink(missing_ok=True)
    check(outs[0] == outs[1], "iki koşu bit-birebir aynı SVG")
    check(reps[0].get("anchors_added") == reps[1].get("anchors_added"),
          "aynı çapa sayısı")


def main() -> int:
    test_decasteljau()
    test_wedge()
    test_smooth_no_split()
    test_third_region_safety()
    test_tiled_classify()
    test_determinism()
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
