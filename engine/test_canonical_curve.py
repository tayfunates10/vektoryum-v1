"""HG-3 alt-piksel canonical curve fitting birim regresyonları (SHADOW).

Straight/cubic/circle fit, junction endpoint kilidi, twin reverse eşitliği,
determinizm, alt-piksel örnekleme, düşük-gradyan fallback.

Çalıştırma::  .venv/bin/python test_canonical_curve.py   (~10 sn)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _fit(lab, fills, rgb=None):
    from app.half_edge_graph import build_half_edge_graph
    from app.canonical_curve import fit_canonical_curves
    from app.graph_source import fills_to_hex
    if rgb is None:
        rgb = fills[lab]
    g = build_half_edge_graph(lab, fills_hex=fills_to_hex(fills))
    st = fit_canonical_curves(g, rgb, fills)
    return g, st


def test_straight_boundary_fit() -> None:
    print("== Düz sınır → line primitive (tek segment) ==")
    lab = np.zeros((20, 30), np.uint8)
    lab[:, 15:] = 1
    fills = np.array([[0, 0, 0], [227, 0, 11]], np.uint8)
    g, st = _fit(lab, fills)
    internal = [c for c in g.curves.values() if not c.is_exterior]
    check(all(c.primitive_kind == "line" for c in internal), "iç düz sınır line")
    check(all(c.command_count == 1 for c in internal), "düz sınır tek komut")
    check(st["fit_error_max"] < 1.5, f"düz sınır hatası küçük ({st['fit_error_max']:.2f})")


def test_rectangle_sharp_corners() -> None:
    print("== Dikdörtgen exterior silhouette: keskin köşe (çok line) ==")
    lab = np.zeros((20, 30), np.uint8)
    lab[:, 15:] = 1
    fills = np.array([[0, 0, 0], [227, 0, 11]], np.uint8)
    g, st = _fit(lab, fills)
    check(st["fit_error_max"] < 1.5, f"dikdörtgen tüm sınır hatası küçük ({st['fit_error_max']:.2f})")
    ext = [c for c in g.curves.values() if c.is_exterior]
    check(all(c.command_count <= 4 for c in ext), "exterior silhouette ≤4 komut (keskin köşe)")


def test_circle_low_command_cubic() -> None:
    print("== Daire → düşük komutlu cubic, düşük hata ==")
    n = 240
    img = np.full((n, n, 3), (255, 255, 255), np.uint8)
    cv2.circle(img, (n // 2, n // 2), 70, (227, 0, 11), -1, cv2.LINE_8)
    fills = np.array([[255, 255, 255], [227, 0, 11]], np.uint8)
    from app.palette_ops import classify_rgb
    lab = classify_rgb(img, fills.astype(np.float32)).astype(np.uint8)
    g, st = _fit(lab, fills, img)
    circ = max((c for c in g.curves.values() if not c.is_exterior),
               key=lambda c: len(c.polyline))
    check(circ.primitive_kind == "cubic", "daire sınırı cubic")
    check(circ.command_count <= 24, f"daire ≤24 komut ({circ.command_count})")
    check(circ.fit_error_max < 2.0, f"daire hatası <2px ({circ.fit_error_max:.2f})")
    check(st["fit_error_max"] < 2.5, f"tüm curve maks hata makul ({st['fit_error_max']:.2f})")


def test_junction_endpoint_lock() -> None:
    print("== Junction endpoint kilidi: incident curve uçları TAM shared vertex ==")
    lab = np.zeros((30, 30), np.uint8)
    lab[:15, :15] = 0
    lab[:15, 15:] = 1
    lab[15:, :] = 2
    fills = np.array([[0, 0, 0], [227, 0, 11], [255, 237, 0]], np.uint8)
    g, _ = _fit(lab, fills)
    ok = True
    for c in g.curves.values():
        if not c.fitted_segments:
            continue
        vs = g.vertices[c.start_vertex_id].point
        ve = g.vertices[c.end_vertex_id].point
        p0 = c.fitted_segments[0].p0
        p1 = c.fitted_segments[-1].p1
        # kapalı loop hariç: açık curve uçları vertex'e tam otursun
        if c.start_vertex_id != c.end_vertex_id:
            if abs(p0[0] - vs[0]) > 1e-6 or abs(p0[1] - vs[1]) > 1e-6:
                ok = False
            if abs(p1[0] - ve[0]) > 1e-6 or abs(p1[1] - ve[1]) > 1e-6:
                ok = False
    check(ok, "açık curve uçları shared vertex'e tam kilitli")


def test_canonical_twin_reverse() -> None:
    print("== Canonical twin reverse: aynı curve, ters segment; yeniden fit yok ==")
    from app.canonical_curve import BezierSegment
    lab = np.zeros((20, 30), np.uint8)
    lab[:, 15:] = 1
    fills = np.array([[0, 0, 0], [227, 0, 11]], np.uint8)
    g, _ = _fit(lab, fills)
    # twin half-edge çifti AYNI curve nesnesine (dolayısıyla aynı segment listesi) bakar
    ok = True
    for h in g.half_edges.values():
        t = g.half_edges[h.twin_id]
        if h.curve_id != t.curve_id:
            ok = False
    check(ok, "twin half-edge'ler aynı curve_id (tek fit)")
    seg = BezierSegment((0., 0.), (1., 2.), (3., 4.), (5., 6.))
    r = seg.reversed()
    check(r.p0 == seg.p1 and r.p1 == seg.p0 and r.c1 == seg.c2 and r.c2 == seg.c1,
          "segment reverse uçları+kontrolleri takas eder")


def test_fitting_determinism() -> None:
    print("== Fitting determinizmi: 3 koşu aynı segmentler ==")
    lab = np.zeros((40, 40), np.uint8)
    cv2.circle(lab, (20, 20), 12, 1, -1)
    fills = np.array([[0, 0, 0], [227, 0, 11]], np.uint8)

    def sig():
        g, _ = _fit(lab, fills)
        return tuple((cid, tuple((round(s.p0[0], 4), round(s.p0[1], 4),
                                  round(s.p1[0], 4), round(s.p1[1], 4))
                                 for s in g.curves[cid].fitted_segments))
                     for cid in sorted(g.curves))
    a, b, c = sig(), sig(), sig()
    check(a == b == c, "3 koşu bit-aynı fitted segmentler")


def test_low_gradient_fallback() -> None:
    print("== Düşük gradyan / belirsiz → fallback (ham polyline korunur) ==")
    # iki renk neredeyse aynı → renk ayrımı zayıf → alt-piksel güveni düşük
    lab = np.zeros((20, 30), np.uint8)
    lab[:, 15:] = 1
    fills = np.array([[100, 100, 100], [101, 101, 101]], np.uint8)  # neredeyse aynı
    g, st = _fit(lab, fills)
    internal = [c for c in g.curves.values() if not c.is_exterior]
    # fit yine üretilir (fallback ham polyline'a düşse de segment döner)
    check(all(len(c.fitted_segments) >= 1 for c in internal), "fallback'te de segment var")
    check(st["curves"] == len(g.curves), "tüm curve işlendi")


def test_subpixel_offset_direction() -> None:
    print("== Alt-piksel offset: keskin renk geçişini yakalar ==")
    from app.canonical_curve import _subpixel_offset
    # sol yarı siyah, sağ yarı beyaz; sınır x=10'da
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, 10:] = 255
    cL = np.array([0, 0, 0], np.float32)
    cR = np.array([255, 255, 255], np.float32)
    # crack noktası (10,10), normal +x
    off, conf = _subpixel_offset(img, 10.0, 10.0, 1.0, 0.0, cL, cR)
    check(conf > 0.5, f"net renk ayrımında güven yüksek ({conf:.2f})")
    check(abs(off) <= 1.5, f"offset makul aralıkta ({off:.2f})")


def main() -> int:
    test_straight_boundary_fit()
    test_rectangle_sharp_corners()
    test_circle_low_command_cubic()
    test_junction_endpoint_lock()
    test_canonical_twin_reverse()
    test_fitting_determinism()
    test_low_gradient_fallback()
    test_subpixel_offset_direction()
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
