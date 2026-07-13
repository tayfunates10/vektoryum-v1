"""HG-4 shadow graph cut-out serializer birim regresyonları (SHADOW).

Outer/inner evenodd cycle, background rect (eraser YOK), disconnected face,
z-order, source-coordinate + transform yokluğu, komut bütçesi, determinizm,
render kalite (palette agreement, hole doğruluğu).

Çalıştırma::  .venv/bin/python test_graph_serializer.py   (~15 sn)
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


def _shadow(img, fills, **kw):
    from app.shadow_pipeline import build_shadow_graph
    return build_shadow_graph(img, fills_rgb=np.array(fills, np.uint8), **kw)


def _donut(n=120):
    img = np.full((n, n, 3), (255, 255, 255), np.uint8)
    cv2.circle(img, (n // 2, n // 2), 40, (255, 237, 0), -1, cv2.LINE_8)
    cv2.circle(img, (n // 2, n // 2), 16, (0, 0, 0), -1, cv2.LINE_8)
    return img, [(255, 255, 255), (255, 237, 0), (0, 0, 0)]


def test_outer_cycle_svg() -> None:
    print("== Outer cycle → geçerli path (M...Z) ==")
    img = np.full((40, 40, 3), (255, 255, 255), np.uint8)
    img[10:30, 10:30] = (227, 0, 11)
    res = _shadow(img, [(255, 255, 255), (227, 0, 11)])
    check("<path" in res.svg and " d=\"M" in res.svg, "path + M başlangıcı var")
    check(res.svg.count("Z") >= 1, "en az bir kapalı Z")
    check(res.svg_metrics["visible_paths"] >= 1, "en az bir görünür path")


def test_inner_cycle_evenodd() -> None:
    print("== Inner cycle → evenodd hole ==")
    img, fills = _donut()
    res = _shadow(img, fills)
    check('fill-rule="evenodd"' in res.svg, "evenodd kullanıldı")
    check(res.graph.stats()["inner_cycles"] >= 1, "inner cycle (hole) var")


def test_background_independent_hole() -> None:
    print("== Hole gerçek evenodd (render'da doğru renk görünür) ==")
    from app.fidelity import render_svg_to_rgb
    from app.palette_ops import classify_rgb
    img, fills = _donut()
    res = _shadow(img, fills)
    p = ENGINE_DIR / "_shadow_tmp_donut.svg"
    p.write_text(res.svg)
    rnd = render_svg_to_rgb(p, img.shape[1], img.shape[0])
    p.unlink(missing_ok=True)
    check(rnd is not None, "shadow SVG render edilebilir")
    if rnd is not None:
        n = img.shape[0]
        black = tuple(int(v) for v in rnd[n // 2, n // 2])
        yellow = tuple(int(v) for v in rnd[n // 2, n // 2 - 30])
        check(black == (0, 0, 0), f"merkez siyah (hole içi) {black}")
        check(yellow[0] > 200 and yellow[2] < 60, f"halka sarı {yellow}")
        fa = np.array(fills, np.float32)
        agree = float((classify_rgb(img, fa) == classify_rgb(rnd, fa)).mean())
        check(agree > 0.97, f"palette agreement yüksek ({agree:.4f})")


def test_no_eraser_overlay() -> None:
    print("== Background renkli eraser/overlay path YOK ==")
    img, fills = _donut()
    res = _shadow(img, fills)
    # background sadece <rect>; delik ayrı eraser path'i olarak zemin rengiyle çizilmez
    bg_hex = "#ffffff"
    # zemin rengiyle bir <path> (rect değil) olmamalı
    import re
    paths = re.findall(r'<path[^>]*fill="([^"]+)"', res.svg)
    check(bg_hex not in [p.lower() for p in paths], "zemin rengiyle path (eraser) yok")
    check(res.svg_metrics["has_background_rect"], "background <rect> olarak")


def test_disconnected_face_serialization() -> None:
    print("== Disconnected aynı renk: iki ayrı path ==")
    img = np.full((40, 80, 3), (255, 255, 255), np.uint8)
    img[10:30, 8:20] = (227, 0, 11)
    img[10:30, 60:72] = (227, 0, 11)
    res = _shadow(img, [(255, 255, 255), (227, 0, 11)], consolidate=False)
    # iki ayrı kırmızı face → iki path
    import re
    red_paths = re.findall(r'<path[^>]*fill="#e3000b"', res.svg)
    check(len(red_paths) == 2, f"iki ayrı kırmızı path ({len(red_paths)})")


def test_face_z_order() -> None:
    print("== Z-order determinist (alan azalan) ==")
    img, fills = _donut()
    r1 = _shadow(img, fills)
    r2 = _shadow(img, fills)
    # aynı sırada aynı face id dizisi
    import re
    ids1 = re.findall(r'data-face-id="([^"]+)"', r1.svg)
    ids2 = re.findall(r'data-face-id="([^"]+)"', r2.svg)
    check(ids1 == ids2, "face sırası iki koşuda aynı (determinist z-order)")


def test_source_coordinate_and_transform_absence() -> None:
    print("== Source coordinate contract + transform yokluğu ==")
    img, fills = _donut()
    res = _shadow(img, fills)
    check("transform=" not in res.svg, "hiç transform yok")
    check('viewBox="0 0 120 120"' in res.svg, "viewBox source boyutu")
    check("scale(" not in res.svg and "translate(" not in res.svg, "scale/translate yok")


def test_command_budget() -> None:
    print("== Komut bütçesi: ham crack serialization'dan çok düşük ==")
    img, fills = _donut()
    res = _shadow(img, fills)
    # ham crack noktası sayısı (her curve polyline uzunluğu toplamı) >> fitted komut
    raw_pts = sum(len(c.polyline) for c in res.graph.curves.values())
    check(res.svg_metrics["total_commands"] < raw_pts * 0.5,
          f"fitted komut ({res.svg_metrics['total_commands']}) << ham nokta ({raw_pts})")


def test_deterministic_shadow_svg() -> None:
    print("== Deterministik shadow SVG (byte-aynı) ==")
    img, fills = _donut()
    a = _shadow(img, fills).svg
    b = _shadow(img, fills).svg
    check(a == b, "iki koşu byte-aynı shadow SVG")


def main() -> int:
    test_outer_cycle_svg()
    test_inner_cycle_evenodd()
    test_background_independent_hole()
    test_no_eraser_overlay()
    test_disconnected_face_serialization()
    test_face_z_order()
    test_source_coordinate_and_transform_absence()
    test_command_budget()
    test_deterministic_shadow_svg()
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
