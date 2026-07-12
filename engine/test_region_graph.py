"""Region-adjacency graph + mantıksal çapa muhasebesi birim regresyonları.

Çalıştırma::  .venv/bin/python test_region_graph.py   (~20 sn)
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


def test_two_region_edge() -> None:
    print("== İki komşu renk: tek kenar ==")
    from app.region_graph import build_region_graph

    lab = np.zeros((100, 100), np.uint8)
    lab[:, 50:] = 1
    g = build_region_graph(lab, min_area=50)
    check(len(g.nodes) == 2, f"2 düğüm ({len(g.nodes)})")
    check(len(g.edges) == 1, f"1 kenar ({len(g.edges)})")
    check(g.edges[0].boundary_length >= 90, "sınır uzunluğu makul")
    check(len(g.junctions) == 0, "junction yok (2 bölge)")


def test_disconnected_same_color() -> None:
    print("== Aynı renkte iki ayrık bölge: iki düğüm ==")
    from app.region_graph import build_region_graph

    lab = np.zeros((100, 200), np.uint8)
    lab[20:80, 10:60] = 1   # bölge A (sınıf 1)
    lab[20:80, 140:190] = 1  # bölge B (sınıf 1, ayrık)
    g = build_region_graph(lab, min_area=50)
    ones = [n for n in g.nodes if n.color_id == 1]
    check(len(ones) == 2, f"aynı renkte 2 ayrı düğüm ({len(ones)})")


def test_three_region_junction() -> None:
    print("== Üç renk junction ==")
    from app.region_graph import build_region_graph

    lab = np.zeros((120, 120), np.uint8)
    lab[:, 60:] = 1
    lab[60:, :] = 2  # alt yarı sınıf 2 → üç bölge bir noktada buluşur
    g = build_region_graph(lab, min_area=50)
    check(len(g.nodes) == 3, f"3 düğüm ({len(g.nodes)})")
    check(len(g.junctions) >= 1, f"en az 1 junction ({len(g.junctions)})")
    j = g.junctions[0]
    check(len(j.incident_regions) >= 3, "junction 3+ bölgeye komşu")


def test_third_region_safety() -> None:
    print("== İnce üçüncü bölge güvenliği ==")
    from app.region_graph import build_region_graph

    lab = np.zeros((120, 120), np.uint8)   # sol: sınıf 0
    lab[:, 62:] = 1                          # sağ: sınıf 1
    lab[:, 58:62] = 2                        # arada ince sınıf 2 şeridi
    g = build_region_graph(lab, min_area=50)
    # sınıf 0 ve sınıf 1 DOĞRUDAN komşu olmamalı (arada 2 var)
    e01 = [e for e in g.edges
           if {g.nodes[int(e.region_a[1:])].color_id,
               g.nodes[int(e.region_b[1:])].color_id} == {0, 1}]
    check(not e01, "sınıf 0-1 doğrudan kenarı yok (üçüncü bölge araya girdi)")


def test_graph_determinism() -> None:
    print("== Graph determinizmi (3 koşu aynı) ==")
    from app.region_graph import build_region_graph

    rng = np.random.RandomState(5)
    lab = (rng.rand(80, 80) < 0.5).astype(np.uint8)
    lab[30:50, 30:50] = 2
    sigs = []
    for _ in range(3):
        g = build_region_graph(lab, min_area=30)
        sigs.append(tuple((n.region_id, n.color_id, n.area) for n in g.nodes))
    check(sigs[0] == sigs[1] == sigs[2], "3 koşu aynı düğüm imzası")


def test_logical_anchor_accounting() -> None:
    print("== Mantıksal çapa muhasebesi (eşleşmiş = 1 mantıksal, 2 fiziksel) ==")
    from app.cusp_refine import refine_cusp_regions
    from app.fidelity import render_svg_to_rgb

    w = h = 512

    def scene(body: str) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                f'viewBox="0 0 {w} {h}"><path d="M0 0 L{w} 0 L{w} {h} L0 {h} Z" '
                f'fill="#e3000b"/>{body}</svg>')

    # kaynak: keskin sarı kama, siyah blok — ortak sınır cusp'ı iki path'e
    src_svg = scene('<path d="M100 300 L256 60 L412 300 Z" fill="#ffed00"/>'
                    '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>')
    bad_svg = scene('<path d="M100 300 C160 210, 205 120, 250 72 L262 72 '
                    'C307 120, 352 210, 412 300 Z" fill="#ffed00"/>'
                    '<path d="M0 300 L512 300 L512 512 L0 512 Z" fill="#000000"/>')
    f0 = Path(tempfile.mkstemp(suffix=".svg")[1])
    f0.write_text(src_svg)
    src = render_svg_to_rgb(f0, w, h)
    f = Path(tempfile.mkstemp(suffix=".svg")[1])
    f.write_text(bad_svg)
    rep = refine_cusp_regions(f, src, w, h, render_svg_to_rgb)
    la = rep.get("logical_anchors", 0)
    pa = rep.get("physical_anchors", 0)
    check(rep.get("anchors_added", 0) >= 1, f"mantıksal çapa eklendi ({rep.get('anchors_added')})")
    check(rep.get("anchors_added") == la, "anchors_added == mantıksal çapa sayısı")
    check(pa >= la, f"fiziksel ({pa}) >= mantıksal ({la}) [eşleşmiş çift 2 fiziksel]")
    check(rep.get("anchors_added", 99) <= 12, "görsel mantıksal bütçe ≤12")
    f0.unlink(missing_ok=True)
    f.unlink(missing_ok=True)


def test_active_boundary_two_sided() -> None:
    print("== Aktif sınır eşleme: iki komşu (kabul) ==")
    from app.region_graph import build_region_graph, match_active_boundary

    lab = np.zeros((120, 120), np.uint8)
    lab[:, 60:] = 1
    g = build_region_graph(lab, min_area=50)
    # sınır çevresindeki bir hata blob kutusu
    m = match_active_boundary(g, (55, 40, 12, 40))
    check(m.accepted, f"iki komşu duvar kabul ({m.rejection_reason})")
    check(m.graph_edge_id is not None, "graph kenarı bulundu")
    check({m.color_a, m.color_b} == {0, 1}, "iki farklı renk tarafı")
    check(not m.third_region_risk, "üçüncü renk riski yok")


def test_active_boundary_third_region_reject() -> None:
    print("== Aktif sınır eşleme: üçüncü renk reddi ==")
    from app.region_graph import build_region_graph, match_active_boundary

    lab = np.zeros((120, 120), np.uint8)
    lab[:, 62:] = 1
    lab[:, 58:62] = 2  # ince üçüncü şerit
    g = build_region_graph(lab, min_area=50)
    m = match_active_boundary(g, (50, 40, 20, 40))
    check(not m.accepted, "üçüncü renk varken senkron refit reddedildi")
    check(m.rejection_reason is not None, "ret gerekçesi verildi")


def test_active_boundary_junction_reject() -> None:
    print("== Aktif sınır eşleme: junction reddi ==")
    from app.region_graph import build_region_graph, match_active_boundary

    lab = np.zeros((120, 120), np.uint8)
    lab[:, 60:] = 1
    lab[60:, :] = 2  # üç bölge tek noktada (junction)
    g = build_region_graph(lab, min_area=50)
    m = match_active_boundary(g, (50, 50, 20, 20))  # junction çevresi
    check(not m.accepted, f"junction'da senkron refit reddedildi ({m.rejection_reason})")


def main() -> int:
    test_two_region_edge()
    test_disconnected_same_color()
    test_three_region_junction()
    test_third_region_safety()
    test_graph_determinism()
    test_logical_anchor_accounting()
    test_active_boundary_two_sided()
    test_active_boundary_third_region_reject()
    test_active_boundary_junction_reject()
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
