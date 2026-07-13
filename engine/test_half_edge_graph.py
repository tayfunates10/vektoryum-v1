"""Shared-boundary half-edge (DCEL) graph birim regresyonları — FAZ HG-1/HG-2.

Bu testler SHADOW graph modelini doğrular (production SVG'ye bağlı değil):
twin invariant'ı, face cycle kapalılığı, next/previous tutarlılığı, canonical
curve ters paylaşımı, determinist ID, ayrık-aynı-renk yüz, exterior face,
Euler karakteristiği, junction paylaşımlı vertex ve validation-failure.

Çalıştırma::  .venv/bin/python test_half_edge_graph.py   (~1 sn)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _two_region() -> "object":
    from app.half_edge_graph import build_half_edge_graph
    lab = np.zeros((6, 8), np.uint8)
    lab[:, 4:] = 1
    return build_half_edge_graph(lab)


def test_half_edge_twin() -> None:
    print("== Twin invariant: her half-edge tam bir twin, aynı curve, ters yön ==")
    g = _two_region()
    check(len(g.half_edges) > 0, "half-edge üretildi")
    all_ok = True
    for h in g.half_edges.values():
        t = g.half_edges.get(h.twin_id or "")
        if t is None:
            all_ok = False
            break
        if t.curve_id != h.curve_id:
            all_ok = False
        if t.reversed == h.reversed:
            all_ok = False
        if t.origin_vertex_id != h.destination_vertex_id or \
                t.destination_vertex_id != h.origin_vertex_id:
            all_ok = False
    check(all_ok, "twin aynı curve + ters yön + origin/dest takas")
    check(g.stats()["twins"] == len(g.half_edges), "tüm half-edge'ler twin'li")


def test_face_cycle_closed() -> None:
    print("== Face cycle kapalı: next zinciri başa döner, tek yüz ==")
    g = _two_region()
    ok = True
    for c in g.cycles.values():
        chain = c.half_edge_ids
        # zincir tek yüze ait
        faces = {g.half_edges[h].face_id for h in chain}
        if len(faces) != 1:
            ok = False
        # next zinciri kapalı: son.next == ilk
        if g.half_edges[chain[-1]].next_id != chain[0]:
            ok = False
    check(ok, "her cycle tek yüz + next zinciri kapalı")
    # sınırlı her görünür yüzün tam bir outer cycle'ı olmalı
    vis = [f for f in g.faces.values() if f.visible and not f.is_exterior]
    check(all(f.outer_cycle_id is not None for f in vis),
          "her görünür yüzün outer cycle'ı var")


def test_next_previous_consistency() -> None:
    print("== next/previous tutarlılığı: next.previous == self ==")
    g = _two_region()
    ok = True
    for h in g.half_edges.values():
        if h.next_id is not None:
            nx = g.half_edges.get(h.next_id)
            if nx is None or nx.previous_id != h.half_edge_id:
                ok = False
        if h.previous_id is not None:
            pv = g.half_edges.get(h.previous_id)
            if pv is None or pv.next_id != h.half_edge_id:
                ok = False
    check(ok, "next/previous çift-yön tutarlı")
    check(g.valid, "graph validate geçti")


def test_canonical_curve_reverse() -> None:
    print("== Canonical curve: twin'ler TEK curve'ü ters yönde paylaşır ==")
    g = _two_region()
    # her curve tam iki half-edge'e (twin çifti) hizmet eder
    from collections import Counter
    cnt = Counter(h.curve_id for h in g.half_edges.values())
    check(all(v == 2 for v in cnt.values()),
          "her canonical curve tam 2 half-edge (paylaşımlı sınır)")
    # forward/reversed karışımı
    ok = True
    for cid in g.curves:
        hs = [h for h in g.half_edges.values() if h.curve_id == cid]
        if {h.reversed for h in hs} != {False, True}:
            ok = False
    check(ok, "her curve'ün bir forward + bir reversed half-edge'i var")


def test_deterministic_graph_id() -> None:
    print("== Determinist ID: aynı girdi → bit-aynı yapı ==")
    from app.half_edge_graph import build_half_edge_graph
    lab = np.zeros((10, 10), np.uint8)
    lab[2:8, 2:8] = 1
    lab[4:6, 4:6] = 2

    def sig(g):
        parts = []
        for hid, h in sorted(g.half_edges.items()):
            parts.append((hid, h.face_id, h.origin_vertex_id,
                          h.destination_vertex_id, h.twin_id, h.next_id,
                          h.reversed, h.curve_id))
        for cid, c in sorted(g.curves.items()):
            parts.append((cid, c.start_vertex_id, c.end_vertex_id,
                          tuple(c.polyline), c.adjacent_face_ids))
        return tuple(parts)

    sigs = [sig(build_half_edge_graph(lab)) for _ in range(3)]
    check(sigs[0] == sigs[1] == sigs[2], "3 koşu bit-aynı graph imzası")


def test_disconnected_same_color_face() -> None:
    print("== Ayrık aynı renk: iki ayrı yüz ==")
    from app.half_edge_graph import build_half_edge_graph
    lab = np.zeros((6, 12), np.uint8)
    lab[1:5, 1:3] = 1
    lab[1:5, 9:11] = 1  # ayrık ikinci sınıf-1 blok
    g = build_half_edge_graph(lab)
    ones = [f for f in g.faces.values() if f.color_id == 1]
    check(len(ones) == 2, f"aynı renkte 2 ayrı yüz ({len(ones)})")
    check(g.valid, "graph geçerli")


def test_exterior_face() -> None:
    print("== Exterior face: dış silhouette twin'li, tam 1 exterior yüz ==")
    g = _two_region()
    ext = [f for f in g.faces.values() if f.is_exterior]
    check(len(ext) == 1, f"tam 1 exterior yüz ({len(ext)})")
    check(not ext[0].visible, "exterior görünmez")
    # exterior'a komşu her curve is_exterior işaretli
    exc = [c for c in g.curves.values()
           if ext[0].face_id in c.adjacent_face_ids]
    check(exc and all(c.is_exterior for c in exc),
          "exterior'a komşu curve'ler is_exterior=True")


def test_graph_euler() -> None:
    print("== Euler: bağlı düzlemsel graph V - E + F = 2 ==")
    g = _two_region()  # bağlı (iç sınır dışa değer): tek bileşen
    V = len(g.vertices)
    E = len(g.curves)
    F = len(g.faces)  # exterior dahil
    check(V - E + F == 2, f"V-E+F=2 (V={V} E={E} F={F} → {V - E + F})")


def test_graph_euler_disconnected() -> None:
    print("== Euler (ayrık graph): V - E + F = 1 + C, F=yüz sayısı ==")
    from app.half_edge_graph import build_half_edge_graph, _connected_components
    # üç ayrık kapalı sınır (arka plan + iki ayrık kare) → C=3
    lab = np.zeros((6, 12), np.uint8)
    lab[1:5, 1:3] = 1
    lab[1:5, 9:11] = 1
    g = build_half_edge_graph(lab)
    V, E, F = len(g.vertices), len(g.curves), len(g.faces)
    C = _connected_components(g)
    check(V - E + F == 1 + C, f"V-E+F={V - E + F} == 1+C={1 + C}")
    check(g.valid, "ayrık graph geçerli (Euler validator geçti)")
    # corner-pinch bug regresyonu: 2×2 dama tahtası 4 ayrı yüz olmalı
    checker = build_half_edge_graph(np.array([[0, 1], [1, 0]], np.uint8))
    vis = [f for f in checker.faces.values() if f.visible and not f.is_exterior]
    check(len(vis) == 4, f"2×2 dama → 4 ayrı yüz (4-bağlantı) [{len(vis)}]")
    check(checker.valid, "dama tahtası geçerli (corner-pinch yok)")
    check(checker.stats()["outer_cycles"] == 4,
          "her yüzün tam 1 outer cycle'ı (pinch yok)")


def test_junction_shared_vertex() -> None:
    print("== Junction: üç yüz TEK paylaşımlı vertex'te buluşur ==")
    from app.half_edge_graph import build_half_edge_graph
    lab = np.zeros((6, 6), np.uint8)
    lab[:3, :3] = 0
    lab[:3, 3:] = 1
    lab[3:, :] = 2  # üç bölge bir noktada
    g = build_half_edge_graph(lab)
    js = [v for v in g.vertices.values() if v.junction_id]
    check(len(js) >= 1, f"en az 1 junction vertex ({len(js)})")
    # junction'da ≥3 farklı yüz incident
    ok = False
    for v in js:
        faces = {g.half_edges[h].face_id for h in v.incident_half_edge_ids}
        if len(faces) >= 3:
            ok = True
    check(ok, "bir junction vertex'inde ≥3 farklı yüz paylaşımlı")
    check(g.valid, "3-bölge graph geçerli")


def test_graph_validation_failure() -> None:
    print("== Validation-failure: bozuk graph reddedilir (fallback tetikler) ==")
    from app.half_edge_graph import build_half_edge_graph, validate_graph
    lab = np.zeros((8, 8), np.uint8)
    lab[2:6, 2:6] = 1

    g = build_half_edge_graph(lab)
    check(g.valid, "sağlam graph geçerli")

    g2 = build_half_edge_graph(lab)
    next(iter(g2.half_edges.values())).twin_id = "he_yok"
    check(not validate_graph(g2), "kayıp twin → validate False")

    g3 = build_half_edge_graph(lab)
    h = next(h for h in g3.half_edges.values() if h.twin_id)
    g3.half_edges[h.twin_id].reversed = h.reversed  # aynı yön
    check(not validate_graph(g3), "twin aynı yön → validate False")

    g4 = build_half_edge_graph(lab)
    v = next(iter(g4.vertices.values()))
    v.point = (float("nan"), 0.0)
    check(not validate_graph(g4), "NaN koordinat → validate False")


def main() -> int:
    test_half_edge_twin()
    test_face_cycle_closed()
    test_next_previous_consistency()
    test_canonical_curve_reverse()
    test_deterministic_graph_id()
    test_disconnected_same_color_face()
    test_exterior_face()
    test_graph_euler()
    test_graph_euler_disconnected()
    test_junction_shared_vertex()
    test_graph_validation_failure()
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
