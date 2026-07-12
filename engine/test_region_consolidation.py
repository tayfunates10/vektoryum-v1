"""HG-2.5 semantik region consolidation birim regresyonları (SHADOW).

Fringe/island merge + gerçek ince şerit / üçüncü renk / hole / disconnected
same-color koruması, topology-change reddi, determinizm, ölçek-normalize eşik,
ölçek-kararlılığı (consolidation ham face büyümesini anlamlı azaltır).

Çalıştırma::  .venv/bin/python test_region_consolidation.py   (~15 sn)
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


def _lin(c):
    c = np.array(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlin(l):
    l = np.clip(l, 0, 1)
    s = np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)
    return tuple(int(round(x * 255)) for x in s)


BLACK, RED, WHITE, YELLOW = (0, 0, 0), (227, 0, 11), (255, 255, 255), (255, 237, 0)
FRINGE = _unlin(0.5 * _lin(BLACK) + 0.5 * _lin(RED))


def _build(cm, palette):
    fills = np.array(palette, np.uint8)
    labels = cm.astype(np.uint8)
    return labels, fills[labels], fills


def test_antialias_fringe_merge() -> None:
    print("== Anti-alias fringe merge ==")
    from app.region_consolidation import consolidate_regions
    cm = np.zeros((20, 30), np.uint8)
    cm[:, :14] = 0
    cm[:, 14:16] = 2
    cm[:, 16:] = 1
    lab, rgb, fills = _build(cm, [BLACK, RED, FRINGE])
    r = consolidate_regions(lab, rgb, fills)
    check(r.summary()["fringe_merges"] >= 1, "fringe merge oldu")
    check(2 not in set(r.consolidated_labels.ravel().tolist()), "fringe rengi kalmadı")


def test_quantization_island_merge() -> None:
    print("== Quantization island merge (renk komşuya çok yakın) ==")
    from app.region_consolidation import consolidate_regions
    NEAR_BLACK = (6, 6, 6)
    cm = np.zeros((20, 20), np.uint8)
    cm[9:11, 9:11] = 1   # küçük ada, rengi neredeyse siyah
    lab, rgb, fills = _build(cm, [BLACK, NEAR_BLACK])
    r = consolidate_regions(lab, rgb, fills)
    check(r.summary()["island_merges"] >= 1, "island merge oldu")


def test_preserve_real_thin_strip() -> None:
    print("== Gerçek ince şerit (bağımsız renk) korunur ==")
    from app.region_consolidation import consolidate_regions
    cm = np.zeros((20, 30), np.uint8)
    cm[:, :14] = 0
    cm[:, 14:16] = 2   # ince ama bağımsız SARI
    cm[:, 16:] = 1
    lab, rgb, fills = _build(cm, [BLACK, RED, YELLOW])
    r = consolidate_regions(lab, rgb, fills)
    check(2 in set(r.consolidated_labels.ravel().tolist()), "bağımsız ince şerit korundu")


def test_preserve_third_region() -> None:
    print("== Üçüncü renk region iki komşu arasında erimez ==")
    from app.region_consolidation import consolidate_regions
    cm = np.zeros((30, 45), np.uint8)
    cm[:, :20] = 0
    cm[:, 20:25] = 2   # gerçek üçüncü renk (sarı) şerit
    cm[:, 25:] = 1
    lab, rgb, fills = _build(cm, [BLACK, RED, YELLOW])
    r = consolidate_regions(lab, rgb, fills)
    from app.region_consolidation import _region_map
    _, info = _region_map(r.consolidated_labels)
    check(any(v[0] == 2 for v in info.values()), "üçüncü renk korundu")
    # siyah ve kırmızı hâlâ DOĞRUDAN komşu değil (arada sarı)
    ro, info2 = _region_map(r.consolidated_labels)
    from app.region_consolidation import _adjacency
    _, neigh, _, _ = _adjacency(ro)
    col = {k: v[0] for k, v in info2.items()}
    br = any(col[a] == 0 and col[b] == 1 for a in neigh for b in neigh[a])
    check(not br, "siyah-kırmızı doğrudan komşu değil (üçüncü renk araya girdi)")


def test_preserve_hole_boundary() -> None:
    print("== Hole/counter sınırı korunur ==")
    from app.region_consolidation import consolidate_regions
    from app.half_edge_graph import build_half_edge_graph
    cm = np.zeros((30, 30), np.uint8)
    cm[8:22, 8:22] = 1
    cm[13:17, 13:17] = 0   # delik
    lab, rgb, fills = _build(cm, [BLACK, YELLOW])
    r = consolidate_regions(lab, rgb, fills)
    g = build_half_edge_graph(r.consolidated_labels)
    check(g.stats()["inner_cycles"] >= 1, "delik (inner cycle) korundu")


def test_preserve_disconnected_same_color() -> None:
    print("== Aynı renkte disconnected gerçek yüzler ayrık kalır ==")
    from app.region_consolidation import consolidate_regions, _region_map
    cm = np.zeros((20, 40), np.uint8)
    cm[5:15, 3:8] = 1
    cm[5:15, 32:37] = 1
    lab, rgb, fills = _build(cm, [BLACK, YELLOW])
    r = consolidate_regions(lab, rgb, fills)
    _, info = _region_map(r.consolidated_labels)
    check(sum(1 for v in info.values() if v[0] == 1) == 2, "iki ayrı sarı korundu")


def test_reject_topology_changing_merge() -> None:
    print("== Hedef renkte >1 bileşen: birleşme reddedilir ==")
    from app.region_consolidation import consolidate_regions
    # ince fringe iki AYRI siyah bloğa değiyor; merge iki siyahı birleştirmemeli
    cm = np.zeros((10, 20), np.uint8)
    cm[:, :] = 1          # zemin kırmızı
    cm[:4, 8:12] = 0      # siyah blok A
    cm[6:, 8:12] = 0      # siyah blok B (ayrık)
    cm[4:6, 8:12] = 2     # arada fringe (siyah-kırmızı karışımı) iki siyaha komşu
    lab, rgb, fills = _build(cm, [RED, BLACK, FRINGE])
    r = consolidate_regions(lab, rgb, fills)
    # eğer fringe siyaha merge edilseydi iki siyah birleşirdi → reddedilmeli
    rej = [m for m in r.merge_log if not m.accepted and "bileşen" in (m.rejection_reason or "")]
    from app.region_consolidation import _region_map
    _, info = _region_map(r.consolidated_labels)
    blacks = sum(1 for v in info.values() if v[0] == 1)
    check(blacks == 2, f"iki siyah blok ayrık kaldı ({blacks})")


def test_deterministic_region_merge() -> None:
    print("== Determinizm: aynı girdi aynı hash ==")
    from app.region_consolidation import consolidate_regions
    cm = np.tile(np.array([[0, 2, 1]], np.uint8), (12, 8))
    lab, rgb, fills = _build(cm, [BLACK, RED, FRINGE])
    hs = [consolidate_regions(lab, rgb, fills).deterministic_hash for _ in range(3)]
    check(hs[0] == hs[1] == hs[2], "3 koşu aynı deterministik hash")


def test_scale_normalized_merge() -> None:
    print("== Ölçek-normalize: ince (~2px) fringe farklı görsel boylarında merge ==")
    # AA fringe ABSOLUTE olarak incedir (~1-2px); görsel boyu değişse de eşik
    # tabanı (≥2.5px) korunur. Fringe genişliğini değil GÖRSEL BOYUNU değiştir.
    from app.region_consolidation import consolidate_regions
    for size in (30, 120):
        cm = np.zeros((20, size), np.uint8)
        mid = size // 2
        cm[:, :mid - 1] = 0
        cm[:, mid - 1:mid + 1] = 2   # 2px fringe (ölçekten bağımsız ince)
        cm[:, mid + 1:] = 1
        lab, rgb, fills = _build(cm, [BLACK, RED, FRINGE])
        r = consolidate_regions(lab, rgb, fills)
        check(r.summary()["fringe_merges"] >= 1, f"görsel genişlik {size}px: ince fringe merge")


def test_scale_stability_reduces_growth() -> None:
    print("== Ölçek kararlılığı: consolidation ham face büyümesini azaltır ==")
    from app.region_consolidation import consolidate_regions
    from app.half_edge_graph import build_half_edge_graph
    from app.graph_source import canonical_segmentation

    def synth(n):
        img = np.full((n, n, 3), WHITE, np.uint8)
        cv2.circle(img, (n // 2, n // 2), int(n * 0.38), RED, -1, cv2.LINE_AA)
        cv2.rectangle(img, (int(n * .30), int(n * .30)), (int(n * .55), int(n * .55)), BLACK, -1, cv2.LINE_AA)
        return img

    raw_counts, con_counts, holes = [], [], []
    for n in (256, 512, 1024):
        img = synth(n)
        labels, fills = canonical_segmentation(img, k=8)
        g0 = build_half_edge_graph(labels)
        cons = consolidate_regions(labels, img, fills)
        g1 = build_half_edge_graph(cons.consolidated_labels)
        raw_counts.append(g0.stats()["visible_faces"])
        con_counts.append(g1.stats()["visible_faces"])
        holes.append(g1.stats()["inner_cycles"])
        check(g1.stats()["valid"], f"n={n} consolidated graph geçerli")
    raw_growth = raw_counts[-1] / max(1, raw_counts[0])
    con_growth = con_counts[-1] / max(1, con_counts[0])
    print(f"    raw {raw_counts} (×{raw_growth:.1f})  consolidated {con_counts} (×{con_growth:.1f})")
    check(con_counts[-1] < raw_counts[-1] * 0.85, "consolidated face sayısı ham'dan belirgin düşük")
    check(con_growth <= raw_growth + 0.05, "consolidation ölçek büyümesini artırmadı")
    check(all(h >= 1 for h in holes), "delik yapısı her ölçekte korundu")


def main() -> int:
    test_antialias_fringe_merge()
    test_quantization_island_merge()
    test_preserve_real_thin_strip()
    test_preserve_third_region()
    test_preserve_hole_boundary()
    test_preserve_disconnected_same_color()
    test_reject_topology_changing_merge()
    test_deterministic_region_merge()
    test_scale_normalized_merge()
    test_scale_stability_reduces_growth()
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
