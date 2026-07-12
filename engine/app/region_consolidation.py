"""HG-2.5 — Semantik region consolidation (SHADOW).

Ham palet/k-means etiketleri anti-aliasing ve quantization gürültüsünden çok
sayıda mikro-region üretir; bunlar matematiksel olarak geçerli ama semantik
olarak yanlış bir planar subdivision verir. Bu modül SADECE yüksek güvenli
mikro-region'ları (anti-alias fringe + quantization island) semantik ana yüze
bağlar; gerçek ince şeritleri, üçüncü renk region'larını, hole/counter'ları ve
anlamlı disconnected component'leri KORUR.

Kurallar (şartname):
- Kör morphology/blur YOK, alan-eşiğiyle toptan silme YOK.
- Aynı renkte disconnected gerçek yüzleri birleştirme YOK.
- Üçüncü ince region'ı iki komşu arasında eritme YOK.
- Source raster ve production label map YERİNDE mutate EDİLMEZ (türetilmiş kopya).
- Determinist: union grubu + compact yeniden numaralama sabit sıradadır.

Ayrımın matematiksel çekirdeği: anti-alias fringe rengi, iki komşusunun LİNEER
RGB karışımıdır (blend residual küçük); gerçek üçüncü renk bağımsız palet
merkezidir (blend residual büyük → korunur).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Veri modeli
# ---------------------------------------------------------------------------
@dataclass
class RegionMergeCandidate:
    source_region_id: int
    target_region_id: int
    source_color_id: int
    target_color_id: int
    area: int
    width_estimate: float
    elongation: float
    shared_boundary_length: float
    color_distance: float
    blend_residual: float
    category: str                 # fringe | island | protected:<reason>
    topology_safe: bool
    semantic_score: float
    accepted: bool
    rejection_reason: str | None = None


@dataclass
class ConsolidatedRegionMap:
    consolidated_labels: np.ndarray
    original_to_consolidated: dict[int, int]       # region_id -> anchor region_id
    merged_region_groups: dict[int, list[int]]     # anchor -> [merged region ids]
    preserved_micro_regions: list[int]
    merge_log: list[RegionMergeCandidate]
    stats_before: dict[str, Any]
    stats_after: dict[str, Any]
    deterministic_hash: str

    def summary(self) -> dict[str, Any]:
        acc = [m for m in self.merge_log if m.accepted]
        return {
            "regions_before": self.stats_before["regions"],
            "regions_after": self.stats_after["regions"],
            "merges": len(acc),
            "fringe_merges": sum(1 for m in acc if m.category == "fringe"),
            "island_merges": sum(1 for m in acc if m.category == "island"),
            "preserved_micro": len(self.preserved_micro_regions),
            "rejected": sum(1 for m in self.merge_log if not m.accepted),
            "hash": self.deterministic_hash,
        }


# ---------------------------------------------------------------------------
# Bölge haritası + komşuluk (vektörize, bellek dostu)
# ---------------------------------------------------------------------------
def _region_map(labels: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, tuple]]]:
    """connectivity=4 bağlı bileşenler. Döner: (region_of int32, info).

    info[rid] = (color_id, area, bbox=(x,y,w,h)). half_edge_graph ile aynı
    4-bağlantı sözleşmesi (corner-pinch'siz planar subdivision).
    """
    h, w = labels.shape
    region_of = np.full((h, w), -1, np.int32)
    info: dict[int, tuple[int, int, tuple]] = {}
    nxt = 0
    for cid in range(int(labels.max()) + 1):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            region_of[lab == i] = nxt
            x, y, bw, bh, area = (int(v) for v in stats[i])
            info[nxt] = (cid, area, (x, y, bw, bh))
            nxt += 1
    return region_of, info


def _adjacency(region_of: np.ndarray):
    """Vektörize 4-komşu sınır muhasebesi.

    Döner: shared{(a,b):len}, neighbors{r:set}, bcolor{r:{color:len}},
    btotal{r:len}, touches_canvas{r}.
    """
    from collections import defaultdict

    shared: dict[tuple[int, int], int] = defaultdict(int)

    def acc(a: np.ndarray, b: np.ndarray) -> None:
        m = a != b
        if not m.any():
            return
        aa = a[m].astype(np.int64)
        bb = b[m].astype(np.int64)
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        key = lo * (region_of.max().astype(np.int64) + 2) + hi
        uk, cnt = np.unique(key, return_counts=True)
        base = region_of.max().astype(np.int64) + 2
        for k, c in zip(uk.tolist(), cnt.tolist()):
            shared[(int(k // base), int(k % base))] += int(c)

    acc(region_of[:, :-1], region_of[:, 1:])
    acc(region_of[:-1, :], region_of[1:, :])

    neighbors: dict[int, set] = defaultdict(set)
    btotal: dict[int, int] = defaultdict(int)
    for (a, b), c in shared.items():
        neighbors[a].add(b)
        neighbors[b].add(a)
        btotal[a] += c
        btotal[b] += c

    touches = set(region_of[0, :].tolist()) | set(region_of[-1, :].tolist()) \
        | set(region_of[:, 0].tolist()) | set(region_of[:, -1].tolist())
    return shared, neighbors, btotal, touches


# ---------------------------------------------------------------------------
# Renk karışım testi (lineer RGB) — fringe vs gerçek üçüncü renk ayrımı
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _blend_residual(cR: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> tuple[float, float]:
    """cR, cA ve cB'nin lineer-RGB karışımına ne kadar yakın? (residual, t).

    residual: en iyi t için |lin(cR) - (t·lin(cA)+(1-t)·lin(cB))| (0..~1.7).
    t: karışım oranı (0=cB, 1=cA). Küçük residual + orta t → anti-alias fringe.
    """
    lR, lA, lB = _srgb_to_linear(cR), _srgb_to_linear(cA), _srgb_to_linear(cB)
    d = lA - lB
    denom = float(d @ d)
    if denom < 1e-9:
        return float(np.linalg.norm(lR - lA)), 0.5
    t = float((lR - lB) @ d / denom)
    t = min(1.0, max(0.0, t))
    proj = t * lA + (1 - t) * lB
    return float(np.linalg.norm(lR - proj)), t


def _color_distance_lin(cR: np.ndarray, cT: np.ndarray) -> float:
    return float(np.linalg.norm(_srgb_to_linear(cR) - _srgb_to_linear(cT)))


# ---------------------------------------------------------------------------
# Küçük region geometrisi (yalnız aday small region'larda — hızlı/bellek dostu)
# ---------------------------------------------------------------------------
def _region_geom(region_of: np.ndarray, rid: int, bbox: tuple) -> tuple[float, float, bool]:
    """(max_width, elongation, has_hole). Yalnız bbox crop'unda hesap."""
    x, y, w, h = bbox
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = x + w + 1, y + h + 1
    crop = (region_of[y0:y1, x0:x1] == rid).astype(np.uint8)
    if crop.sum() == 0:
        return 0.0, 0.0, False
    dt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    max_width = 2.0 * float(dt.max())
    area = float(crop.sum())
    # elongation ≈ perimeter²/(4π·area) (compactness); ince şeritte büyük
    cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts) if cnts else 0.0
    elong = (perim * perim) / (4.0 * np.pi * area) if area > 0 else 0.0
    # delik: dış konturla doldurulmuş alan > gerçek alan
    filled = np.zeros_like(crop)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    has_hole = bool(filled.sum() - crop.sum() > 0)
    return max_width, float(elong), has_hole


# ---------------------------------------------------------------------------
# Topoloji özeti (önce/sonra karşılaştırma)
# ---------------------------------------------------------------------------
def _topology_stats(labels: np.ndarray) -> dict[str, Any]:
    region_of, info = _region_map(labels)
    holes = 0
    # delik sayısı: her renk için (bağlı bileşen) - (dış+delikli Euler) yerine
    # basit ölçüt — her renk maskesinin Euler numarası ile bileşen-delik farkı
    for cid in range(int(labels.max()) + 1):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n_cc, _ = cv2.connectedComponents(mask, connectivity=4)
        # arka planın (0) delikleri = maskenin içindeki dış-bağlı olmayan boşluk
        inv = 1 - mask
        n_bg, _ = cv2.connectedComponents(inv.astype(np.uint8), connectivity=8)
        holes += max(0, (n_bg - 1) - 0)  # kaba; karşılaştırma için tutarlı
    return {"regions": len(info), "holes_proxy": holes,
            "colors": int(labels.max()) + 1}


# ---------------------------------------------------------------------------
# Ana consolidation
# ---------------------------------------------------------------------------
def consolidate_regions(labels: np.ndarray, source_rgb: np.ndarray,
                        fills_rgb: np.ndarray,
                        blend_tol: float = 0.10,
                        island_color_tol: float = 0.06) -> ConsolidatedRegionMap:
    """Mikro-region'ları semantik yüze bağlar. Girdi DEĞİŞMEZ (kopya döner).

    ``blend_tol``: fringe için maks lineer-RGB blend residual.
    ``island_color_tol``: island için maks lineer-RGB renk uzaklığı.
    Eşikler ek olarak ölçek-normalize genişlik kapısıyla birleşir.
    """
    labels = np.ascontiguousarray(labels)
    h, w = labels.shape
    diag = float(np.hypot(h, w))
    # GÜVENLİK gerçek testlerdedir (blend/renk/topoloji); genişlik/alan yalnız
    # performans ön-filtresidir. width_gate ölçek-normalize ve MODERATE (yalnız
    # ince AA fringe absorbe edilir — geniş gri bant "gerçek" kabul edilir);
    # area_gate cömert (büyük blob'larda DT hesaplamayı atlamak için).
    width_gate = max(2.5, round(diag / 900.0))
    area_gate = 0.15 * h * w

    region_of, info = _region_map(labels)
    shared, neighbors, btotal, touches = _adjacency(region_of)

    def border_colors(rid: int) -> dict[int, int]:
        out: dict[int, int] = {}
        for nb in neighbors.get(rid, ()):
            c = info[nb][0]
            key = (min(rid, nb), max(rid, nb))
            out[c] = out.get(c, 0) + shared.get(key, 0)
        return out

    fills = fills_rgb.astype(np.uint8)
    merge_target: dict[int, int] = {}       # rid -> hedef region
    log: list[RegionMergeCandidate] = []
    preserved: list[int] = []

    # aday sırası determinist (region_id)
    for rid in sorted(info):
        color_id, area, bbox = info[rid]
        if area > area_gate:
            continue  # büyük region → otomatik korunur (aday değil)
        max_width, elong, has_hole = _region_geom(region_of, rid, bbox)
        if max_width > width_gate:
            continue  # yeterince ince değil → aday değil (gerçek yüz olabilir)
        bc = border_colors(rid)
        other_colors = {c: L for c, L in bc.items() if c != color_id}
        cR = fills[color_id]

        cand: RegionMergeCandidate | None = None

        # --- FRINGE: tam 2 komşu renk, cR onların lineer karışımı ---
        if len(other_colors) == 2 and not has_hole:
            (cAid, LA), (cBid, LB) = sorted(other_colors.items())
            resid, t = _blend_residual(cR, fills[cAid], fills[cBid])
            if resid <= blend_tol:
                # hedef renk: karışımda baskın taraf (t büyükse cA)
                tgt_color = cAid if t >= 0.5 else cBid
                tgt = _pick_target_region(rid, tgt_color, neighbors, shared, info)
                safe, reason = _topology_safe(rid, tgt, tgt_color, neighbors, info)
                cand = RegionMergeCandidate(
                    rid, tgt if tgt is not None else -1, color_id, tgt_color,
                    area, max_width, elong, float(bc.get(tgt_color, 0)),
                    _color_distance_lin(cR, fills[tgt_color]), resid, "fringe",
                    safe and tgt is not None,
                    semantic_score=(blend_tol - resid) + bc.get(tgt_color, 0) / max(1, btotal[rid]),
                    accepted=safe and tgt is not None,
                    rejection_reason=None if (safe and tgt is not None) else (reason or "hedef yok"))

        # --- ISLAND: tek baskın komşu renk, cR ≈ o renk ---
        elif len(other_colors) == 1 and not has_hole:
            (cAid, LA), = other_colors.items()
            cdist = _color_distance_lin(cR, fills[cAid])
            dom = LA / max(1, btotal[rid])
            if cdist <= island_color_tol and dom >= 0.80:
                tgt = _pick_target_region(rid, cAid, neighbors, shared, info)
                safe, reason = _topology_safe(rid, tgt, cAid, neighbors, info)
                cand = RegionMergeCandidate(
                    rid, tgt if tgt is not None else -1, color_id, cAid, area,
                    max_width, elong, float(LA), cdist, 0.0, "island",
                    safe and tgt is not None,
                    semantic_score=(island_color_tol - cdist) + dom,
                    accepted=safe and tgt is not None,
                    rejection_reason=None if (safe and tgt is not None) else (reason or "hedef yok"))

        if cand is None:
            preserved.append(rid)
            continue
        log.append(cand)
        if cand.accepted:
            merge_target[rid] = cand.target_region_id
        else:
            preserved.append(rid)

    # --- determinist union-find: her region grubunun anchor'ı = en büyük non-merge ---
    parent = {rid: rid for rid in info}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid in sorted(merge_target):
        parent[find(rid)] = find(merge_target[rid])

    groups: dict[int, list[int]] = {}
    for rid in info:
        groups.setdefault(find(rid), []).append(rid)

    orig_to_cons: dict[int, int] = {}
    merged_groups: dict[int, list[int]] = {}
    consolidated = labels.copy()
    for root, members in groups.items():
        # anchor = grup içinde merge edilmemiş en büyük region (yoksa en büyük)
        non_merged = [m for m in members if m not in merge_target]
        pool = non_merged if non_merged else members
        anchor = max(pool, key=lambda m: (info[m][1], -m))
        anchor_color = info[anchor][0]
        if len(members) > 1:
            merged_groups[anchor] = sorted(members)
        for m in members:
            orig_to_cons[m] = anchor
            if info[m][0] != anchor_color:
                consolidated[region_of == m] = anchor_color

    stats_before = _topology_stats(labels)
    stats_after = _topology_stats(consolidated)
    dh = hashlib.blake2b(consolidated.tobytes()
                         + fills.tobytes(), digest_size=16).hexdigest()

    return ConsolidatedRegionMap(
        consolidated_labels=consolidated, original_to_consolidated=orig_to_cons,
        merged_region_groups=merged_groups, preserved_micro_regions=preserved,
        merge_log=log, stats_before=stats_before, stats_after=stats_after,
        deterministic_hash=dh)


def _pick_target_region(rid: int, tgt_color: int, neighbors, shared, info) -> int | None:
    """Hedef renkte, en uzun ortak sınıra sahip TEK komşu region'ı seç."""
    cands = [nb for nb in neighbors.get(rid, ()) if info[nb][0] == tgt_color]
    if not cands:
        return None
    return max(cands, key=lambda nb: (shared.get((min(rid, nb), max(rid, nb)), 0), -nb))


def _topology_safe(rid: int, tgt: int | None, tgt_color: int, neighbors, info) -> tuple[bool, str | None]:
    """Merge topoloji-güvenli mi? (iki ayrı hedef-renk bileşenini birleştirmez)."""
    if tgt is None:
        return False, "hedef region yok"
    # rid, hedef renkten BİRDEN çok bileşene komşuysa → birleştirme iki gerçek
    # yüzü yanlış birleştirir (aynı-renk disconnected koruması)
    tgt_comps = [nb for nb in neighbors.get(rid, ()) if info[nb][0] == tgt_color]
    if len(tgt_comps) > 1:
        return False, "hedef renkte >1 komşu bileşen (yanlış birleşme riski)"
    return True, None
