"""Kaynak palet-sınıflı rasterden bölge komşuluk grafiği (region adjacency).

Aynı fiziksel sınırın iki komşu renk bölgesinde bağımsız fit edilmesini
önlemek için ÖNCE kaynağın gerçek bölge yapısını çıkarırız: her bağlı
bölge bir düğüm, komşu bölge çiftleri bir kenar, üç+ bölgenin buluştuğu
noktalar junction'dır. Kanonik sınır düzeltmesi (cusp_refine) bu grafiğin
güvenilir kenarlarında uygulanır.

Tasarım ilkeleri:
* Düğüm = palet sınıfı DEĞİL, gerçek BAĞLI bileşen: aynı renkte iki ayrık
  bölge iki düğümdür (connectedComponents).
* Komşuluk = 4-bağlantı sınır teması; köşegen tek-piksel teması komşuluk
  SAYILMAZ (sahte kenar üretir).
* Üçüncü-bölge güvenliği: iki bölge arasında ince üçüncü renk şeridi varsa
  o iki bölge DOĞRUDAN komşu değildir (third_region_risk işaretlenir).
* Anti-alias koruması: sınıflandırma zaten sert (bant bazlı) olduğundan AA
  pikselleri ayrı bölge oluşturmaz; ayrıca minimum alan eşiği gürültüyü eler.
* Kimlikler DETERMINİST: bölgeler (renk, alan azalan, y, x) ile sıralanır.

Bu modül SALT-OKUR analiz üretir (SVG'yi değiştirmez); cusp_refine kanonik
düğüm üretimini zaten yapar. Grafik, güvenli kenar seçimi ve testler için.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_MIN_REGION_AREA_FRAC = 0.00002   # tuvalin bu kadarından küçük bölge gürültü
_MIN_SHARED_BOUNDARY = 6          # bu kadar sınır pikselinden az = komşu değil


@dataclass
class RegionNode:
    region_id: str
    color_id: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_canvas: bool


@dataclass
class RegionAdjacencyEdge:
    edge_id: str
    region_a: str
    region_b: str
    boundary_length: int
    third_region_risk: bool = False


@dataclass
class RegionJunction:
    junction_id: str
    point: tuple[int, int]
    incident_regions: list[str] = field(default_factory=list)
    junction_type: str = "T"  # T (3 bölge) veya X (4+ bölge)


@dataclass
class ActiveBoundaryMatch:
    """Aktif bir hata blob'unu kaynak grafik kenarına bağlar.

    Çift taraflı senkron duvar refit'inin GİRDİSİ: blob'un iki tarafındaki
    komşu bölgeleri, aralarındaki kanonik kenarı ve üçüncü-renk riskini
    belirler. ``accepted=False`` ise (üçüncü renk / güven düşük / kenar yok)
    çağıran ESKİ güvenli geometriyi korur.
    """

    graph_edge_id: str | None
    region_a: str | None
    region_b: str | None
    color_a: int
    color_b: int
    junction_ids: list[str] = field(default_factory=list)
    third_region_risk: bool = False
    match_score: float = 0.0
    accepted: bool = False
    rejection_reason: str | None = None


@dataclass
class RegionGraph:
    nodes: list[RegionNode] = field(default_factory=list)
    edges: list[RegionAdjacencyEdge] = field(default_factory=list)
    junctions: list[RegionJunction] = field(default_factory=list)
    region_of: np.ndarray | None = None  # (H,W) bölge indeksi (-1 gürültü)


def build_region_graph(labels: np.ndarray, min_area: int | None = None) -> RegionGraph:
    """Sınıf-etiketli (H,W) uint8 görüntüden region-adjacency grafiği kurar.

    ``labels``: her piksel palet sınıf indeksi (bant bazlı sınıflandırmadan).
    Determinist: bölgeler renk+alan+konum ile sıralanıp yeniden numaralanır.
    """
    h, w = labels.shape
    if min_area is None:
        min_area = max(20, int(_MIN_REGION_AREA_FRAC * h * w))

    # her sınıf için bağlı bileşenler → bölge etiketi (global)
    region_of = np.full((h, w), -1, dtype=np.int32)
    raw: list[dict] = []
    for cid in range(int(labels.max()) + 1):
        mask = (labels == cid).astype(np.uint8)
        if int(mask.sum()) < min_area:
            continue
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 4)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < min_area:
                continue
            raw.append({
                "color_id": cid, "area": int(area),
                "bbox": (int(x), int(y), int(ww), int(hh)),
                "centroid": (float(cent[i][0]), float(cent[i][1])),
                "touches": bool(x == 0 or y == 0 or x + ww == w or y + hh == h),
                "_mask_lab": (lab, i),
            })
    # determinist sıralama ve numaralama
    raw.sort(key=lambda r: (r["color_id"], -r["area"], round(r["centroid"][1]),
                            round(r["centroid"][0])))
    nodes: list[RegionNode] = []
    for idx, r in enumerate(raw):
        rid = f"r{idx}"
        lab, i = r["_mask_lab"]
        region_of[lab == i] = idx
        nodes.append(RegionNode(
            region_id=rid, color_id=r["color_id"], area=r["area"],
            bbox=r["bbox"], centroid=(round(r["centroid"][0], 1), round(r["centroid"][1], 1)),
            touches_canvas=r["touches"],
        ))

    # 4-komşulukta bölge çiftleri arası sınır pikseli sayımı
    pair_count: dict[tuple[int, int], int] = {}
    a = region_of
    for dy, dx in ((0, 1), (1, 0)):
        b = np.full_like(a, -1)
        if dx:
            b[:, :-1] = a[:, 1:]
        else:
            b[:-1, :] = a[1:, :]
        m = (a >= 0) & (b >= 0) & (a != b)
        ii, jj = a[m], b[m]
        for u, v in zip(ii.tolist(), jj.tolist()):
            k = (u, v) if u < v else (v, u)
            pair_count[k] = pair_count.get(k, 0) + 1

    edges: list[RegionAdjacencyEdge] = []
    for (u, v), cnt in sorted(pair_count.items()):
        if cnt < _MIN_SHARED_BOUNDARY:
            continue
        edges.append(RegionAdjacencyEdge(
            edge_id=f"e{u}_{v}", region_a=nodes[u].region_id,
            region_b=nodes[v].region_id, boundary_length=int(cnt),
        ))

    # junction: 2x2 pencerede 3+ FARKLI bölge (determinist tarama)
    junctions: list[RegionJunction] = []
    seen_j: set[tuple[int, int]] = set()
    win = region_of
    for y in range(h - 1):
        row0, row1 = win[y], win[y + 1]
        for x in range(w - 1):
            vals = {int(row0[x]), int(row0[x + 1]), int(row1[x]), int(row1[x + 1])}
            vals.discard(-1)
            if len(vals) >= 3:
                # yakın junction'ları tekilleştir (8px kümeleme)
                key = (x // 8, y // 8)
                if key in seen_j:
                    continue
                seen_j.add(key)
                junctions.append(RegionJunction(
                    junction_id=f"j{len(junctions)}", point=(x, y),
                    incident_regions=sorted(nodes[i].region_id for i in vals),
                    junction_type="T" if len(vals) == 3 else "X",
                ))

    # üçüncü-bölge riski: iki büyük bölge birbirine komşu görünse de aralarında
    # ince üçüncü bölge varsa (junction'lar aynı çift + üçüncüyü içeriyorsa)
    # işaretle — kanonik birleştirme bu kenarlarda YAPILMAMALI
    for e in edges:
        for j in junctions:
            if e.region_a in j.incident_regions and e.region_b in j.incident_regions \
                    and len(j.incident_regions) >= 3:
                e.third_region_risk = True
                break

    return RegionGraph(nodes=nodes, edges=edges, junctions=junctions,
                       region_of=region_of)


def match_regions_to_paths(
    graph: RegionGraph, path_masks: list[tuple[str, np.ndarray]],
    min_iou: float = 0.5,
) -> dict[str, str]:
    """Kaynak bölge düğümlerini SVG path maskelerine IoU ile eşler.

    ``path_masks``: (path_id, bool_mask) listesi. Her bölge için en yüksek
    IoU'lu path seçilir; IoU < min_iou ise eşleşmemiş bırakılır (fallback,
    güvenli düzeltme uygulanmaz). Determinist.
    """
    result: dict[str, str] = {n.region_id: "" for n in graph.nodes}
    if graph.region_of is None:
        return result
    for node in graph.nodes:
        idx = int(node.region_id[1:])
        rmask = graph.region_of == idx
        ra = int(rmask.sum())
        best_id, best_iou = "", 0.0
        for pid, pmask in path_masks:
            inter = int((rmask & pmask).sum())
            uni = ra + int(pmask.sum()) - inter
            iou = inter / uni if uni else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, pid
        if best_iou >= min_iou:
            result[node.region_id] = best_id
    return result


def match_active_boundary(
    graph: RegionGraph, blob_bbox: tuple[int, int, int, int],
    min_side_frac: float = 0.15,
) -> ActiveBoundaryMatch:
    """Hata blob kutusunu kaynak grafik kenarına bağlar (çift-taraflı refit girdisi).

    Blob kutusu çevresindeki bölgeleri sayar; en baskın iki farklı-renk bölge
    ortak duvarı temsil eder. Bir junction'a (3+ bölge) düşüyorsa ya da baskın
    iki bölge doğrudan komşu değilse (üçüncü renk araya girmiş) ``accepted=False``.
    Determinist. ``blob_bbox`` = (x, y, w, h).
    """
    if graph.region_of is None:
        return ActiveBoundaryMatch(None, None, None, -1, -1, accepted=False,
                                   rejection_reason="region_of yok")
    x, y, bw, bh = blob_bbox
    pad = max(6, (bw + bh) // 4)
    h, w = graph.region_of.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    sub = graph.region_of[y0:y1, x0:x1]
    ids, counts = np.unique(sub[sub >= 0], return_counts=True)
    if len(ids) < 2:
        return ActiveBoundaryMatch(None, None, None, -1, -1, accepted=False,
                                   rejection_reason="yeterli bölge yok")
    order = np.argsort(-counts)
    total = float(counts.sum())
    top = [int(ids[i]) for i in order if counts[i] >= min_side_frac * total][:3]
    if len(top) < 2:
        return ActiveBoundaryMatch(None, None, None, -1, -1, accepted=False,
                                   rejection_reason="baskın iki bölge yok")
    a_idx, b_idx = top[0], top[1]
    node_a, node_b = graph.nodes[a_idx], graph.nodes[b_idx]
    rid_a, rid_b = node_a.region_id, node_b.region_id
    # bu iki bölge arasında gerçek adjacency kenarı var mı?
    edge = None
    for e in graph.edges:
        if {e.region_a, e.region_b} == {rid_a, rid_b}:
            edge = e
            break
    junctions = [j.junction_id for j in graph.junctions
                 if rid_a in j.incident_regions or rid_b in j.incident_regions]
    third_risk = (len(top) >= 3) or (edge is not None and edge.third_region_risk)
    if edge is None:
        return ActiveBoundaryMatch(
            None, rid_a, rid_b, node_a.color_id, node_b.color_id,
            junction_ids=junctions, third_region_risk=third_risk,
            accepted=False, rejection_reason="doğrudan komşu değil (üçüncü renk?)")
    if third_risk:
        return ActiveBoundaryMatch(
            edge.edge_id, rid_a, rid_b, node_a.color_id, node_b.color_id,
            junction_ids=junctions, third_region_risk=True,
            accepted=False, rejection_reason="üçüncü renk riski / junction")
    score = float(min(counts[order[0]], counts[order[1]]) / total)
    return ActiveBoundaryMatch(
        edge.edge_id, rid_a, rid_b, node_a.color_id, node_b.color_id,
        junction_ids=junctions, third_region_risk=False,
        match_score=round(score, 3), accepted=True)
