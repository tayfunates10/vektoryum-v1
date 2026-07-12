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
