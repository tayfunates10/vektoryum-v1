"""Shared-boundary half-edge (DCEL) graph — FAZ HG-1/HG-2 (SHADOW).

Mevcut mimaride aynı fiziksel renk sınırı iki komşu yüz tarafından iki ayrı
kontur olarak temsil edilebiliyor. Bu modül kaynağın gerçek düzlemsel
altbölünmesini kurar: her fiziksel sınır TEK canonical curve, iki komşu yüz
onu TWIN half-edge'lerle ters yönde paylaşır, junction'lar TEK shared vertex
üzerinden bağlanır. Böylece bağımsız-yüz fit farkları, sliver ve seam'ler
kökten önlenir; gerçek graph-tabanlı evenodd cut-out yüzler mümkün olur.

Bu sürüm SHADOW'dur: yalnız kaynak palet-sınıflı rasterden graph kurar ve
DOĞRULAR; production SVG serileştirmesine BAĞLANMAZ (şartname FAZ HG-1:
"Bu faz production SVG'yi değiştirmemeli"). Koordinatlar kafes-köşe (tamsayı,
kaynak uzay) çözünürlüğündedir; alt-piksel canonical curve fitting HG-3'tür.

Kurulum kafes (crack-edge) modeliyle yapılır: iki komşu farklı-bölge piksel
arasındaki birim sınır, piksel-köşe kafesinde bir crack-edge'dir. Görüntü
1-piksel EXTERIOR bölgesiyle çevrilir; böylece dış silhouette da twin'li olur
(exterior face). Determinist: bölgeler/curve'ler/vertex'ler konuma göre
sıralanıp kanonik ID alır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Point = tuple[float, float]
_EXT = -2  # exterior (görüntü dışı) sözde-bölge


# ---------------------------------------------------------------------------
# Veri modeli (BÖLÜM 1)
# ---------------------------------------------------------------------------
@dataclass
class HalfEdgeVertex:
    vertex_id: str
    point: Point
    incident_half_edge_ids: list[str] = field(default_factory=list)
    junction_id: str | None = None
    junction_type: str | None = None
    locked: bool = True
    confidence: float = 1.0


@dataclass
class CanonicalBoundaryCurve:
    curve_id: str
    start_vertex_id: str
    end_vertex_id: str
    polyline: list[Point]                    # kafes noktaları (crack staircase)
    adjacent_face_ids: tuple[str, str | None]
    is_exterior: bool = False
    third_region_risk: bool = False
    confidence: float = 1.0
    # --- HG-3 alt-piksel canonical fit (twin'ler AYNI listeyi paylaşır) -------
    fitted_segments: list[Any] = field(default_factory=list)  # [P0,C1,C2,P1] listesi
    fit_error_max: float = 0.0
    fit_error_p95: float = 0.0
    command_count: int = 0
    primitive_kind: str = ""                 # "" | line | cubic
    fit_fallback: bool = False               # düşük güven → ham polyline


@dataclass
class HalfEdge:
    half_edge_id: str
    origin_vertex_id: str
    destination_vertex_id: str
    twin_id: str | None
    next_id: str | None
    previous_id: str | None
    face_id: str
    curve_id: str
    reversed: bool
    boundary_role: str = "internal"          # internal | exterior
    active: bool = True


@dataclass
class HalfEdgeFace:
    face_id: str
    source_region_id: str
    color_id: int
    fill_color: str
    outer_cycle_id: str | None = None
    inner_cycle_ids: list[str] = field(default_factory=list)
    z_order: int = 0
    area: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    hole_count: int = 0
    touches_canvas: bool = False
    is_background: bool = False
    is_exterior: bool = False
    visible: bool = True


@dataclass
class HalfEdgeCycle:
    cycle_id: str
    face_id: str
    half_edge_ids: list[str]
    signed_area: float
    orientation: str                         # ccw | cw
    is_outer: bool
    closed: bool


@dataclass
class SharedBoundaryHalfEdgeGraph:
    vertices: dict[str, HalfEdgeVertex] = field(default_factory=dict)
    curves: dict[str, CanonicalBoundaryCurve] = field(default_factory=dict)
    half_edges: dict[str, HalfEdge] = field(default_factory=dict)
    faces: dict[str, HalfEdgeFace] = field(default_factory=dict)
    cycles: dict[str, HalfEdgeCycle] = field(default_factory=dict)
    geometry_version: int = 0
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        jt: dict[str, int] = {}
        for v in self.vertices.values():
            if v.junction_type:
                jt[v.junction_type] = jt.get(v.junction_type, 0) + 1
        return {
            "vertices": len(self.vertices),
            "curves": len(self.curves),
            "half_edges": len(self.half_edges),
            "twins": sum(1 for h in self.half_edges.values() if h.twin_id),
            "faces": len(self.faces),
            "visible_faces": sum(1 for f in self.faces.values() if f.visible and not f.is_exterior),
            "cycles": len(self.cycles),
            "outer_cycles": sum(1 for c in self.cycles.values() if c.is_outer),
            "inner_cycles": sum(1 for c in self.cycles.values() if not c.is_outer),
            "junctions": sum(1 for v in self.vertices.values() if v.junction_id),
            "junction_types": jt,
            "valid": self.valid,
            "errors": self.validation_errors[:8],
        }


# ---------------------------------------------------------------------------
# Kurulum (BÖLÜM 4/5) — kafes crack-edge modeli
# ---------------------------------------------------------------------------
def _region_map(labels: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """Her piksele BAĞLI bölge indeksi atar (sınıf başına connectedComponents).

    Aynı renkte ayrık bölgeler ayrı indeks alır. min-area YOK (graph her
    pikseli kapsamalı). Döner: (region_of HxW int32, {region_idx: (color_id, area)}).
    """
    h, w = labels.shape
    region_of = np.full((h, w), -1, np.int32)
    info: dict[int, tuple[int, int]] = {}
    nxt = 0
    for cid in range(int(labels.max()) + 1):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        # connectivity=4: köşegen-temas eden pikseller AYRI bölge olsun (aksi
        # halde corner-pinch yüz = tek yüzde iki ayrık dış kontur → geçersiz
        # düzlemsel altbölünme). Keyword şart: pozisyonel 2. arg=labels.
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            region_of[lab == i] = nxt
            info[nxt] = (cid, int(stats[i][4]))
            nxt += 1
    return region_of, info


def _pad_exterior(region_of: np.ndarray) -> np.ndarray:
    """1-piksel EXTERIOR çerçevesiyle çevir (dış silhouette twin'li olsun)."""
    h, w = region_of.shape
    padded = np.full((h + 2, w + 2), _EXT, np.int32)
    padded[1:-1, 1:-1] = region_of
    return padded


def build_half_edge_graph(labels: np.ndarray, fills_hex: list[str] | None = None,
                          geometry_version: int = 0) -> SharedBoundaryHalfEdgeGraph:
    """Palet-sınıflı (H,W) uint8 rasterden shared-boundary half-edge graph kurar.

    ``fills_hex[color_id]`` verilirse yüz dolgu renkleri atanır. Determinist.
    """
    region_of0, rinfo = _region_map(labels)
    reg = _pad_exterior(region_of0)          # (H+2, W+2), kafes bu uzayda
    ph, pw = reg.shape
    # kafes köşeleri: (H+3)x(W+3) — köşe (i,j) reg[j-1..j, i-1..i] pikselleri arası
    # crack-edge: yatay köşe (i,j)-(i+1,j) reg[j-1,i] vs reg[j,i] farklıysa;
    #             dikey köşe (i,j)-(i,j+1) reg[j,i-1] vs reg[j,i] farklıysa
    # (i: 0..pw, j: 0..ph)
    graph = SharedBoundaryHalfEdgeGraph(geometry_version=geometry_version)

    # --- crack-edge kenar kümesi: köşe->komşu köşe, (regA,regB) ------------
    # her crack-edge'i (v0, v1, left_region, right_region) olarak sakla
    # yönelim: kenarı SOLUNDAKİ bölge yüzün ccw sınırını izler
    edges: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []

    def reg_at(px: int, py: int) -> int:
        if 0 <= px < pw and 0 <= py < ph:
            return int(reg[py, px])
        return _EXT

    # yatay crack-edge'ler: köşe (i,j)->(i+1,j); üst piksel (i,j-1), alt (i,j)
    for j in range(ph + 1):
        for i in range(pw):
            top = reg_at(i, j - 1)
            bot = reg_at(i, j)
            if top != bot:
                # kenar soldan sağa; solundaki (üstteki) bölge = top
                edges.append(((i, j), (i + 1, j), top, bot))
    # dikey crack-edge'ler: köşe (i,j)->(i,j+1) yönü +y (aşağı). Ekran (+y aşağı)
    # yürüyüşünde sol-el yönü (dy,-dx)=(1,0)=+x → sol bölge SAĞ piksel (i,j),
    # sağ bölge SOL piksel (i-1,j). (Yatay kenarla tutarlı winding.)
    for j in range(ph):
        for i in range(pw + 1):
            left = reg_at(i, j)
            right = reg_at(i - 1, j)
            if left != right:
                edges.append(((i, j), (i, j + 1), left, right))

    # --- köşe derecesi + junction tespiti ---------------------------------
    from collections import defaultdict

    corner_deg: dict[tuple[int, int], int] = defaultdict(int)
    for v0, v1, _l, _r in edges:
        corner_deg[v0] += 1
        corner_deg[v1] += 1

    def corner_regions(i: int, j: int) -> set[int]:
        return {reg_at(i - 1, j - 1), reg_at(i, j - 1), reg_at(i - 1, j), reg_at(i, j)}

    # vertex = köşe derecesi != 2 VEYA çevresinde ≥3 bölge (junction)
    vertex_corners: set[tuple[int, int]] = set()
    for c, d in corner_deg.items():
        if d != 2 or len(corner_regions(*c)) >= 3:
            vertex_corners.add(c)

    # --- boundary chain tracing (vertex'ten vertex'e) ---------------------
    # kenar komşuluk: köşe -> [(komşu köşe, left, right)]
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int, int]]] = defaultdict(list)
    for v0, v1, l, r in edges:
        adj[v0].append((v1, l, r))
        adj[v1].append((v0, r, l))  # ters yönde left/right takas

    visited_edge: set[tuple] = set()

    def ekey(a, b):
        return (a, b) if a <= b else (b, a)

    chains: list[dict] = []
    for start in sorted(vertex_corners):
        for (nb, l, r) in sorted(adj[start]):
            if ekey(start, nb) in visited_edge:
                continue
            # zinciri izle
            pts = [start]
            prev, cur = start, nb
            cl, cr = l, r
            visited_edge.add(ekey(prev, cur))
            pts.append(cur)
            while cur not in vertex_corners:
                # derece-2 köşe: gelinen dışındaki tek kenar
                nxts = [(n2, l2, r2) for (n2, l2, r2) in adj[cur]
                        if ekey(cur, n2) not in visited_edge]
                if not nxts:
                    break
                nb2, l2, r2 = nxts[0]
                visited_edge.add(ekey(cur, nb2))
                prev, cur = cur, nb2
                pts.append(cur)
            chains.append({"pts": pts, "left": cl, "right": cr,
                           "v_start": start, "v_end": cur})

    # kapalı loop'lar (junction'sız): kalan ziyaret edilmemiş kenarlar
    for v0, v1, l, r in edges:
        if ekey(v0, v1) in visited_edge:
            continue
        pts = [v0]
        prev, cur = v0, v1
        cl, cr = l, r
        visited_edge.add(ekey(prev, cur))
        pts.append(cur)
        while cur != v0:
            nxts = [(n2, l2, r2) for (n2, l2, r2) in adj[cur]
                    if ekey(cur, n2) not in visited_edge]
            if not nxts:
                break
            nb2, l2, r2 = nxts[0]
            visited_edge.add(ekey(cur, nb2))
            prev, cur = cur, nb2
            pts.append(cur)
        chains.append({"pts": pts, "left": cl, "right": cr,
                       "v_start": v0, "v_end": v0, "loop": True})

    # --- determinist ID: bölge -> face --------------------------------------
    region_ids = sorted(set(region_of0.ravel().tolist()) - {-1}) + [_EXT]
    face_of: dict[int, str] = {}
    for idx, rid in enumerate(sorted(region_ids, key=lambda r: (
            rinfo.get(r, (99, 0))[0], -rinfo.get(r, (0, 0))[1], r))):
        color_id, area = rinfo.get(rid, (-1, 0))
        is_ext = rid == _EXT
        fid = "face_ext" if is_ext else f"face_{idx}"
        face_of[rid] = fid
        fill = ""
        if fills_hex and 0 <= color_id < len(fills_hex):
            fill = fills_hex[color_id]
        graph.faces[fid] = HalfEdgeFace(
            face_id=fid, source_region_id=str(rid), color_id=color_id,
            fill_color=fill, area=float(area), is_exterior=is_ext,
            visible=not is_ext, z_order=idx)

    # --- vertex nesneleri (kaynak uzaya kaydır: -1 padding) ----------------
    vid_of: dict[tuple[int, int], str] = {}
    for k, c in enumerate(sorted(vertex_corners)):
        vid = f"v_{k}"
        vid_of[c] = vid
        jtype = None
        jr = corner_regions(*c)
        if len(jr) >= 4:
            jtype = "X"
        elif len(jr) == 3:
            jtype = "T"
        graph.vertices[vid] = HalfEdgeVertex(
            vertex_id=vid, point=(float(c[0] - 1), float(c[1] - 1)),
            junction_id=(f"j_{k}" if jtype else None), junction_type=jtype)

    # loop zincirleri için sözde-vertex (başlangıç köşesi)
    def ensure_vertex(c: tuple[int, int]) -> str:
        if c in vid_of:
            return vid_of[c]
        vid = f"vl_{len(vid_of)}"
        vid_of[c] = vid
        graph.vertices[vid] = HalfEdgeVertex(
            vertex_id=vid, point=(float(c[0] - 1), float(c[1] - 1)))
        return vid

    # --- canonical curve + twin half-edge --------------------------------
    he_counter = 0
    for ci, ch in enumerate(chains):
        la, rb = ch["left"], ch["right"]
        fa, fb = face_of.get(la), face_of.get(rb)
        if fa is None or fb is None:
            continue
        vs = ensure_vertex(ch["v_start"])
        ve = ensure_vertex(ch["v_end"])
        poly = [(float(x - 1), float(y - 1)) for (x, y) in ch["pts"]]
        cid = f"curve_{ci}"
        is_ext = (la == _EXT or rb == _EXT)
        graph.curves[cid] = CanonicalBoundaryCurve(
            curve_id=cid, start_vertex_id=vs, end_vertex_id=ve, polyline=poly,
            adjacent_face_ids=(fa, fb), is_exterior=is_ext)
        # twin half-edge'ler: he_a (face la, forward), he_b (face rb, reversed)
        ha = f"he_{he_counter}"; hb = f"he_{he_counter + 1}"; he_counter += 2
        graph.half_edges[ha] = HalfEdge(
            half_edge_id=ha, origin_vertex_id=vs, destination_vertex_id=ve,
            twin_id=hb, next_id=None, previous_id=None, face_id=fa, curve_id=cid,
            reversed=False, boundary_role=("exterior" if is_ext else "internal"))
        graph.half_edges[hb] = HalfEdge(
            half_edge_id=hb, origin_vertex_id=ve, destination_vertex_id=vs,
            twin_id=ha, next_id=None, previous_id=None, face_id=fb, curve_id=cid,
            reversed=True, boundary_role=("exterior" if is_ext else "internal"))
        graph.vertices[vs].incident_half_edge_ids.append(ha)
        graph.vertices[ve].incident_half_edge_ids.append(hb)

    _link_cycles(graph)
    validate_graph(graph)
    return graph


def _link_cycles(graph: SharedBoundaryHalfEdgeGraph) -> None:
    """Half-edge'leri face cycle'larına bağlar (vertex rotasyonu ile next/prev).

    Standart DCEL: bir half-edge h'nin next'i, h.destination'da h.twin'den
    saat yönünde sonraki half-edge'dir. Köşe etrafındaki half-edge stub'ları
    açıyla sıralanır. Kafes geometrisinde açılar 0/90/180/270'tir; çok-kollu
    junction'da bu sıralama yüzleri doğru ayırır.
    """
    import math

    # her vertex'te ÇIKAN half-edge'ler (origin==v) açı sırasıyla
    out_by_v: dict[str, list[tuple[float, str]]] = {}
    for h in graph.half_edges.values():
        v = h.origin_vertex_id
        p0 = graph.vertices[h.origin_vertex_id].point
        # ikinci nokta: curve polyline'ının origin ucundan bir sonraki
        cur = graph.curves[h.curve_id]
        poly = cur.polyline if not h.reversed else cur.polyline[::-1]
        p1 = poly[1] if len(poly) > 1 else poly[0]
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        out_by_v.setdefault(v, []).append((ang, h.half_edge_id))
    for v in out_by_v:
        out_by_v[v].sort()

    # next(h): h.destination = v; h.twin v'den ÇIKAR; twin'in açısal olarak
    # bir SONRAKİ çıkan half-edge = next(h) (saat yönü / ccw yüz için)
    for h in graph.half_edges.values():
        v = h.destination_vertex_id
        outs = out_by_v.get(v, [])
        if not outs:
            continue
        twin = h.twin_id
        idx = next((k for k, (_a, hid) in enumerate(outs) if hid == twin), None)
        if idx is None:
            continue
        nxt = outs[(idx + 1) % len(outs)][1]
        h.next_id = nxt
        graph.half_edges[nxt].previous_id = h.half_edge_id

    # cycle'ları topla (next zinciri)
    seen: set[str] = set()
    cyc_i = 0
    for hid in sorted(graph.half_edges):
        if hid in seen:
            continue
        chain = []
        cur = hid
        ok = True
        for _ in range(len(graph.half_edges) + 1):
            if cur is None:
                ok = False
                break
            chain.append(cur)
            seen.add(cur)
            nx = graph.half_edges[cur].next_id
            if nx == hid:
                break
            if nx in seen and nx != hid:
                ok = False
                break
            cur = nx
        else:
            ok = False
        if not ok or not chain:
            continue
        face_id = graph.half_edges[chain[0]].face_id
        # imzalı alan (shoelace, curve polyline'larıyla). Ekran uzayı +y AŞAĞI:
        # sınırlı yüzün dış konturu saat yönünde döner → shoelace NEGATİF; delik
        # (inner) POZİTİF. Bu yüzden outer = (area < 0).
        area = _cycle_signed_area(graph, chain)
        is_outer = area < 0
        cyc_id = f"cycle_{cyc_i}"; cyc_i += 1
        graph.cycles[cyc_id] = HalfEdgeCycle(
            cycle_id=cyc_id, face_id=face_id, half_edge_ids=chain,
            signed_area=area, orientation="ccw" if is_outer else "cw",
            is_outer=is_outer, closed=True)
        f = graph.faces.get(face_id)
        if f is not None:
            if is_outer and f.outer_cycle_id is None:
                f.outer_cycle_id = cyc_id
            else:
                f.inner_cycle_ids.append(cyc_id)
    for f in graph.faces.values():
        f.hole_count = len(f.inner_cycle_ids)


def _cycle_signed_area(graph: SharedBoundaryHalfEdgeGraph, chain: list[str]) -> float:
    pts: list[Point] = []
    for hid in chain:
        h = graph.half_edges[hid]
        cur = graph.curves[h.curve_id]
        poly = cur.polyline if not h.reversed else cur.polyline[::-1]
        pts.extend(poly[:-1])
    if len(pts) < 3:
        return 0.0
    a = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        a += x0 * y1 - x1 * y0
    return a / 2.0


# ---------------------------------------------------------------------------
# Validator (BÖLÜM 2) — graph invariant'ları
# ---------------------------------------------------------------------------
def validate_graph(graph: SharedBoundaryHalfEdgeGraph) -> bool:
    errors: list[str] = []
    he = graph.half_edges

    # 1-4: twin varlığı + aynı curve + ters yön + origin/dest ters
    for h in he.values():
        if h.twin_id is None or h.twin_id not in he:
            errors.append(f"{h.half_edge_id}: twin yok")
            continue
        t = he[h.twin_id]
        if t.curve_id != h.curve_id:
            errors.append(f"{h.half_edge_id}: twin farklı curve")
        if t.reversed == h.reversed:
            errors.append(f"{h.half_edge_id}: twin aynı yön")
        if t.origin_vertex_id != h.destination_vertex_id or \
                t.destination_vertex_id != h.origin_vertex_id:
            errors.append(f"{h.half_edge_id}: twin origin/dest ters değil")

    # 5-7: cycle kapalılığı + next/prev tutarlılığı
    for h in he.values():
        if h.next_id is not None:
            nx = he.get(h.next_id)
            if nx is None or nx.previous_id != h.half_edge_id:
                errors.append(f"{h.half_edge_id}: next.previous != self")
        if h.previous_id is not None:
            pv = he.get(h.previous_id)
            if pv is None or pv.next_id != h.half_edge_id:
                errors.append(f"{h.half_edge_id}: previous.next != self")

    # 9: her internal curve en fazla iki görünür face
    for c in graph.curves.values():
        fa, fb = c.adjacent_face_ids
        vis = sum(1 for f in (fa, fb) if f and graph.faces.get(f)
                  and graph.faces[f].visible)
        if vis > 2:
            errors.append(f"{c.curve_id}: >2 görünür face")

    # 10: junction'daki incident half-edge endpoint'leri aynı vertex koordinatı
    #     (yapı gereği: incident listesi tek vertex nesnesine bağlı — kontrol)
    for v in graph.vertices.values():
        for hid in v.incident_half_edge_ids:
            h = he.get(hid)
            if h and h.origin_vertex_id != v.vertex_id:
                errors.append(f"{v.vertex_id}: incident half-edge origin uyumsuz")

    # 18: NaN/Inf yok
    for v in graph.vertices.values():
        if not (np.isfinite(v.point[0]) and np.isfinite(v.point[1])):
            errors.append(f"{v.vertex_id}: NaN/Inf koordinat")

    # 23: twin curve koordinat eşitliği (aynı curve nesnesi → bit düzeyinde)
    for h in he.values():
        if h.twin_id and h.twin_id in he:
            if he[h.twin_id].curve_id != h.curve_id:
                errors.append(f"{h.half_edge_id}: twin curve id eşit değil")

    # 24: her görünür yüzün TAM bir outer cycle'ı (corner-pinch yoksa)
    for f in graph.faces.values():
        if f.visible and not f.is_exterior and f.outer_cycle_id is None:
            errors.append(f"{f.face_id}: görünür yüzde outer cycle yok")
    # her yüzün inner cycle'ları gerçekten inner (is_outer=False) olmalı
    for f in graph.faces.values():
        for cyc in f.inner_cycle_ids:
            c = graph.cycles.get(cyc)
            if c is not None and c.is_outer:
                errors.append(f"{f.face_id}: outer-yönlü cycle inner listesinde "
                              "(corner-pinch?)")

    # 25: Euler karakteristiği — geçerli düzlemsel altbölünme V-E+F=1+C.
    # F = YÜZ (bölge) sayısı = len(faces) (delikli yüz TEK yüz; kenar başına
    # curve, köşe başına vertex). C = kafes graph'ının bağlı bileşen sayısı.
    # (Not: DCEL cycle sayısı = F + toplam-delik-bileşeni; Euler için F kullan.)
    if he:
        V = len(graph.vertices)
        E = len(graph.curves)
        F = len(graph.faces)
        C = _connected_components(graph)
        if V - E + F != 1 + C:
            errors.append(f"Euler ihlali: V-E+F={V - E + F} != 1+C={1 + C}")

    graph.validation_errors = errors
    graph.valid = not errors
    return graph.valid


def _connected_components(graph: SharedBoundaryHalfEdgeGraph) -> int:
    """Kafes graph'ının bağlı bileşen sayısı (vertex'ler + curve kenarları)."""
    parent: dict[str, str] = {v: v for v in graph.vertices}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in graph.curves.values():
        if c.start_vertex_id in parent and c.end_vertex_id in parent:
            union(c.start_vertex_id, c.end_vertex_id)
    roots = {find(v) for v in graph.vertices}
    return len(roots)
