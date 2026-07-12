"""Dar kama / cusp bölgelerinde segment bölme + yeni çapa ekleme.

Çapa TAŞIMA tabanlı refinement (boundary_refit, local_refine) iki durumda
tıkanır: (1) dar kamada profil penceresi iki kenarı birden görür ve geçiş
reddedilir; (2) kaynak cusp'ı mevcut çapalar arasındaki uzun bir segmentin
İÇİNDE kalır — segment kordu cusp'a hiç uğramaz, çapa taşımak yetmez
(LEGO sarı-siyah kama tepesi: 1101 px'lik blob, maks 7.8 px sapma).

Bu modül kalan hata bloblarını inceler, kaynak cusp noktasını (eksik sınıf
bölgesinin en derin noktası, alt-piksel) bulur, ilgili path segmentini
De Casteljau ile TAM böler (bölme render'ı değiştirmez) ve yeni ortak
çapayı cusp'a taşır. Kama iki komşu rengin ortak sınırıysa İKİ path de
AYNI kaynak cusp koordinatına bölünür (CanonicalBoundary düğümü) — aynı
fiziksel sınır iki farklı eğriyle temsil edilmez, sliver kapanır.

Bütçeler: blob başına tur başına 1 çapa, blob başına toplam 4, görsel
başına toplam 12; en çok 3 tur. Kabul ÖLÇÜM KAPILIDIR: blob alanı ≥%15
veya maks hata ≥%20 azalmalı, toplam hata artmamalı, bölge dışına maddi
fark taşmamalı, komut tavanı aşılmamalı; aksi tur eksiksiz geri alınır.
Determinist: sabit tarama sırası, rastgelelik yok. Harf/içerik semantiği
kullanılmaz. ``VEKTORYUM_CUSP_REFINE=off`` kapatır.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_MIN_BLOB = 12             # değerlendirilecek maddi hata blobu tabanı (px)
                           # (40 değil: dar kama TEPESİNİN son kalıntısı ~42px
                           # alanla 7px derin kalabiliyor — ölçüldü; 3x3 açma
                           # AA çizgi gürültüsünü zaten eler)
_MAX_BLOBS = 6             # tur başına işlenecek en çok blob
_MAX_ROUNDS = 6            # kesin tur sınırı (önbellekle her tur ucuz; kama
                           # tepesi gidişatı 35->16.8->7.2->5.8 <2px için
                           # ~5-6 tur gerektiriyor — ölçüldü)
_MAX_PER_BLOB = 4          # blob başına toplam MANTIKSAL çapa
_MAX_TOTAL = 12            # görsel başına toplam MANTIKSAL çapa
_MAX_RENDER_BUDGET = 40    # istek başına en çok deneme render'ı (süre koruması)
_ANCHOR_DEDUP_TOL = 6.0    # kaynak-uzay: bu mesafedeki cusp aynı mantıksal
                           # birimdir (yeni bütçe değil; birim yerinde ilerler)
_MAX_ANCHOR_MOVE = 10.0    # cusp'a taşıma sınırı (px)
_MIN_SEG_LEN = 6.0         # bundan kısa segment bölünmez (uç çapa taşınır)
_PATH_MATCH_DIST = 40.0    # blob cusp'ına bu mesafedeki sınırlar aday
                           # (derin kesik cusp'larda sınır uzak başlar;
                           # kademeli 10px taşımalarla yaklaşılır)
_PAD = 12


@dataclass
class CuspRefinementCandidate:
    component_id: str
    path_index: int
    subpath_index: int
    segment_index: int
    error_blob_id: int
    source_cusp_point: tuple[float, float]
    current_curve_point: tuple[float, float]
    local_curvature: float
    opening_angle: float
    edge_separation: float
    max_error: float
    p95_error: float
    confidence: float
    accepted: bool = False
    rejection_reason: str | None = None


@dataclass
class CanonicalBoundary:
    """İki komşu bölgenin ortak sınır düğümü: iki path aynı float koordinatı
    paylaşır (ters yönlerde) — aynı fiziksel sınır iki kez fit edilmez."""

    boundary_id: str
    region_a_id: str
    region_b_id: str
    source_points: list[tuple[float, float]] = field(default_factory=list)
    path_a_refs: list[tuple[int, int, int]] = field(default_factory=list)
    path_b_refs: list[tuple[int, int, int]] = field(default_factory=list)
    start_node: tuple[float, float] = (0.0, 0.0)
    end_node: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0


@dataclass
class LogicalBoundaryAnchor:
    """Kaynak uzayda TEK cusp noktası. Ortak sınırın iki tarafına eklenen
    eşleşmiş fiziksel çapalar (region A + region B) TEK mantıksal birimdir:
    bütçeden bir kez düşer, komut sayısı gerçek fiziksel artışı yansıtır.
    Aynı cusp'ın küçük koordinat farklarıyla tekrarı yeni birim OLUŞTURMAZ
    (kaynak-uzay dedup); onun yerine mevcut birim yerinde ilerletilir."""

    logical_id: str
    source_point: tuple[float, float]
    physical_insertions: int = 0
    error_blob_ids: set = field(default_factory=set)
    iteration_created: int = 0
    accepted: bool = True

try:
    from svgpathtools import CubicBezier, Line, parse_path
except ImportError:  # pragma: no cover
    parse_path = None


def is_available() -> bool:
    return parse_path is not None


def _round_d(d: str) -> str:
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.2f}".rstrip("0").rstrip("."), d)


def _subpath_d(sub, closed: bool) -> str:
    d = sub.d()
    return f"{d} Z" if closed else d


def _seg_curvature(seg, t: float) -> float:
    try:
        return abs(seg.curvature(t))
    except Exception:  # noqa: BLE001
        return 0.0


def _nearest_on_path(subs, cusp: complex) -> tuple[int, int, float, complex]:
    """(alt-yol, segment, t, nokta): cusp'a en yakın eğri konumu (determinist)."""
    best = (0, 0, 0.5, None)
    best_d = float("inf")
    for si, sub in enumerate(subs):
        for gi, seg in enumerate(sub):
            for i in range(33):
                t = i / 32.0
                p = seg.point(t)
                dd = abs(p - cusp)
                if dd < best_d:
                    best_d = dd
                    best = (si, gi, t, p)
    # ikili arama ile t incelt (2 adım)
    si, gi, t, _p = best
    seg = subs[si][gi]
    lo, hi = max(0.0, t - 1 / 32), min(1.0, t + 1 / 32)
    for _ in range(8):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if abs(seg.point(m1) - cusp) < abs(seg.point(m2) - cusp):
            hi = m2
        else:
            lo = m1
    t = (lo + hi) / 2
    return si, gi, t, seg.point(t)


def _split_and_place(subs, si: int, gi: int, t: float, cusp: complex) -> bool:
    """Segmenti t'de TAM böler (De Casteljau) ve yeni ortak çapayı cusp'a taşır.

    Bölme tek başına render'ı değiştirmez; görünür değişiklik yalnız çapa
    taşımadan gelir. Cusp KÖŞE olarak ele alınır: C0 korunur, C1 dayatılmaz
    (iki teğet bağımsız), bitişik kontrol noktaları aynı delta ile sürüklenir
    (yerel şekil korunur, overshoot sınırı üstte uygulanır).
    """
    seg = subs[si][gi]
    if isinstance(seg, CubicBezier) or isinstance(seg, Line):
        if t < 0.12:
            # uca çok yakın: bölme yerine mevcut baş çapayı taşı
            delta = cusp - seg.start
            seg1 = seg
            prev = subs[si][gi - 1] if gi > 0 else subs[si][-1]
            _move_joint(prev, seg1, delta)
            return True
        if t > 0.88:
            delta = cusp - seg.end
            nxt = subs[si][(gi + 1) % len(subs[si])]
            _move_joint(seg, nxt, delta)
            return True
        a, b = seg.split(t)
        delta = cusp - a.end
        if isinstance(a, CubicBezier):
            a = CubicBezier(a.start, a.control1, a.control2 + delta, a.end + delta)
            b = CubicBezier(b.start + delta, b.control1 + delta, b.control2, b.end)
        else:
            a = Line(a.start, a.end + delta)
            b = Line(b.start + delta, b.end)
        subs[si][gi:gi + 1] = [a, b]
        return True
    return False  # yay: analitik daire çıktısı — cusp bölmesi uygulanmaz


def _move_joint(seg_a, seg_b, delta: complex) -> None:
    """İki segmentin ORTAK çapasını delta kadar taşır (C0 korunur)."""
    if isinstance(seg_a, CubicBezier):
        seg_a.control2 += delta
        seg_a.end += delta
    elif isinstance(seg_a, Line):
        seg_a.end += delta
    if isinstance(seg_b, CubicBezier):
        seg_b.start += delta
        seg_b.control1 += delta
    elif isinstance(seg_b, Line):
        seg_b.start += delta


def refine_cusp_regions(
    svg_path: Path,
    source_rgb: np.ndarray,
    width: int,
    height: int,
    render_fn: Callable[[Path, int, int], np.ndarray | None],
    max_total_anchors: int = _MAX_TOTAL,
    cache: Any = None,
) -> dict[str, Any]:
    """Kalan hata bloblarındaki dar kama/cusp'ları segment bölerek kapatır."""
    if not is_available():
        return {"status": "skipped", "reason": "svgpathtools yok"}
    from app.palette_ops import abs_diff_sum  # noqa: PLC0415
    from app.palette_ops import classify_rgb as _raw_classify  # noqa: PLC0415

    def classify_rgb(img: np.ndarray, fills: np.ndarray) -> np.ndarray:
        # önbellek varsa render/kaynak sınıflandırması yeniden hesaplanmaz
        if cache is not None:
            if img is source_rgb:
                return cache.classify_source(fills)
            return cache.classify(img, fills)
        return _raw_classify(img, fills)

    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}
    for el in root.iter():
        if el.get("transform"):
            return {"status": "skipped", "reason": "transform içeren belge"}

    path_els = [el for el in root.iter()
                if el.tag.split("}")[-1] == "path" and el.get("d")]
    hexes = [(el.get("fill") or "").lower() for el in path_els]
    if not any(re.fullmatch(r"#[0-9a-f]{6}", h or "") for h in hexes):
        return {"status": "no_change", "reason": "hex dolgu yok"}
    uniq_fills = sorted({h for h in hexes if re.fullmatch(r"#[0-9a-f]{6}", h or "")})
    fills_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                          for h in uniq_fills], dtype=np.float32)
    fill_class = {h: i for i, h in enumerate(uniq_fills)}

    src_cls = classify_rgb(source_rgb, fills_rgb)
    cur = render_fn(svg_path, width, height)
    if cur is None:
        return {"status": "skipped", "reason": "render backend yok"}

    def err_mask(rnd: np.ndarray) -> np.ndarray:
        e = (src_cls != classify_rgb(rnd, fills_rgb)).astype(np.uint8)
        return cv2.morphologyEx(e, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def blob_list(e: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        n, _l, stats, _c = cv2.connectedComponentsWithStats(e, 8)
        out = [(int(a), int(x), int(y), int(w2), int(h2))
               for x, y, w2, h2, a in (stats[i] for i in range(1, n)) if a >= _MIN_BLOB]
        out.sort(reverse=True)
        return out[:_MAX_BLOBS]

    e0 = err_mask(cur)
    err_before = int(e0.sum())
    blobs0 = blob_list(e0)
    if not blobs0:
        return {"status": "no_change", "reason": "maddi hata blobu yok",
                "err_px": err_before}
    # dışarı-taşma koruması maskesi snap erişimini KAPSAMALI: snap bölgesi
    # blob±20 ve tur başına ≤2.5px kayma → ±36 pay. Maske dar kalırsa meşru
    # düzeltme "bölge dışına taşma" sayılıp reddediliyordu (ölçüldü: 108px).
    _GUARD_PAD = 36
    region_mask = np.zeros((height, width), np.uint8)
    for _a, x, y, w2, h2 in blobs0:
        region_mask[max(0, y - _GUARD_PAD):min(height, y + h2 + _GUARD_PAD),
                    max(0, x - _GUARD_PAD):min(width, x + w2 + _GUARD_PAD)] = 1
    out_before = int((e0 & (region_mask == 0)).sum())
    max_dev_before = _max_dev_in_blobs(src_cls, classify_rgb(cur, fills_rgb), blobs0)

    # path geometri önbelleği (alt-yollar segment listesi olarak, düzenlenebilir)
    parsed: dict[int, list[list]] = {}
    closed_flags: dict[int, list[bool]] = {}

    def get_parsed(pi: int):
        if pi not in parsed:
            try:
                p = parse_path(path_els[pi].get("d"))
                subs = [list(s) for s in p.continuous_subpaths()]
                closed_flags[pi] = [s.isclosed() for s in p.continuous_subpaths()]
                parsed[pi] = subs
            except Exception:  # noqa: BLE001
                parsed[pi] = []
                closed_flags[pi] = []
        return parsed[pi]

    def serialize(pi: int) -> str:
        from svgpathtools import Path as SPath  # noqa: PLC0415

        parts = []
        for sub_segs, closed in zip(parsed[pi], closed_flags[pi]):
            parts.append(_subpath_d(SPath(*sub_segs), closed))
        return _round_d(" ".join(parts))

    candidates: list[CuspRefinementCandidate] = []
    boundaries: list[CanonicalBoundary] = []
    logical_anchors: list[LogicalBoundaryAnchor] = []
    tmp = svg_path.with_suffix(".cusp.svg")
    total_added = 0          # MANTIKSAL çapa sayısı (bütçe birimi)
    total_physical = 0       # gerçek fiziksel bölme sayısı (rapor)
    render_budget = _MAX_RENDER_BUDGET
    best_err = err_before
    accepted_blobs = 0

    def _find_logical(pt: complex) -> LogicalBoundaryAnchor | None:
        """pt'ye _ANCHOR_DEDUP_TOL içinde kabul edilmiş mantıksal çapa (varsa)."""
        for la in logical_anchors:
            if abs(complex(la.source_point[0], la.source_point[1]) - pt) <= _ANCHOR_DEDUP_TOL:
                return la
        return None

    from app.local_refine import _snap_subpath_wide  # noqa: PLC0415
    from app.boundary_refit import (  # noqa: PLC0415
        _parse_subpaths_arc,
        _serialize_subpaths_arc,
    )
    ref_f32 = source_rgb.astype(np.float32)

    # blob başına işle ve blob başına KAPILA: uzun bant bloblar tek çapayla
    # %15 küçülmez; tur-toplamı kapısı doğru işi de reddediyordu (ölçüldü).
    # Dış geçiş (≤ _MAX_ROUNDS): kabul edilen değişiklikler komşu blobları
    # erişilebilir kılar; değişmeyen geometri determinist olarak aynı sonucu
    # vereceği için başarısız blob imzaları tekrar denenmez.
    failed_sigs: set[tuple[int, int, int, int]] = set()
    # kümülatif bölgesel bütçe: aynı fiziksel bölge geçişler arasında "yeni
    # blob" olarak tazelenip bütçeyi yenileyemez (bileşen başına ≤ _MAX_PER_BLOB)
    spent_regions: list[list[int]] = []  # [x0,y0,x1,y1,harcanan]

    def _region_budget(bx0: int, by0: int, bx1: int, by1: int) -> list[int]:
        for r in spent_regions:
            if not (bx1 < r[0] or bx0 > r[2] or by1 < r[1] or by0 > r[3]):
                r[0], r[1] = min(r[0], bx0), min(r[1], by0)
                r[2], r[3] = max(r[2], bx1), max(r[3], by1)
                return r
        r = [bx0, by0, bx1, by1, 0]
        spent_regions.append(r)
        return r

    for _pass in range(_MAX_ROUNDS):
      pass_accepted = 0
      rnd_cls = classify_rgb(cur, fills_rgb)
      e_now = err_mask(cur)
      # dışarı tabanı geçiş başına güncellenir (kabul edilen kılcal etkiler
      # sonraki blob kapılarını haksız düşürmesin)
      out_before = int((e_now & (region_mask == 0)).sum())
      blobs = [b for b in blob_list(e_now) if (b[1], b[2], b[3], b[4]) not in failed_sigs]
      if not blobs or total_added >= max_total_anchors or render_budget <= 0:
          break
      for bi, (_a, bx, by, bw, bh) in enumerate(blobs):
        if total_added >= max_total_anchors or render_budget <= 0:
            break
        x0, y0 = max(0, bx - _PAD), max(0, by - _PAD)
        x1, y1 = min(width, bx + bw + _PAD), min(height, by + bh + _PAD)
        sc, rc = src_cls[y0:y1, x0:x1], rnd_cls[y0:y1, x0:x1]
        bad = sc != rc
        if not bad.any():
            continue
        blob_err0 = int(bad.sum())
        blob_uid = _pass * 100 + bi  # pass'lar arası benzersiz (rapor/rollback)
        # (kaynak, render) sınıf çiftleri: ilk 2 çift işlenir — kama tepesinde
        # taşma (A->B) ve çentik (B->A) birlikte görülür; yalnız baskın çifti
        # işlemek çentiği bırakıyordu (ölçüldü: sahte kenar 7.8px)
        pairs = sc[bad].astype(np.int32) * 256 + rc[bad].astype(np.int32)
        vals, cnts = np.unique(pairs, return_counts=True)
        order = np.argsort(-cnts)
        top_pairs = [int(vals[i]) for i in order[:2]
                     if cnts[i] >= max(12, 0.15 * blob_err0)]
        if not top_pairs:
            continue
        budget_rec = _region_budget(x0, y0, x1, y1)
        blob_budget = _MAX_PER_BLOB - budget_rec[4]  # MANTIKSAL bütçe
        if blob_budget <= 0:
            continue  # bu bölgenin mantıksal bütçesi önceki geçişlerde tükendi
        backup_d = {i: path_els[i].get("d") for i in range(len(path_els))}
        parsed_backup = {i: [list(map(_copy_seg, s)) for s in parsed.get(i, [])]
                         for i in parsed}
        blob_changed: set[int] = set()
        blob_added = 0            # fiziksel bölme (komut)
        blob_logical = 0         # YENİ mantıksal çapa (bütçe)
        new_logicals: list[LogicalBoundaryAnchor] = []
        node_refs: dict[str, list[tuple[int, int, int]]] = {"a": [], "b": []}
        cusp0 = None
        cls_a = cls_b = -1
        for pair_v in top_pairs:
            cls_a, cls_b = pair_v // 256, pair_v % 256
            miss = bad & (sc == cls_a) & (rc == cls_b)
            if not miss.any():
                continue
            # derin noktalar: eksik bölgenin render sınırından uzak yerel
            # maksimumlar (uzun bantta birden çok; NMS blob boyutuna uyarlı)
            not_a_rnd = (rc != cls_a).astype(np.uint8)
            dt = cv2.distanceTransform(not_a_rnd, cv2.DIST_L2, 5)
            dt[~miss] = 0
            max_err = float(dt.max())
            if max_err <= 1.0:
                continue
            p95 = float(np.percentile(dt[miss], 95))
            span = max(bw, bh)
            n_pts = int(min(_MAX_PER_BLOB, max(1, round(span / 40))))
            deep_pts = _deep_points(dt, n_pts, nms_radius=max(10.0, span / 6.0))
            inside_dt = cv2.distanceTransform(miss.astype(np.uint8), cv2.DIST_L2, 5)
            edge_sep = float(2.0 * inside_dt.max())
            for px, py in deep_pts:
                if blob_logical >= blob_budget or total_added + blob_logical >= max_total_anchors:
                    break
                cusp = complex(px + x0 + 0.5, py + y0 + 0.5)
                if cusp0 is None:
                    cusp0 = cusp
                # KAYNAK-UZAY DEDUP: bu cusp mevcut bir mantıksal çapaya çok
                # yakınsa YENİ birim değildir — birim yerinde ilerletilir
                # (bütçe tekrar düşmez). Yalnız gerçekten yeni cusp bütçe yer.
                existing = _find_logical(cusp)
                if existing is None and blob_logical >= blob_budget:
                    break
                # her iki taraf path'i: fill'i A / B sınıfında, sınırı yakın
                # olan EN ÜSTTEKİ path (belge sırası → determinist)
                picked: dict[int, int] = {}
                for cls_want in (cls_a, cls_b):
                    for pi in range(len(path_els) - 1, -1, -1):
                        if fill_class.get(hexes[pi]) != cls_want:
                            continue
                        subs = get_parsed(pi)
                        if not subs:
                            continue
                        _si, _gi, _t, ppt = _nearest_on_path(subs, cusp)
                        if abs(ppt - cusp) <= _PATH_MATCH_DIST:
                            picked[cls_want] = pi
                            break
                cusp_physical = 0
                for cls_want, pi in sorted(picked.items()):
                    subs = get_parsed(pi)
                    si, gi, t, ppt = _nearest_on_path(subs, cusp)
                    move = abs(ppt - cusp)
                    cand = CuspRefinementCandidate(
                        component_id=f"blob{blob_uid}", path_index=pi, subpath_index=si,
                        segment_index=gi, error_blob_id=blob_uid,
                        source_cusp_point=(round(cusp.real, 2), round(cusp.imag, 2)),
                        current_curve_point=(round(ppt.real, 2), round(ppt.imag, 2)),
                        local_curvature=round(_seg_curvature(subs[si][gi], t), 4),
                        opening_angle=0.0, edge_separation=round(edge_sep, 2),
                        max_error=round(max_err, 2), p95_error=round(p95, 2),
                        confidence=round(max(0.0, 1.0 - move / 20.0), 3),
                    )
                    # derin cusp: taşıma sınırı AŞILIRSA reddetme, sınıra KIRP —
                    # sonraki geçiş kalan blobu yeniden işler (kademeli erişim,
                    # geçiş başına ≤10px; bütçe toplam erişimi sınırlar)
                    target = cusp
                    if move > _MAX_ANCHOR_MOVE:
                        target = ppt + (cusp - ppt) * (_MAX_ANCHOR_MOVE / move)
                    if move < 0.3:
                        cand.rejection_reason = "eğri zaten cusp üzerinde"
                    elif subs[si][gi].length() < _MIN_SEG_LEN and 0.12 <= t <= 0.88:
                        cand.rejection_reason = "segment bölünemeyecek kadar kısa"
                    elif not _split_and_place(subs, si, gi, t, target):
                        cand.rejection_reason = "desteklenmeyen segment türü (yay)"
                    else:
                        cand.accepted = True
                        blob_added += 1
                        cusp_physical += 1
                        blob_changed.add(pi)
                        node_refs["a" if cls_want == cls_a else "b"].append((pi, si, gi))
                    candidates.append(cand)
                if cusp_physical > 0:
                    if existing is not None:
                        # mevcut mantıksal çapa yerinde ilerledi: bütçe düşmez,
                        # fiziksel komut (varsa) ayrıca sayılır
                        existing.physical_insertions += cusp_physical
                        existing.error_blob_ids.add(blob_uid)
                    else:
                        blob_logical += 1
                        new_logicals.append(LogicalBoundaryAnchor(
                            logical_id=f"la_{len(logical_anchors)+len(new_logicals)}",
                            source_point=(round(cusp.real, 2), round(cusp.imag, 2)),
                            physical_insertions=cusp_physical,
                            error_blob_ids={blob_uid}, iteration_created=_pass,
                        ))
        if not blob_changed:
            continue
        md0 = _crop_max_dev(sc, rc)
        # bölme sonrası BÖLGE-KAPSAMLI geniş snap: yeni çapalar snap'e
        # serbestlik verir; bölge dışı çapalar yerinde kalır (aksi hâlde
        # büyük path'lerde bölge dışı render değişimi seam korumasına
        # takılıyordu — ölçüldü). Maks-sapma koruması (md0/md1) snap'in
        # dar kamada yaptığı zararı kapıda yakalar.
        snap_region = (x0 - 20.0, y0 - 20.0, x1 + 20.0, y1 + 20.0)
        for pi in blob_changed:
            path_els[pi].set("d", serialize(pi))
            sp_list = _parse_subpaths_arc(path_els[pi].get("d"))
            if sp_list is None:
                continue
            moved_any = 0
            for _snap_pass in range(2):
                for sp in sp_list:
                    moved_any += _snap_subpath_wide(sp, ref_f32, region=snap_region)
            if moved_any:
                path_els[pi].set("d", _serialize_subpaths_arc(sp_list))
        tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
        render_budget -= 1
        after = render_fn(tmp, width, height)
        ok = False
        err_new = best_err
        if after is not None:
            after_cls = classify_rgb(after, fills_rgb)
            e1 = (src_cls != after_cls).astype(np.uint8)
            e1 = cv2.morphologyEx(e1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            err_new = int(e1.sum())
            out_new = int((e1 & (region_mask == 0)).sum())
            blob_err1 = int((sc != after_cls[y0:y1, x0:x1]).sum())
            txt = serialize_all_check(path_els)
            cmds = len(re.findall(r"[MLCQAZHVSTmlcqazhvst]", txt))
            import os as _os  # noqa: PLC0415

            dbg = _os.environ.get("VEKTORYUM_CUSP_DEBUG")
            if dbg:
                md1_dbg = _crop_max_dev(sc, after_cls[y0:y1, x0:x1])
                logger.warning(
                    "cusp blob%d kapı: blob_err %d->%d (eşik %.0f) toplam %d->%d "
                    "dışarı %d->%d md %.2f->%.2f cmds %d",
                    blob_uid, blob_err0, blob_err1, blob_err0 * 0.85,
                    best_err, err_new, out_before, out_new, md0, md1_dbg, cmds)
            # dışarı koruması ORANTILI: bölünen uzun segmentin uzak ucundaki
            # kılcal eğri etkisi (seam değil) meşrudur; dışarı artışı blob
            # kazancının %25'ini aşarsa reddedilir (ölçüldü: 498px kazanca
            # karşı 108px kılcal etki — mutlak eşik 8px bunu reddediyordu)
            out_budget = max(8, int(0.25 * max(0, blob_err0 - blob_err1)))
            md1 = _crop_max_dev(sc, after_cls[y0:y1, x0:x1]) if after is not None else md0
            if "NaN" in txt or "Infinity" in txt or "nan" in txt.lower():
                pass  # reddet
            elif cmds > 900:
                pass
            elif out_new > out_before + out_budget:
                pass  # bölge dışına orantısız hata taştı (seam koruması)
            elif (blob_err1 <= blob_err0 * 0.85 or md1 <= md0 * 0.8) and err_new <= best_err:
                # şartname kabulü: blob alanı >=%15 VEYA maks hata >=%20
                # azalmalı; toplam artmamalı. Ek koruma: maks sınır sapması
                # derinleşmemeli (piksel düşerken sahte kenar açılması ölçüldü)
                ok = md1 <= md0 + 0.5
        if ok:
            accepted_blobs += 1
            pass_accepted += 1
            total_added += blob_logical          # MANTIKSAL bütçe düşer
            total_physical += blob_added
            budget_rec[4] += blob_logical
            logical_anchors.extend(new_logicals)  # kabul edilen birimler kalıcı
            best_err = err_new
            cur = after
            rnd_cls = classify_rgb(cur, fills_rgb)
            if node_refs["a"] and node_refs["b"] and cusp0 is not None:
                boundaries.append(CanonicalBoundary(
                    boundary_id=f"cb_{bi}",
                    region_a_id=uniq_fills[cls_a], region_b_id=uniq_fills[cls_b],
                    source_points=[(round(cusp0.real, 2), round(cusp0.imag, 2))],
                    path_a_refs=node_refs["a"], path_b_refs=node_refs["b"],
                    start_node=(round(cusp0.real, 2), round(cusp0.imag, 2)),
                    end_node=(round(cusp0.real, 2), round(cusp0.imag, 2)),
                    confidence=1.0,
                ))
        else:
            failed_sigs.add((bx, by, bw, bh))
            for pi in blob_changed:
                path_els[pi].set("d", backup_d[pi])
            for pi in parsed_backup:
                parsed[pi] = parsed_backup[pi]
            for c in candidates:
                if c.accepted and c.error_blob_id == blob_uid:
                    c.accepted = False
                    c.rejection_reason = "blob kapıyı geçemedi"
      if pass_accepted == 0:
          break
    rounds_done = accepted_blobs

    tmp.unlink(missing_ok=True)
    if rounds_done:
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        status = "completed"
    else:
        status = "no_change"
    return {
        "status": status, "rounds": rounds_done,
        "anchors_added": total_added,          # MANTIKSAL çapa sayısı
        "physical_anchors": total_physical,    # gerçek fiziksel bölme sayısı
        "logical_anchors": len(logical_anchors),
        "renders_used": _MAX_RENDER_BUDGET - render_budget,
        "err_px_before": err_before, "err_px_after": best_err,
        "candidates": [asdict(c) for c in candidates],
        "logical_anchor_list": [
            {"logical_id": la.logical_id, "source_point": la.source_point,
             "physical_insertions": la.physical_insertions,
             "iteration_created": la.iteration_created}
            for la in logical_anchors
        ],
        "canonical_boundaries": [asdict(b) for b in boundaries],
    }


def serialize_all_check(path_els) -> str:
    return " ".join(el.get("d") or "" for el in path_els)


def _copy_seg(seg):
    import copy  # noqa: PLC0415

    return copy.copy(seg)


def _crop_max_dev(sc: np.ndarray, rc: np.ndarray) -> float:
    """Kırpım içinde sınıf sınırları arasındaki simetrik maks sapma (px)."""
    def edges(c: np.ndarray) -> np.ndarray:
        e = np.zeros(c.shape, np.uint8)
        e[:-1, :] |= (c[:-1, :] != c[1:, :]).astype(np.uint8)
        e[:, :-1] |= (c[:, :-1] != c[:, 1:]).astype(np.uint8)
        return e
    es, er = edges(sc), edges(rc)
    if not es.any() or not er.any():
        return 0.0
    dt_r = cv2.distanceTransform((er == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_s = cv2.distanceTransform((es == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return max(float(dt_r[es > 0].max()), float(dt_s[er > 0].max()))


def _deep_points(dt: np.ndarray, k: int, nms_radius: float) -> list[tuple[int, int]]:
    """dt'nin yerel maksimumları: en derin k nokta, NMS ile ayrık (determinist)."""
    pts: list[tuple[int, int]] = []
    work = dt.copy()
    for _ in range(k):
        idx = int(np.argmax(work))
        v = float(work.flat[idx])
        if v <= 1.0:
            break
        y, x = divmod(idx, work.shape[1])
        pts.append((int(x), int(y)))
        yy, xx = np.ogrid[:work.shape[0], :work.shape[1]]
        work[(yy - y) ** 2 + (xx - x) ** 2 <= nms_radius ** 2] = 0
    return pts


def _max_dev_in_blobs(src_cls: np.ndarray, rnd_cls: np.ndarray,
                      blobs: list[tuple[int, int, int, int, int]]) -> float:
    """Blob kutuları içinde maks sınır sapması (dt tabanlı, px)."""
    worst = 0.0
    for _a, x, y, w2, h2 in blobs:
        x0, y0 = max(0, x - _PAD), max(0, y - _PAD)
        x1, y1 = min(src_cls.shape[1], x + w2 + _PAD), min(src_cls.shape[0], y + h2 + _PAD)
        sc, rc = src_cls[y0:y1, x0:x1], rnd_cls[y0:y1, x0:x1]
        bad = sc != rc
        if not bad.any():
            continue
        for cls_v in np.unique(sc[bad]):
            miss = bad & (sc == cls_v)
            not_v = (rc != cls_v).astype(np.uint8)
            dt = cv2.distanceTransform(not_v, cv2.DIST_L2, 5)
            dt[~miss] = 0
            worst = max(worst, float(dt.max()))
    return worst
