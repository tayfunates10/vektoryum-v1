"""HG-3 — Alt-piksel canonical boundary curve fitting (SHADOW).

Crack-edge kafesindeki eksen-hizalı staircase polyline'ları, kaynak renk
sınırına oturan DÜŞÜK KOMUTLU kübil Bézier eğrilerine dönüştürür. Her fiziksel
boundary YALNIZ BİR KEZ fit edilir; twin half-edge'ler aynı fitted segment
listesini ters yönde kullanır (yeniden fit YOK).

Kurallar (şartname):
- Junction/exterior endpoint koordinatı KİLİTLİ (shared vertex, hareket etmez).
- Alt-piksel örnekleme kaynak RGB üzerinden (crack normali boyunca renk geçişi).
- Düşük gradyan / belirsiz sample fitting'i zorlamaz (fallback: ham polyline).
- Straight boundary → tek line; aksi halde adaptif Schneider kübik.
- Determinist: aynı graph + kaynak → aynı segmentler.

Schneider en küçük kareler kübik uydurma ``curve_refit`` modülünden yeniden
kullanılır (endpoint'leri tam korur).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.curve_refit import _fit_cubic, _unit
from app.half_edge_graph import SharedBoundaryHalfEdgeGraph

Point = tuple[float, float]


@dataclass
class BezierSegment:
    p0: Point
    c1: Point
    c2: Point
    p1: Point
    is_line: bool = False

    def reversed(self) -> "BezierSegment":
        return BezierSegment(self.p1, self.c2, self.c1, self.p0, self.is_line)


# ---------------------------------------------------------------------------
# Alt-piksel örnekleme — crack normali boyunca renk geçişi
# ---------------------------------------------------------------------------
def _bilinear_rgb(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Kaynak RGB'yi (H,W,3) sürekli piksel-merkez koordinatında örnekler."""
    h, w = img.shape[:2]
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    a = img[y0, x0].astype(np.float32)
    b = img[y0, x1].astype(np.float32)
    c = img[y1, x0].astype(np.float32)
    d = img[y1, x1].astype(np.float32)
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _subpixel_offset(source_rgb: np.ndarray, px: float, py: float,
                     nx: float, ny: float, cL: np.ndarray, cR: np.ndarray,
                     reach: float = 1.5) -> tuple[float, float]:
    """Crack noktasından normal boyunca renk sınırı offset'i. (offset, güven).

    Membership m(s) = |sample-cR| - |sample-cL|; sıfır geçişi = sınır. Kaynak
    korner-uzayında; piksel-merkez örneklemesi için +0.5 kaydır. Güven, iki renk
    ayrımının netliğidir (düşükse fitting zorlanmaz).
    """
    samples = np.linspace(-reach, reach, 7)
    ms = []
    for s in samples:
        # korner-uzay -> piksel-merkez: pikselin merkezi (i+0.5) korner-uzayda i+... ;
        # crack noktası korner koordinatı, örnek piksel merkezine 0.5 kayar
        c = _bilinear_rgb(source_rgb, px + s * nx - 0.5, py + s * ny - 0.5)
        dL = float(np.linalg.norm(c - cL))
        dR = float(np.linalg.norm(c - cR))
        ms.append(dR - dL)   # >0 → cL'ye yakın
    ms = np.array(ms)
    # sıfır geçişini bul (işaret değişimi): -reach tarafı cR (negatif), +reach cL
    conf = float(min(np.linalg.norm(cL - cR) / 40.0, 1.0))  # renk ayrımı gücü
    for i in range(len(ms) - 1):
        if ms[i] == 0:
            return float(samples[i]), conf
        if ms[i] * ms[i + 1] < 0:
            t = ms[i] / (ms[i] - ms[i + 1])
            off = float(samples[i] + t * (samples[i + 1] - samples[i]))
            return off, conf
    return 0.0, 0.0   # geçiş yok → güven 0


# ---------------------------------------------------------------------------
# Straight / primitive tespiti
# ---------------------------------------------------------------------------
def _detect_corners(pts: list[Point], win: int = 4, deg: float = 50.0,
                    closed: bool = False) -> list[int]:
    """Pencereli dönüş açısıyla GERÇEK köşe indekslerini bulur (staircase gürültüsü
    pencere üzerinde ortalanıp elenir). ``closed`` ise sarmalı. İç köşe indeksleri."""
    from app.curve_refit import _win_turn_signed
    n = len(pts)
    if n < 2 * win + 1:
        return []
    w = min(win, max(2, n // 6))
    rng = range(n) if closed else range(w, n - w)
    cand: list[tuple[float, int]] = []
    for i in rng:
        t = _win_turn_signed(pts, i, n, closed, w)
        if t >= deg:
            cand.append((t, i))
    cand.sort(reverse=True)
    chosen: list[int] = []
    for _t, i in cand:
        if all(abs(i - j) > w and abs(i - j) < n - w for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def _split_runs(pts: list[Point], closed: bool = False) -> list[list[Point]]:
    """Polyline'ı gerçek köşelerde koşulara böler (her koşu köşede uçlanır).

    ``closed`` (start==end) ise köşeler sarmalı bulunur; polyline ilk köşeye
    döndürülür ve koşular seam'i köşeden geçirir (yalancı-vertex ortada kalmaz).
    """
    if closed and len(pts) > 3:
        loop = pts[:-1]                       # kapanış tekrarını at
        n = len(loop)
        corners = _detect_corners(loop, closed=True)
        # kapalı loop koşuları UÇLARI FARKLI olmalı (aksi halde Schneider chord
        # parametrelemesi çöker). ≥2 köşe → köşeler; aksi halde deterministik
        # çeyrek bölme (köşesiz daire de düzgün fit olsun).
        if len(corners) >= 2:
            breaks = sorted(corners)
        else:
            base = corners[0] if corners else 0
            breaks = sorted({(base + n * i // 4) % n for i in range(4)})
        runs: list[list[Point]] = []
        for a, b in zip(breaks, breaks[1:] + [breaks[0] + n]):
            runs.append([loop[i % n] for i in range(a, b + 1)])
        return [r for r in runs if len(r) >= 2]
    corners = _detect_corners(pts, closed=False)
    if not corners:
        return [pts]
    runs = []
    prev = 0
    for c in corners:
        runs.append(pts[prev:c + 1])
        prev = c
    runs.append(pts[prev:])
    return [r for r in runs if len(r) >= 2]


def _win_tangent(run: list[Point], end: int, w: int = 4) -> Point:
    """Koşu ucunda PENCERELİ teğet yönü (staircase gürültüsünü ortalar).

    end=0 → başlangıç (koşu içine); end=-1 → bitiş (koşu içine)."""
    n = len(run)
    k = min(w, n - 1)
    if k < 1:
        return (0.0, 0.0)
    if end == 0:
        return _unit(run[0], run[k])
    return _unit(run[-1], run[-1 - k])


def _max_dev_from_chord(pts: list[Point]) -> float:
    if len(pts) < 3:
        return 0.0
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return max(math.hypot(p[0] - x0, p[1] - y0) for p in pts)
    dmax = 0.0
    for x, y in pts[1:-1]:
        d = abs((x - x0) * dy - (y - y0) * dx) / L
        dmax = max(dmax, d)
    return dmax


# ---------------------------------------------------------------------------
# Ana fit
# ---------------------------------------------------------------------------
def fit_canonical_curves(graph: SharedBoundaryHalfEdgeGraph,
                         source_rgb: np.ndarray | None,
                         fills_rgb: np.ndarray | None,
                         tol_scale: float = 1.0) -> dict[str, Any]:
    """Graph'taki her canonical curve'ü alt-piksel Bézier'lere fit eder.

    ``curve.fitted_segments`` (BezierSegment listesi), fit_error_max/p95,
    command_count, primitive_kind, fit_fallback DOLDURULUR. Twin half-edge
    aynı curve nesnesine baktığından yeniden fit edilmez. İstatistik döner.
    """
    h, w = (source_rgb.shape[:2] if source_rgb is not None else (0, 0))
    diag = float(math.hypot(h, w)) if h else 1000.0
    # ölçek-normalize tolerans: crack staircase gürültü tabanı ~0.5px olduğundan
    # taban ~1.2px (aksi halde eğri wiggle'ı kovalayıp komut patlar); büyük
    # görselde orantılı gevşer.
    base_tol = max(1.2, diag / 1400.0) * tol_scale
    straight_tol = 0.7 * tol_scale

    fills = fills_rgb.astype(np.float32) if fills_rgb is not None else None
    errs_all: list[float] = []
    n_line = n_cubic = n_fallback = total_segs = 0

    for cid in sorted(graph.curves):
        cur = graph.curves[cid]
        poly = [(float(x), float(y)) for x, y in cur.polyline]
        if len(poly) < 2:
            cur.fitted_segments = []
            continue
        vs = graph.vertices[cur.start_vertex_id].point
        ve = graph.vertices[cur.end_vertex_id].point
        # endpoint'leri shared vertex'e KİLİTLE
        poly[0] = (float(vs[0]), float(vs[1]))
        poly[-1] = (float(ve[0]), float(ve[1]))

        # --- alt-piksel refine (iç noktalar; endpoint kilitli) ---
        fallback = False
        refined = list(poly)
        fa, fb = cur.adjacent_face_ids
        can_refine = (source_rgb is not None and fills is not None
                      and not cur.is_exterior and fa is not None and fb is not None)
        if can_refine:
            cLid = graph.faces[fa].color_id
            cRid = graph.faces[fb].color_id
            if 0 <= cLid < len(fills) and 0 <= cRid < len(fills):
                cL, cR = fills[cLid], fills[cRid]
                low_conf = 0
                for i in range(1, len(poly) - 1):
                    px, py = poly[i]
                    tx, ty = _unit(poly[i - 1], poly[i + 1])
                    nx, ny = -ty, tx           # normal
                    off, conf = _subpixel_offset(source_rgb, px, py, nx, ny, cL, cR)
                    if conf < 0.25:
                        low_conf += 1
                        continue
                    refined[i] = (px + off * nx, py + off * ny)
                if len(poly) > 2 and low_conf > 0.6 * (len(poly) - 2):
                    fallback = True            # sinyal çoğunlukla belirsiz
                    refined = list(poly)

        # --- fit: gerçek köşelerde böl, her koşuyu line ya da kübik uydur ---
        segs: list[BezierSegment] = []
        is_closed = cur.start_vertex_id == cur.end_vertex_id
        runs = _split_runs(refined, closed=is_closed)
        any_cubic = False
        for run in runs:
            if _max_dev_from_chord(run) <= straight_tol:
                p0, p1 = run[0], run[-1]
                c1 = (p0[0] + (p1[0] - p0[0]) / 3.0, p0[1] + (p1[1] - p0[1]) / 3.0)
                c2 = (p0[0] + 2 * (p1[0] - p0[0]) / 3.0, p0[1] + 2 * (p1[1] - p0[1]) / 3.0)
                segs.append(BezierSegment(p0, c1, c2, p1, is_line=True))
            else:
                # uç teğetleri PENCERE üzerinden (tek staircase adımı eksen-hizalı
                # olur, gerçek eğri yönünü vermez → fit uçta bozulur)
                t1 = _win_tangent(run, 0)
                t2 = _win_tangent(run, -1)
                out: list = []
                _fit_cubic(run, t1, t2, base_tol, out)
                for bez in out:
                    segs.append(BezierSegment(tuple(bez[0]), tuple(bez[1]),
                                              tuple(bez[2]), tuple(bez[3])))
                any_cubic = True
        cur.primitive_kind = "cubic" if any_cubic else "line"
        if any_cubic:
            n_cubic += 1
        else:
            n_line += 1

        # --- hata metriği (fit vs refined sample noktaları) ---
        emax, e95 = _fit_error(segs, refined)
        cur.fitted_segments = segs
        cur.fit_error_max = emax
        cur.fit_error_p95 = e95
        cur.command_count = len(segs)
        cur.fit_fallback = fallback
        if fallback:
            n_fallback += 1
        total_segs += len(segs)
        errs_all.append(emax)

    errs = np.array(errs_all) if errs_all else np.array([0.0])
    return {
        "curves": len(graph.curves),
        "segments": total_segs,
        "line_curves": n_line,
        "cubic_curves": n_cubic,
        "fallback_curves": n_fallback,
        "fit_error_max": float(errs.max()),
        "fit_error_p95": float(np.percentile(errs, 95)),
        "fit_error_mean": float(errs.mean()),
        "avg_segments_per_curve": round(total_segs / max(1, len(graph.curves)), 3),
    }


def _fit_error(segs: list[BezierSegment], pts: list[Point]) -> tuple[float, float]:
    """Fit edilmiş segmentlerden örnek noktalarına maks/p95 mesafe (vektörize).

    Örnek noktaları ≤256'ya alt-örneklenir (uzun curve'de maliyet sınırlı) ve
    segmentler ~1px aralıkla örneklenir; mesafe broadcast ile tek seferde. Sabit
    12 örnek uzun segmentte sahte granülerlik "hatası" verirdi.
    """
    if not segs or len(pts) < 3:
        return 0.0, 0.0
    dense: list[Point] = []
    for s in segs:
        bez = [s.p0, s.c1, s.c2, s.p1]
        clen = math.hypot(bez[3][0] - bez[0][0], bez[3][1] - bez[0][1])
        m = min(256, max(12, int(round(clen))))
        ts = np.linspace(0.0, 1.0, m + 1)
        mt = 1 - ts
        bx = (mt ** 3 * bez[0][0] + 3 * mt * mt * ts * bez[1][0]
              + 3 * mt * ts * ts * bez[2][0] + ts ** 3 * bez[3][0])
        by = (mt ** 3 * bez[0][1] + 3 * mt * mt * ts * bez[1][1]
              + 3 * mt * ts * ts * bez[2][1] + ts ** 3 * bez[3][1])
        dense.extend(zip(bx.tolist(), by.tolist()))
    da = np.asarray(dense, dtype=np.float64)
    ptn = np.asarray(pts, dtype=np.float64)
    if len(ptn) > 256:                       # maliyet sınırı: düzgün alt-örnekle
        idx = np.linspace(0, len(ptn) - 1, 256).astype(int)
        ptn = ptn[idx]
    # broadcast: her örnek noktasının en yakın dense mesafesi
    dx = ptn[:, None, 0] - da[None, :, 0]
    dy = ptn[:, None, 1] - da[None, :, 1]
    dmin = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return float(dmin.max()), float(np.percentile(dmin, 95))
