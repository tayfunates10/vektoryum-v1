"""Geometrik şekil oturtma (shape / arc fitting) katmanı.

Trace çıktısı bezier eğrileri kaynağı sadık izler ama matematiksel olarak tam
simetrik değildir. Bu katman, düz çizgi + dairesel yay primitiflerine oturtarak
çıktıyı idealize eder:

* Yoğun polyline'a açılır.
* Eğrilik (curvature) profiline göre DÜZ ve YAY bölgelerine ayrılır; keskin
  köşeler ayrı kırılma noktasıdır.
* Düz bölgeler tam çizgiye (``L``), yay bölgeleri tam dairesel yaya (``A r r``)
  oturtulur -> simetrik oval kenarlar + keskin köşeler.
* Bir path güvenle oturtulamazsa (yüksek artık hata veya bbox sapması) orijinal
  ``d`` aynen korunur; çıktı asla bozulmaz.

Yalnız düz/geometrik logo profillerinde kullanılır (renkli/foto modlarında değil).
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from svgpathtools import parse_path
except ImportError:  # pragma: no cover
    parse_path = None

logger = logging.getLogger(__name__)
SVG_NS = "http://www.w3.org/2000/svg"


def _fmt(v: float) -> str:
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


# ---------------------------------------------------------------------------
# Geometri yardımcıları
# ---------------------------------------------------------------------------
def _flatten_subpaths(d: str, max_pts: int = 700) -> list[tuple[np.ndarray, bool, str]]:
    path = parse_path(d)
    out: list[tuple[np.ndarray, bool, str]] = []
    for sub in path.continuous_subpaths():
        try:
            sub_d = sub.d()
        except Exception:  # noqa: BLE001
            sub_d = ""
        try:
            length = sub.length()
        except Exception:  # noqa: BLE001
            length = 0.0
        n = int(min(max_pts, max(12, (length or 12) / 2.0)))
        pts = []
        for i in range(n + 1):
            try:
                p = sub.point(i / n)
            except Exception:  # noqa: BLE001
                continue
            pts.append((p.real, p.imag))
        if len(pts) < 4:
            continue
        arr = np.array(pts, dtype=float)
        # ardışık çok yakın noktaları temizle
        keep = [0]
        for i in range(1, len(arr)):
            if np.hypot(*(arr[i] - arr[keep[-1]])) > 0.4:
                keep.append(i)
        arr = arr[keep]
        closed = bool(np.hypot(*(sub.start - sub.end if False else (arr[0] - arr[-1]))) < 1.5)
        if closed and len(arr) > 1:
            arr = arr[:-1]  # son tekrar noktayı at (kapalı)
        if len(arr) < 4:
            continue
        out.append((arr, closed, sub_d))
    return out


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float] | None:
    x = pts[:, 0]
    y = pts[:, 1]
    a_mat = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b_vec = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 1e-6:
        return None
    return (float(cx), float(cy), float(math.sqrt(r2)))


def _line_residual(pts: np.ndarray) -> float:
    a = pts[0]
    b = pts[-1]
    d = b - a
    L = math.hypot(d[0], d[1])
    if L < 1e-9:
        return float(np.max(np.hypot(pts[:, 0] - a[0], pts[:, 1] - a[1])))
    nrm = np.array([-d[1], d[0]]) / L
    return float(np.max(np.abs((pts - a) @ nrm)))


def _circle_residual(pts: np.ndarray, circ: tuple[float, float, float]) -> float:
    cx, cy, r = circ
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return float(np.max(np.abs(dist - r)))


def _curvature(pts: np.ndarray, closed: bool, w: int) -> np.ndarray:
    n = len(pts)
    k = np.zeros(n)
    for i in range(n):
        if closed:
            ia, ic = (i - w) % n, (i + w) % n
        else:
            ia, ic = max(0, i - w), min(n - 1, i + w)
        tri = pts[[ia, i, ic]]
        circ = _fit_circle(tri)
        k[i] = 1.0 / circ[2] if circ and circ[2] > 1e-6 else 0.0
    return k


def _turn(pts: np.ndarray, i: int, n: int, closed: bool, w: int) -> float:
    if closed:
        a, b, c = pts[(i - w) % n], pts[i], pts[(i + w) % n]
    else:
        if i - w < 0 or i + w >= n:
            return 0.0
        a, b, c = pts[i - w], pts[i], pts[i + w]
    v1 = b - a
    v2 = c - b
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = max(-1.0, min(1.0, float((v1 @ v2) / (n1 * n2))))
    return math.degrees(math.acos(cosv))


# ---------------------------------------------------------------------------
# Segmentasyon + oturtma
# ---------------------------------------------------------------------------
def _segment_indices(pts: np.ndarray, closed: bool, diag: float) -> tuple[list[int], np.ndarray]:
    """Kırılma noktalarını (köşeler + düz/yay sınıf değişimi) ve yay-maskesini döndürür."""
    n = len(pts)
    w = max(2, n // 80)
    curv = _curvature(pts, closed, w)
    k_line = 1.0 / max(diag * 0.85, 1.0)  # yarıçap > ~0.85*diag ise düz say
    is_arc = curv > k_line

    # köşeler: küçük pencere üzerinde keskin dönüş
    cw = max(2, n // 120)
    corner = np.zeros(n, dtype=bool)
    rng = range(n) if closed else range(cw, n - cw)
    for i in rng:
        if _turn(pts, i, n, closed, cw) > 33.0:
            corner[i] = True

    # köşeleri non-max bastır
    corner_idx = [i for i in range(n) if corner[i]]

    breaks: set[int] = set(corner_idx)
    # sınıf (düz/yay) değişim noktaları
    for i in range(1, n):
        if is_arc[i] != is_arc[i - 1]:
            breaks.add(i)
    if closed and is_arc[0] != is_arc[-1]:
        breaks.add(0)
    if not closed:
        breaks.add(0)
        breaks.add(n - 1)
    return sorted(breaks), is_arc


def _runs_from_breaks(n: int, breaks: list[int], closed: bool) -> list[list[int]]:
    if not closed:
        runs = []
        for s, e in zip(breaks[:-1], breaks[1:]):
            runs.append(list(range(s, e + 1)))
        return runs
    if not breaks:
        breaks = [0]
    runs = []
    bs = breaks + [breaks[0] + n]
    for s, e in zip(bs[:-1], bs[1:]):
        runs.append([(k % n) for k in range(s, e + 1)])
    return runs


def _arc_flags(p0: np.ndarray, pmid: np.ndarray, p1: np.ndarray, circ: tuple[float, float, float]) -> tuple[int, int]:
    cx, cy, _ = circ

    def ang(p: np.ndarray) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)

    def norm(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi

    a0 = ang(p0)
    am = ang(pmid)
    a1 = ang(p1)
    total = norm(am - a0) + norm(a1 - am)
    sweep = 1 if total > 0 else 0
    large = 1 if abs(total) > math.pi else 0
    return large, sweep


def _fit_run(pts: np.ndarray, prefer_arc: bool, line_tol: float, arc_tol: float, depth: int = 0):
    """Bir run'ı line/arc primitiflerine oturtur. Döner: list[(type,p0,p1,circ,pmid)]."""
    if len(pts) < 3 or depth > 60:
        return [("line", pts[0], pts[-1], None, None)]

    le = _line_residual(pts)
    circ = _fit_circle(pts)
    ce = _circle_residual(pts, circ) if circ else float("inf")
    span_ok = circ is not None and circ[2] < 1e5
    line_ok = le <= line_tol
    arc_ok = span_ok and ce <= arc_tol

    # Eğrilik sınıflandırmasına saygı: run yay ise önce yay dene; düz ise önce çizgi.
    if prefer_arc and arc_ok:
        return [("arc", pts[0], pts[-1], circ, pts[len(pts) // 2])]
    if line_ok:
        return [("line", pts[0], pts[-1], None, None)]
    if arc_ok:
        return [("arc", pts[0], pts[-1], circ, pts[len(pts) // 2])]

    # ikisi de yetmedi -> en sapan noktadan böl
    a = pts[0]
    b = pts[-1]
    d = b - a
    L = math.hypot(d[0], d[1])
    if L < 1e-9:
        return [("line", pts[0], pts[-1], None, None)]
    nrm = np.array([-d[1], d[0]]) / L
    dev = np.abs((pts - a) @ nrm)
    k = int(np.argmax(dev))
    if k <= 0 or k >= len(pts) - 1:
        return [("line", pts[0], pts[-1], None, None)]
    return (
        _fit_run(pts[: k + 1], prefer_arc, line_tol, arc_tol, depth + 1)
        + _fit_run(pts[k:], prefer_arc, line_tol, arc_tol, depth + 1)
    )


def _seg_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py])


def _line_dir_deg(p0, p1) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0


def _postprocess_prims(prims: list, closed: bool, collinear_tol: float = 5.0) -> list:
    """Sığ yayları düz çizgiye çevirir, ardışık aynı-yön çizgileri birleştirir,
    çizgi-çizgi köşelerini tam kesişime oturtur (keskin + tam düz)."""
    if len(prims) < 1:
        return prims

    # 0) SIĞ yay (chord'dan sapması < ~0.8px) = pratikte düz -> çizgiye çevir
    flattened: list = []
    for typ, p0, p1, circ, pmid in prims:
        if typ == "arc" and circ:
            chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            r = circ[2]
            half = chord / 2.0
            sag = (r - math.sqrt(max(0.0, r * r - half * half))) if r > half else r
            if sag < 0.8:
                flattened.append(("line", p0, p1, None, None))
                continue
        flattened.append((typ, p0, p1, circ, pmid))
    prims = flattened
    if len(prims) < 2:
        return prims

    # 1) ardışık collinear çizgileri birleştir
    merged: list = []
    for prim in prims:
        if merged and prim[0] == "line" and merged[-1][0] == "line":
            d1 = _line_dir_deg(merged[-1][1], merged[-1][2])
            d2 = _line_dir_deg(prim[1], prim[2])
            diff = abs(d1 - d2)
            diff = min(diff, 180.0 - diff)
            if diff <= collinear_tol:
                merged[-1] = ("line", merged[-1][1], prim[2], None, None)
                continue
        merged.append(list(prim) if False else prim)

    # kapalıysa baş/son collinear çizgi birleşimi
    if closed and len(merged) >= 2 and merged[0][0] == "line" and merged[-1][0] == "line":
        d1 = _line_dir_deg(merged[-1][1], merged[-1][2])
        d2 = _line_dir_deg(merged[0][1], merged[0][2])
        diff = abs(d1 - d2)
        diff = min(diff, 180.0 - diff)
        if diff <= collinear_tol:
            first = merged.pop(0)
            merged[-1] = ("line", merged[-1][1], first[2], None, None)

    # 2) çizgi-çizgi köşelerini tam kesişime taşı (keskin uç)
    prims2 = [list(p) for p in merged]
    n = len(prims2)
    pairs = list(range(n - 1)) + ([n - 1] if closed and n > 1 else [])
    for i in pairs:
        j = (i + 1) % n
        A = prims2[i]
        B = prims2[j]
        if A[0] != "line" or B[0] != "line":
            continue
        d1 = _line_dir_deg(A[1], A[2])
        d2 = _line_dir_deg(B[1], B[2])
        ang = abs(d1 - d2)
        ang = min(ang, 180.0 - ang)
        if ang < 8.0:  # neredeyse düz -> köşe değil
            continue
        inter = _seg_intersection(A[1], A[2], B[1], B[2])
        if inter is None:
            continue
        # kesişim mevcut köşeye makul yakınsa uygula (taşma/uzama engeli)
        seg_len = max(np.hypot(*(A[2] - A[1])), np.hypot(*(B[2] - B[1])))
        if np.hypot(*(inter - A[2])) <= 0.5 * seg_len:
            A[2] = inter
            B[1] = inter
    return [tuple(p) for p in prims2]


def _primitives_to_d(prims: list, closed: bool) -> str:
    if not prims:
        return ""
    start = prims[0][1]
    parts = [f"M {_fmt(start[0])} {_fmt(start[1])}"]
    for typ, p0, p1, circ, pmid in prims:
        if typ == "line":
            parts.append(f"L {_fmt(p1[0])} {_fmt(p1[1])}")
        else:
            large, sweep = _arc_flags(p0, pmid, p1, circ)
            parts.append(f"A {_fmt(circ[2])} {_fmt(circ[2])} 0 {large} {sweep} {_fmt(p1[0])} {_fmt(p1[1])}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


# ---------------------------------------------------------------------------
# BÜTÜNSEL şekil oturtma (whole shape fitting)
# ---------------------------------------------------------------------------
def _signed_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def _sample_d(d: str, max_pts: int = 420) -> np.ndarray | None:
    """Bir d-string'i yoğun nokta dizisine örnekler (doğrulama için)."""
    if parse_path is None:
        return None
    try:
        samples: list[tuple[float, float]] = []
        for s in parse_path(d).continuous_subpaths():
            try:
                L = s.length()
            except Exception:  # noqa: BLE001
                L = 60.0
            m = int(min(max_pts, max(32, (L or 32) / 1.5)))
            for i in range(m + 1):
                p = s.point(i / m)
                samples.append((p.real, p.imag))
        return np.array(samples, dtype=float) if len(samples) >= 8 else None
    except Exception:  # noqa: BLE001
        return None


def _max_dist_to_polyline(points: np.ndarray, poly: np.ndarray) -> float:
    """Nokta kümesinin KAPALI polyline'a maksimum uzaklığı (nokta-doğru parçası).

    Nokta-noktaya Hausdorff, örnekleme yoğunluğuna bağlı sahte sapma üretir
    (aralık/2); doğru parçası mesafesi ayrıklaştırmadan bağımsızdır.
    """
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ab_len2 = (ab * ab).sum(axis=1)
    ab_len2 = np.where(ab_len2 < 1e-12, 1e-12, ab_len2)
    # (N nokta, M segment) proje + kelepçele
    ap = points[:, None, :] - a[None, :, :]
    t = (ap * ab[None, :, :]).sum(axis=2) / ab_len2[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d2 = ((points[:, None, :] - closest) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).max())


def _bidirectional_dev(orig: np.ndarray, shape: np.ndarray) -> float:
    """Çift yönlü maksimum sapma: hem izin dışına taşma hem de şeklin
    izlenmeyen bölgesi (ör. L-poligonun minAreaRect'e oturtulması) yakalanır."""
    o = orig if len(orig) <= 220 else orig[np.linspace(0, len(orig) - 1, 220).astype(int)]
    s = shape if len(shape) <= 420 else shape[np.linspace(0, len(shape) - 1, 420).astype(int)]
    fwd = _max_dist_to_polyline(o, s)   # orijinal -> şekil
    bwd = _max_dist_to_polyline(s, o)   # şekil -> orijinal
    return max(fwd, bwd)


def _rot(p: tuple[float, float], ang_rad: float, c: tuple[float, float]) -> tuple[float, float]:
    ca, sa = math.cos(ang_rad), math.sin(ang_rad)
    dx, dy = p[0], p[1]
    return (c[0] + dx * ca - dy * sa, c[1] + dx * sa + dy * ca)


def _circle_d(cx: float, cy: float, r: float, ccw: bool) -> str:
    # SVG'de (y-aşağı) shoelace-pozitif sarım sweep=0'a denk gelir; ccw
    # bayrağımız shoelace işaretinden türediği için eşleme: ccw -> sweep=1'in
    # TERSİ deneysel olarak doğrulandı (test: sampled_ccw eşleşmesi)
    sweep = 1 if ccw else 0
    return (
        f"M {_fmt(cx - r)} {_fmt(cy)} "
        f"A {_fmt(r)} {_fmt(r)} 0 1 {sweep} {_fmt(cx + r)} {_fmt(cy)} "
        f"A {_fmt(r)} {_fmt(r)} 0 1 {sweep} {_fmt(cx - r)} {_fmt(cy)} Z"
    )


def _ellipse_d(cx: float, cy: float, a: float, b: float, ang_deg: float, ccw: bool) -> str:
    rad = math.radians(ang_deg)
    p0 = _rot((a, 0.0), rad, (cx, cy))
    p1 = _rot((-a, 0.0), rad, (cx, cy))
    sweep = 1 if ccw else 0
    rot = _fmt(ang_deg % 180.0)
    return (
        f"M {_fmt(p0[0])} {_fmt(p0[1])} "
        f"A {_fmt(a)} {_fmt(b)} {rot} 1 {sweep} {_fmt(p1[0])} {_fmt(p1[1])} "
        f"A {_fmt(a)} {_fmt(b)} {rot} 1 {sweep} {_fmt(p0[0])} {_fmt(p0[1])} Z"
    )


def _rounded_rect_d(
    cx: float, cy: float, w: float, h: float, ang_deg: float, r: float, ccw: bool
) -> str:
    """Merkez/boyut/açı verilen (yuvarlak köşeli) dikdörtgen path'i üretir.

    r <= 0.75 keskin dikdörtgen üretir. Köşe yayları daireseldir; rotasyon
    yalnız uç noktaları döndürür (dairesel yay rotasyona dayanıklıdır).
    """
    hw, hh = w / 2.0, h / 2.0
    r = max(0.0, min(r, hw - 0.01, hh - 0.01))
    rad = math.radians(ang_deg)
    c = (cx, cy)

    if r <= 0.75:
        # taban sıra shoelace-POZİTİF örneklenir (deneysel doğrulama);
        # negatif sarım istendiğinde ters çevrilir
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        if not ccw:
            local.reverse()
        pts = [_rot(p, rad, c) for p in local]
        parts = [f"M {_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
        parts += [f"L {_fmt(p[0])} {_fmt(p[1])}" for p in pts[1:]]
        return " ".join(parts) + " Z"

    # taban sıra (üst kenardan) shoelace-POZİTİF örneklenir ve köşe yayları
    # sweep=1 ister (deneysel doğrulama: ters sweep yayları içe büker, dev~10px);
    # negatif sarımda sıra ters + sweep=0. Sarımı nokta SIRASI belirler, sweep
    # yalnız köşe yayının dışbükeyliğini.
    local_seq: list[tuple[str, tuple[float, float]]] = [
        ("M", (-hw + r, -hh)),
        ("L", (hw - r, -hh)), ("A", (hw, -hh + r)),
        ("L", (hw, hh - r)), ("A", (hw - r, hh)),
        ("L", (-hw + r, hh)), ("A", (-hw, hh - r)),
        ("L", (-hw, -hh + r)), ("A", (-hw + r, -hh)),
    ]
    if not ccw:
        # sırayı ters çevir: komutlar uçlara göre yeniden kurulur
        pts_only = [p for _, p in local_seq]
        kinds = [k for k, _ in local_seq]
        pts_only.reverse()
        # ters yönde: M sonra sırayla; L/A tipleri aynadaki segmente göre
        rev_seq: list[tuple[str, tuple[float, float]]] = [("M", pts_only[0])]
        seg_kinds = kinds[1:]  # M sonrası segment tipleri
        seg_kinds.reverse()
        for kind, p in zip(seg_kinds, pts_only[1:]):
            rev_seq.append((kind, p))
        local_seq = rev_seq
    sweep = 1 if ccw else 0

    parts: list[str] = []
    for kind, p in local_seq:
        gp = _rot(p, rad, c)
        if kind == "M":
            parts.append(f"M {_fmt(gp[0])} {_fmt(gp[1])}")
        elif kind == "L":
            parts.append(f"L {_fmt(gp[0])} {_fmt(gp[1])}")
        else:
            parts.append(f"A {_fmt(r)} {_fmt(r)} 0 0 {sweep} {_fmt(gp[0])} {_fmt(gp[1])}")
    return " ".join(parts) + " Z"


def _validated(d_cand: str, pts: np.ndarray, tol: float, want_ccw: bool) -> str | None:
    """Aday şekil d'sini çift yönlü sapma + sarım yönüyle doğrular."""
    samples = _sample_d(d_cand)
    if samples is None:
        return None
    if _bidirectional_dev(pts, samples) > tol:
        return None
    # sarım yönü (fill-rule delikleri için kritik): uyuşmuyorsa reddet;
    # çağıran ccw bayrağını zaten orijinalden türetir, bu son güvencedir
    if (_signed_area(samples) > 0) != want_ccw:
        return None
    return d_cand


def try_fit_whole_shape(pts: np.ndarray, closed: bool) -> str | None:
    """Kapalı alt yolu TAM parametrik şekle oturtmayı dener.

    Sıra: daire -> elips -> dikdörtgen -> yuvarlak köşeli dikdörtgen. Kabul
    kriteri çift yönlü maksimum sapmadır (izin dışına taşma VE şeklin
    izlenmeyen bölgesi); sarım yönü korunur. Oturtulamazsa ``None`` — çağıran
    orijinal geometriyi sürdürür (asla bozulma yok).
    """
    if not closed or len(pts) < 12:
        return None
    x0, y0, x1, y1 = _bbox(pts)
    diag = math.hypot(x1 - x0, y1 - y0)
    if diag < 12:
        return None
    tol = max(1.5, 0.008 * diag)
    ccw = _signed_area(pts) > 0
    pf = pts.astype(np.float32)

    # 1) daire
    circ = _fit_circle(pts)
    if circ and circ[2] >= 4:
        cx, cy, r = circ
        if _circle_residual(pts, circ) <= tol:
            d = _validated(_circle_d(cx, cy, r, ccw), pts, tol, ccw)
            if d:
                return d

    # 2) elips
    if len(pts) >= 5:
        try:
            (ecx, ecy), (d1, d2), ang = cv2.fitEllipse(pf)
            a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
            ang_major = ang + (90.0 if d1 < d2 else 0.0)
            if b >= 2.5 and a / max(b, 1e-6) <= 20.0:
                d = _validated(_ellipse_d(ecx, ecy, a, b, ang_major, ccw), pts, tol, ccw)
                if d:
                    return d
        except cv2.error:
            pass

    # 3) dikdörtgen (döndürülmüş) ve 4) yuvarlak köşeli dikdörtgen
    try:
        (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(pf)
    except cv2.error:
        return None
    if min(rw, rh) < 4:
        return None
    # yerel çerçevede sağlam kenar/yarıçap tahmini: kenarlar yüzdelikle
    # (minAreaRect'in gürültü şişmesine dayanıklı), köşe yarıçapı KÖŞE-UZAKLIĞI
    # yöntemiyle (ideal keskin köşenin yola en yakın mesafesi d -> r = d/(√2-1))
    rad = math.radians(rang)
    ca, sa = math.cos(rad), math.sin(rad)
    dxv = pts[:, 0] - rcx
    dyv = pts[:, 1] - rcy
    qx = dxv * ca + dyv * sa
    qy = -dxv * sa + dyv * ca
    x_lo, x_hi = float(np.percentile(qx, 0.5)), float(np.percentile(qx, 99.5))
    y_lo, y_hi = float(np.percentile(qy, 0.5)), float(np.percentile(qy, 99.5))
    hw2, hh2 = (x_hi - x_lo) / 2.0, (y_hi - y_lo) / 2.0
    lcx, lcy = (x_hi + x_lo) / 2.0, (y_hi + y_lo) / 2.0
    if min(hw2, hh2) < 2:
        return None
    gcx = rcx + lcx * ca - lcy * sa
    gcy = rcy + lcx * sa + lcy * ca
    corner_d = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            d2c = np.hypot(qx - (lcx + sx * hw2), qy - (lcy + sy * hh2))
            corner_d.append(float(d2c.min()))
    r_est = float(np.median(corner_d)) / (math.sqrt(2.0) - 1.0)
    r_est = max(0.0, min(r_est, min(hw2, hh2)))

    r_candidates = [0.0] + ([round(r_est, 2)] if r_est >= 1.2 else [])
    for r_try in r_candidates:
        d = _validated(
            _rounded_rect_d(gcx, gcy, 2 * hw2, 2 * hh2, rang, r_try, ccw), pts, tol, ccw
        )
        if d:
            return d
    return None




def fit_whole_shapes_svg(svg_path: Path) -> dict[str, Any]:
    """SVG'de yalnız BÜTÜNSEL şekil oturtma uygular (çizgi/yay dilimleme yok).

    Renkli modlar için: organik path'lere dokunmaz; sadece gerçekten daire/
    elips/dikdörtgen/yuvarlak-dikdörtgen olan alt yollar idealize edilir.
    Alt yol bazlı fallback: oturtulamayan alt yol orijinal haliyle korunur.
    """
    if parse_path is None:
        return {"status": "skipped", "error": "svgpathtools yok"}
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}

    shapes_fitted = 0
    changed_any = False
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        try:
            subs = _flatten_subpaths(d)
        except Exception:  # noqa: BLE001
            continue
        if not subs:
            continue
        new_parts: list[str] = []
        fitted_here = 0
        for pts, closed, sub_d in subs:
            fitted = None
            try:
                fitted = try_fit_whole_shape(pts, closed)
            except Exception:  # noqa: BLE001
                fitted = None
            if fitted:
                new_parts.append(fitted)
                fitted_here += 1
            elif sub_d:
                new_parts.append(sub_d)
        if fitted_here and new_parts:
            el.set("d", " ".join(new_parts))
            shapes_fitted += fitted_here
            changed_any = True

    if changed_any:
        try:
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}
    return {"status": "completed" if changed_any else "no_change", "shapes_fitted": shapes_fitted}


def _fit_one_subpath(pts: np.ndarray, closed: bool) -> str | None:
    """Tek bir alt yolu line+arc primitiflerine oturtur. Güvenli değilse None."""
    x0, y0, x1, y1 = _bbox(pts)
    diag = math.hypot(x1 - x0, y1 - y0)
    if diag < 6:
        return None
    # önce BÜTÜNSEL şekil (tam daire/elips/dikdörtgen/yuvarlak-dikdörtgen):
    # parça parça yay/çizgiden her zaman daha temiz ve daha az düğüm
    try:
        whole = try_fit_whole_shape(pts, closed)
    except Exception:  # noqa: BLE001
        whole = None
    if whole:
        return whole
    # sıkı toleranslar: tek daireye oturmayan (eliptik) eğriler birden çok dairesel
    # yaya bölünüp gerçek şekli yakından takip eder (faset değil, çok-yay)
    line_tol = max(0.8, 0.004 * diag)
    arc_tol = max(1.4, 0.0038 * diag)

    breaks, is_arc = _segment_indices(pts, closed, diag)
    runs = _runs_from_breaks(len(pts), breaks, closed)
    prims: list = []
    for run in runs:
        if len(run) < 2:
            continue
        seg = pts[run]
        prefer_arc = bool(np.mean(is_arc[run]) > 0.5)
        prims.extend(_fit_run(seg, prefer_arc, line_tol, arc_tol))

    if not prims:
        return None
    prims = _postprocess_prims(prims, closed)
    d_sub = _primitives_to_d(prims, closed)
    if not d_sub:
        return None

    # SADAKAT kontrolü: idealize edilmiş yol, orijinal örneklenmiş noktalara
    # yakın mı? Değilse (ör. eliptik eğriyi az yayla kötü oturtma) reddet ->
    # çağıran orijinal spline'ı korur (asla kötüleştirme).
    try:
        new_samples = []
        for s in parse_path(d_sub).continuous_subpaths():
            try:
                L = s.length()
            except Exception:  # noqa: BLE001
                L = 50.0
            m = int(min(500, max(24, (L or 24) / 2.0)))
            for i in range(m + 1):
                p = s.point(i / m)
                new_samples.append((p.real, p.imag))
        ns = np.array(new_samples)
        if len(ns) < 2:
            return None
        op = pts if len(pts) <= 220 else pts[np.linspace(0, len(pts) - 1, 220).astype(int)]
        d2 = ((op[:, None, :] - ns[None, :, :]) ** 2).sum(axis=2)
        maxdev = float(np.sqrt(d2.min(axis=1)).max())
        if maxdev > max(1.8, 0.010 * diag):
            return None
    except Exception:  # noqa: BLE001
        return None
    return d_sub


def regularize_path_d(d: str) -> str | None:
    """Bir path'in alt yollarını idealize eder. Oturtulamayan alt yol ORİJİNAL
    halinde korunur (alt-yol bazlı fallback); hiçbiri iyileşmezse None döner.
    """
    if parse_path is None:
        return None
    try:
        subs = _flatten_subpaths(d)
    except Exception:  # noqa: BLE001
        return None
    if not subs:
        return None

    new_parts: list[str] = []
    any_ok = False
    for pts, closed, sub_d in subs:
        try:
            fitted = _fit_one_subpath(pts, closed)
        except Exception:  # noqa: BLE001
            fitted = None
        if fitted:
            new_parts.append(fitted)
            any_ok = True
        elif sub_d:
            new_parts.append(sub_d)  # bu alt yolu orijinal haliyle koru
        # sub_d yoksa o alt yolu atla

    if not any_ok or not new_parts:
        return None
    return " ".join(new_parts)


def regularize_svg_geometry(svg_path: Path) -> dict[str, Any]:
    """SVG'deki path'leri çizgi+yay primitiflerine idealize eder (yerinde yazar)."""
    if parse_path is None:
        return {"status": "skipped", "error": "svgpathtools yok"}
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}

    changed = 0
    kept = 0
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        try:
            new_d = regularize_path_d(d)
        except Exception as e:  # noqa: BLE001
            logger.debug("regularize hata, path korunuyor: %s", e)
            new_d = None
        if new_d and new_d != d:
            el.set("d", new_d)
            changed += 1
        else:
            kept += 1

    if changed:
        try:
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}
    return {"status": "completed" if changed else "no_change", "paths_regularized": changed, "paths_kept": kept}
