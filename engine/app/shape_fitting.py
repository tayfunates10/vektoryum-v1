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


def _fit_one_subpath(pts: np.ndarray, closed: bool) -> str | None:
    """Tek bir alt yolu line+arc primitiflerine oturtur. Güvenli değilse None."""
    x0, y0, x1, y1 = _bbox(pts)
    diag = math.hypot(x1 - x0, y1 - y0)
    if diag < 6:
        return None
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
