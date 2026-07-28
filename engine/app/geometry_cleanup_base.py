"""Geometri temizleme katmanı.

SVG path geometrisini düzenler: yinelenen noktaları siler, çok kısa zigzag
segmentlerini temizler, doğrusal noktaları birleştirir, yatay/dikey (ve agresif
modda 45/135 derece) çizgileri eksene yaslar ve köşe kesişimlerini temizler.

Tasarım notları
---------------
* SVG, ``svgpathtools`` round-trip yerine ``xml.etree.ElementTree`` ile düzenlenir.
  Böylece ``fill``, ``fill-rule``, ``transform``, ``viewBox``, çizim sırası ve
  diğer öznitelikler korunur; yalnızca ``d`` attribute'u yeniden yazılır.
* Yalnızca tamamen poligonal (M/L/H/V/Z) alt path'ler temizlenir. İçinde eğri
  (C/S/Q/T/A) bulunan alt path'ler aynen korunur. Bu sayede ``geo_mixed`` ve
  ``logo_color`` gibi spline adaylarında yuvarlak harf formları bozulmaz.
* Herhangi bir hata durumunda dosya değiştirilmez ve rapor ``status: skipped``
  veya ``failed`` döner; API asla çökmemelidir.
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")


# ---------------------------------------------------------------------------
# Aggressiveness profilleri
# ---------------------------------------------------------------------------
def _profile(aggressiveness: str) -> dict[str, Any]:
    """Aggressiveness seviyesine göre tolerans setini döndürür."""
    presets = {
        "aggressive": {
            "dup_tol": 0.8,
            "min_seg": 2.4,
            "collinear_tol_deg": 4.5,
            "axis_tol_deg": 8.0,
            "diagonal": True,
            "diagonal_tol_deg": 7.0,
            "rdp_epsilon": 1.4,
            "corner_keep_deg": 26.0,
        },
        "standard": {
            "dup_tol": 0.6,
            "min_seg": 1.6,
            "collinear_tol_deg": 2.6,
            "axis_tol_deg": 5.0,
            "diagonal": True,
            "diagonal_tol_deg": 4.5,
            "rdp_epsilon": 0.9,
            "corner_keep_deg": 22.0,
        },
        "balanced": {
            "dup_tol": 0.5,
            "min_seg": 1.1,
            "collinear_tol_deg": 1.6,
            "axis_tol_deg": 3.0,
            "diagonal": False,
            "diagonal_tol_deg": 3.0,
            "rdp_epsilon": 0.55,
            "corner_keep_deg": 18.0,
        },
        "light": {
            "dup_tol": 0.4,
            "min_seg": 0.7,
            "collinear_tol_deg": 0.9,
            "axis_tol_deg": 2.0,
            "diagonal": False,
            "diagonal_tol_deg": 2.0,
            "rdp_epsilon": 0.0,
            "corner_keep_deg": 14.0,
        },
    }
    return presets.get(aggressiveness, presets["standard"])


# ---------------------------------------------------------------------------
# Path data parse / serialize
# ---------------------------------------------------------------------------
def _tokenize_path(d: str) -> list[tuple[str, list[float]]]:
    tokens: list[tuple[str, list[float]]] = []
    for match in _CMD_RE.finditer(d or ""):
        cmd = match.group(1)
        nums = [float(x) for x in _NUM_RE.findall(match.group(2))]
        tokens.append((cmd, nums))
    return tokens


def extract_points_from_path_data(d: str) -> list[dict[str, Any]]:
    """Bir ``d`` string'ini alt path'lere ayrıştırır.

    Dönen her alt path::

        {"points": [(x, y), ...], "closed": bool,
         "polygonal": bool, "tokens": [(cmd, nums), ...]}

    ``polygonal`` yalnızca M/L/H/V/Z komutlarından oluşan alt path'ler için
    ``True`` döner; eğri içerenler aynen yeniden serileştirilmek üzere
    ``tokens`` ile korunur.
    """
    subpaths: list[dict[str, Any]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    sp: dict[str, Any] | None = None

    def _new() -> dict[str, Any]:
        return {"points": [], "closed": False, "polygonal": True, "tokens": []}

    for cmd, nums in _tokenize_path(d):
        c = cmd.upper()
        rel = cmd.islower()

        if c == "M":
            if sp is not None:
                subpaths.append(sp)
            sp = _new()
            sp["tokens"].append((cmd, nums))
            first = True
            for j in range(0, len(nums) - 1, 2):
                x, y = nums[j], nums[j + 1]
                if rel:
                    x += cur[0]
                    y += cur[1]
                cur = (x, y)
                if first:
                    start = cur
                    first = False
                sp["points"].append(cur)
            continue

        if sp is None:
            sp = _new()

        sp["tokens"].append((cmd, nums))

        if c == "L":
            for j in range(0, len(nums) - 1, 2):
                x, y = nums[j], nums[j + 1]
                if rel:
                    x += cur[0]
                    y += cur[1]
                cur = (x, y)
                sp["points"].append(cur)
        elif c == "H":
            for v in nums:
                x = v + (cur[0] if rel else 0.0)
                cur = (x, cur[1])
                sp["points"].append(cur)
        elif c == "V":
            for v in nums:
                y = v + (cur[1] if rel else 0.0)
                cur = (cur[0], y)
                sp["points"].append(cur)
        elif c == "Z":
            sp["closed"] = True
            cur = start
        else:  # C, S, Q, T, A -> eğri; alt path artık poligonal değil
            sp["polygonal"] = False
            cur = _advance_curve(c, nums, cur)

    if sp is not None:
        subpaths.append(sp)
    return subpaths


def _advance_curve(c: str, nums: list[float], cur: tuple[float, float]) -> tuple[float, float]:
    """Eğri komutunun bitiş noktasını döndürür (sadece akış için, mutlak varsayımı)."""
    if not nums:
        return cur
    try:
        if c in ("C",):
            return (nums[-2], nums[-1])
        if c in ("S", "Q"):
            return (nums[-2], nums[-1])
        if c == "T":
            return (nums[-2], nums[-1])
        if c == "A":
            return (nums[-2], nums[-1])
    except IndexError:
        return cur
    return cur


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if "." in f"{v:.2f}" else f"{v:.2f}"


def rebuild_path_from_points(points: list[tuple[float, float]], closed: bool) -> str:
    """Nokta listesinden ``M ... L ... Z`` formatında bir d-string üretir."""
    if len(points) < 2:
        return ""
    parts = [f"M {_fmt(points[0][0])} {_fmt(points[0][1])}"]
    for x, y in points[1:]:
        parts.append(f"L {_fmt(x)} {_fmt(y)}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _reserialize_tokens(tokens: list[tuple[str, list[float]]]) -> str:
    out = []
    for cmd, nums in tokens:
        if nums:
            out.append(cmd + " " + " ".join(_fmt(n) for n in nums))
        else:
            out.append(cmd)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Nokta düzeyinde temizleme yardımcıları
# ---------------------------------------------------------------------------
def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def remove_duplicate_points(points: list[tuple[float, float]], tol: float = 0.6) -> list[tuple[float, float]]:
    """Ardışık çakışan noktaları temizler."""
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if _dist(p, out[-1]) > tol:
            out.append(p)
    # kapalı poligonda baş ve son aynıysa sonu at
    if len(out) > 1 and _dist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def remove_short_segments(points: list[tuple[float, float]], closed: bool, min_len: float) -> list[tuple[float, float]]:
    """Çok kısa segment üreten ara noktaları kaldırır (köşe noktalarını koruyarak)."""
    if len(points) < 4 or min_len <= 0:
        return points
    out = list(points)
    changed = True
    while changed and len(out) > 3:
        changed = False
        n = len(out)
        for i in range(n):
            a = out[i]
            b = out[(i + 1) % n]
            if _dist(a, b) < min_len:
                # i+1 noktasını kaldır; ama keskin köşeyse koru
                j = (i + 1) % n
                prev_p = out[(j - 1) % n]
                next_p = out[(j + 1) % n]
                turn = _turn_angle_deg(prev_p, out[j], next_p)
                if turn < 35.0:  # yumuşak/gereksiz nokta -> sil
                    out.pop(j)
                    changed = True
                    break
    return out


def _turn_angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """b köşesindeki dönüş açısı (0 = düz devam, 180 = tam geri dönüş)."""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def merge_collinear_points(points: list[tuple[float, float]], closed: bool, tol_deg: float) -> list[tuple[float, float]]:
    """Neredeyse aynı doğru üzerindeki ardışık noktaları birleştirir."""
    if len(points) < 3:
        return points
    keep = [True] * len(points)
    n = len(points)
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        a = points[(i - 1) % n]
        b = points[i]
        c = points[(i + 1) % n]
        if _turn_angle_deg(a, b, c) < tol_deg:
            keep[i] = False
    out = [p for p, k in zip(points, keep) if k]
    return out if len(out) >= 3 else points


def snap_angle(angle_deg: float, targets: list[float], tol_deg: float) -> float | None:
    """Verilen açıya en yakın hedef açıyı (tolerans içindeyse) döndürür."""
    a = angle_deg % 180.0
    best = None
    best_d = tol_deg
    for t in targets:
        d = abs(a - (t % 180.0))
        d = min(d, 180.0 - d)
        if d <= best_d:
            best_d = d
            best = t
    return best


def line_intersection(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> tuple[float, float] | None:
    """(p1->p2) ve (p3->p4) doğrularının kesişim noktası; paralel ise None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def snap_axis_aligned_points(points: list[tuple[float, float]], closed: bool, tol_deg: float) -> list[tuple[float, float]]:
    """Yatay segmentlerin y'sini, dikey segmentlerin x'ini eşitleyerek eksene yaslar.

    Union-find ile birbirine eksende bağlı noktalar gruplanır ve grup ortalaması
    atanır. Diyagonal segmentler korunur, böylece şekil kapanışı bozulmaz.
    """
    n = len(points)
    if n < 2:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    uf_x = _UnionFind(n)
    uf_y = _UnionFind(n)

    seg_count = n if closed else n - 1
    for i in range(seg_count):
        j = (i + 1) % n
        dx = xs[j] - xs[i]
        dy = ys[j] - ys[i]
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        # yatay (0/180) -> aynı y
        if min(ang, 180.0 - ang) <= tol_deg:
            uf_y.union(i, j)
        # dikey (90) -> aynı x
        if abs(ang - 90.0) <= tol_deg:
            uf_x.union(i, j)

    # grup ortalamaları
    gx: dict[int, list[int]] = {}
    gy: dict[int, list[int]] = {}
    for i in range(n):
        gx.setdefault(uf_x.find(i), []).append(i)
        gy.setdefault(uf_y.find(i), []).append(i)

    new_x = list(xs)
    new_y = list(ys)
    for members in gx.values():
        if len(members) > 1:
            mean = sum(xs[m] for m in members) / len(members)
            for m in members:
                new_x[m] = mean
    for members in gy.values():
        if len(members) > 1:
            mean = sum(ys[m] for m in members) / len(members)
            for m in members:
                new_y[m] = mean

    return list(zip(new_x, new_y))


def snap_diagonal_points(points: list[tuple[float, float]], closed: bool, tol_deg: float) -> list[tuple[float, float]]:
    """45/135 dereceye yakın segmentleri, orta noktayı koruyarak tam diyagonale yaslar.

    Her segment bağımsız hesaplanır; paylaşılan noktalar komşu hedeflerinin
    ortalamasına taşınır (net kayma olmaz).
    """
    n = len(points)
    if n < 2:
        return points
    targets = [45.0, 135.0]
    acc: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]  # sum_x, sum_y, weight

    seg_count = n if closed else n - 1
    for i in range(seg_count):
        j = (i + 1) % n
        a = points[i]
        b = points[j]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        snapped = snap_angle(ang, targets, tol_deg)
        if snapped is None:
            acc[i][0] += a[0]; acc[i][1] += a[1]; acc[i][2] += 1
            acc[j][0] += b[0]; acc[j][1] += b[1]; acc[j][2] += 1
            continue
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        rad = math.radians(snapped)
        ux, uy = math.cos(rad), math.sin(rad)
        # a ve b'yi orta noktadan geçen tam-açılı doğruya projekte et
        ta = ((a[0] - mid[0]) * ux + (a[1] - mid[1]) * uy)
        tb = ((b[0] - mid[0]) * ux + (b[1] - mid[1]) * uy)
        na = (mid[0] + ta * ux, mid[1] + ta * uy)
        nb = (mid[0] + tb * ux, mid[1] + tb * uy)
        acc[i][0] += na[0]; acc[i][1] += na[1]; acc[i][2] += 1
        acc[j][0] += nb[0]; acc[j][1] += nb[1]; acc[j][2] += 1

    out = []
    for i in range(n):
        if acc[i][2] > 0:
            out.append((acc[i][0] / acc[i][2], acc[i][1] / acc[i][2]))
        else:
            out.append(points[i])
    return out


def snap_line_segment(
    a: tuple[float, float], b: tuple[float, float],
    targets: list[float], tol_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tek bir segmenti en yakın hedef açıya yaslar (b ucu döndürülür)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return a, b
    ang = math.degrees(math.atan2(dy, dx))
    snapped = snap_angle(ang, targets, tol_deg)
    if snapped is None:
        return a, b
    rad = math.radians(snapped if abs((ang % 360)) <= 180 else snapped)
    # yönü koru
    sign = 1.0
    base = math.radians(snap_angle(ang, targets, tol_deg) or 0.0)
    nb = (a[0] + math.cos(base) * length * sign, a[1] + math.sin(base) * length * sign)
    return a, nb


def _rdp(points: list[tuple[float, float]], epsilon: float, keep: list[bool]) -> list[tuple[float, float]]:
    """Köşe koruyan Ramer–Douglas–Peucker sadeleştirme."""
    if epsilon <= 0 or len(points) < 3:
        return points

    def _rdp_rec(pts: list[tuple[float, float]], ks: list[bool]) -> list[tuple[float, float]]:
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = _point_line_distance(pts[i], a, b)
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon or any(ks[1:-1]):
            # zorunlu korunan nokta veya epsilon aşıldı -> böl
            if dmax <= epsilon:
                # korunan noktayı bölme indeksine al
                for i in range(1, len(pts) - 1):
                    if ks[i]:
                        idx = i
                        break
            left = _rdp_rec(pts[: idx + 1], ks[: idx + 1])
            right = _rdp_rec(pts[idx:], ks[idx:])
            return left[:-1] + right
        return [a, b]

    return _rdp_rec(points, keep)


def _point_line_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    if a == b:
        return _dist(p, a)
    num = abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
    den = math.hypot(b[0] - a[0], b[1] - a[1])
    return num / den if den else 0.0


def simplify_polygon_preserve_corners(
    points: list[tuple[float, float]], closed: bool, epsilon: float, corner_keep_deg: float,
) -> list[tuple[float, float]]:
    """Köşeleri (keskin dönüşleri) koruyarak poligonu sadeleştirir."""
    if epsilon <= 0 or len(points) < 4:
        return points
    n = len(points)
    keep = [False] * n
    for i in range(n):
        a = points[(i - 1) % n]
        b = points[i]
        c = points[(i + 1) % n]
        if _turn_angle_deg(a, b, c) >= corner_keep_deg:
            keep[i] = True
    if closed:
        # kapalı poligonu en belirgin köşeden açıp RDP uygula
        try:
            anchor = max(range(n), key=lambda i: _turn_angle_deg(points[(i - 1) % n], points[i], points[(i + 1) % n]))
        except ValueError:
            anchor = 0
        rolled = points[anchor:] + points[:anchor] + [points[anchor]]
        rolled_keep = keep[anchor:] + keep[:anchor] + [keep[anchor]]
        simplified = _rdp(rolled, epsilon, rolled_keep)
        if len(simplified) > 1 and simplified[0] == simplified[-1]:
            simplified = simplified[:-1]
        return simplified if len(simplified) >= 3 else points
    return _rdp(points, epsilon, keep)


def clean_corner_intersections(points: list[tuple[float, float]], closed: bool, tol_deg: float) -> list[tuple[float, float]]:
    """Eksene yaslanmış komşu segmentlerin köşelerini tam kesişime taşır.

    İki ardışık segment de eksene yakınsa, köşe noktası iki çizginin tam
    kesişimine çekilir; böylece dik köşeler net olur.
    """
    n = len(points)
    if n < 4:
        return points
    out = list(points)
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        p0 = out[(i - 1) % n]
        p1 = out[i]
        p2 = out[(i + 1) % n]
        a1 = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
        a2 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
        axis1 = min(a1, 180 - a1) <= tol_deg or abs(a1 - 90) <= tol_deg
        axis2 = min(a2, 180 - a2) <= tol_deg or abs(a2 - 90) <= tol_deg
        if axis1 and axis2:
            inter = line_intersection(p0, p1, p1, p2)
            if inter and _dist(inter, p1) < max(_dist(p0, p1), _dist(p1, p2)):
                out[i] = inter
    return out


# ---------------------------------------------------------------------------
# Skorlama yardımcıları
# ---------------------------------------------------------------------------
def _line_runs_from_d(d: str) -> list[list[tuple[float, float]]]:
    """d-string'ten gerçek DÜZ ÇİZGİ run'larını çıkarır.

    Her run, L/H/V (ve M sonrası örtük L) ile bağlı ardışık noktalardan oluşan
    bir polyline'dır. Eğri komutları (C/S/Q/T/A) run'u keser ve eğri uç noktası
    yeni olası run'un başı olur. Böylece düz-çizgi metrikleri eğrilerden
    etkilenmez; bezier eğrileri "faset" sayılmaz.
    """
    runs: list[list[tuple[float, float]]] = []
    run: list[tuple[float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)

    def flush() -> None:
        nonlocal run
        if len(run) >= 2:
            runs.append(run)
        run = []

    for cmd, nums in _tokenize_path(d):
        c = cmd.upper()
        rel = cmd.islower()
        if c == "M":
            flush()
            first = True
            for j in range(0, len(nums) - 1, 2):
                x, y = nums[j], nums[j + 1]
                if rel:
                    x += cur[0]; y += cur[1]
                cur = (x, y)
                if first:
                    start = cur
                    run = [cur]
                    first = False
                else:
                    run.append(cur)  # örtük L
        elif c == "L":
            for j in range(0, len(nums) - 1, 2):
                x, y = nums[j], nums[j + 1]
                if rel:
                    x += cur[0]; y += cur[1]
                cur = (x, y)
                run.append(cur)
        elif c == "H":
            for v in nums:
                x = v + (cur[0] if rel else 0.0)
                cur = (x, cur[1]); run.append(cur)
        elif c == "V":
            for v in nums:
                y = v + (cur[1] if rel else 0.0)
                cur = (cur[0], y); run.append(cur)
        elif c == "Z":
            if run:
                run.append(start)
            cur = start
            flush()
        else:  # eğri: run'u kes
            flush()
            cur = _advance_curve(c, nums, cur)
            run = [cur]
    flush()
    return runs


def _runs_stats(runs: list[list[tuple[float, float]]]) -> dict[str, Any]:
    total_len = 0.0
    aligned_len = 0.0
    seg = 0
    short = 0
    facet_v = 0
    corner_v = 0
    dirty_v = 0
    verts = 0
    short_thresh = 2.5
    axis_targets = [0.0, 45.0, 90.0, 135.0]

    for run in runs:
        for i in range(len(run) - 1):
            a, b = run[i], run[i + 1]
            length = _dist(a, b)
            if length < 1e-9:
                continue
            seg += 1
            total_len += length
            if length < short_thresh:
                short += 1
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            if snap_angle(ang, axis_targets, 6.0) is not None:
                aligned_len += length
        for i in range(1, len(run) - 1):
            verts += 1
            t = _turn_angle_deg(run[i - 1], run[i], run[i + 1])
            if 25.0 <= t <= 150.0:
                corner_v += 1          # gerçek (kasıtlı) köşe
            elif 3.0 <= t < 25.0:
                facet_v += 1           # eğrinin çizgilerle yaklaştırıldığı faset
            elif t > 150.0:
                dirty_v += 1           # sivri geri-dönüş artefaktı
            # t < 3 -> neredeyse doğrusal, ihmal edilir

    return {
        "seg": seg, "short": short, "total_len": total_len, "aligned_len": aligned_len,
        "facet_v": facet_v, "corner_v": corner_v, "dirty_v": dirty_v, "verts": verts,
    }


def calculate_path_straightness_score(runs: list[list[tuple[float, float]]]) -> float:
    """Düz olması gereken yerlerin temizliği. Faset (eğriyi köşeleştirme) cezalandırılır."""
    s = _runs_stats(runs)
    if s["verts"] == 0 and s["seg"] == 0:
        return 0.85
    facet_ratio = s["facet_v"] / max(s["verts"], 1)
    short_ratio = s["short"] / max(s["seg"], 1)
    return round(max(0.0, 1.0 - 0.7 * facet_ratio - 0.5 * short_ratio), 4)


def calculate_axis_alignment_score(runs: list[list[tuple[float, float]]]) -> float:
    s = _runs_stats(runs)
    if s["total_len"] <= 1e-6:
        return 0.85
    return round(min(1.0, s["aligned_len"] / s["total_len"]), 4)


def calculate_corner_cleanliness_score(runs: list[list[tuple[float, float]]]) -> float:
    s = _runs_stats(runs)
    if s["verts"] == 0:
        return 0.9
    dirty_ratio = s["dirty_v"] / max(s["verts"], 1) + s["short"] / max(s["seg"], 1)
    return round(max(0.0, 1.0 - dirty_ratio), 4)


def calculate_geometry_report(d_list: list[str], stats: dict[str, int] | None = None) -> dict[str, float]:
    """Bir d-string listesinden eğri-duyarlı geometri skorları üretir.

    Bezier eğrileri "düz değil" diye cezalandırılmaz; yalnızca düz çizgilerin
    temizliği, eksen hizası ve faset/sivri-uç artefaktları ölçülür.
    """
    runs: list[list[tuple[float, float]]] = []
    for d in d_list:
        try:
            runs.extend(_line_runs_from_d(d))
        except Exception:  # noqa: BLE001
            continue
    straight = calculate_path_straightness_score(runs)
    axis = calculate_axis_alignment_score(runs)
    corner = calculate_corner_cleanliness_score(runs)
    geometry = round((straight + axis + corner) / 3.0, 4)
    return {
        "straight_edge_score": straight,
        "axis_alignment_score": axis,
        "corner_cleanliness_score": corner,
        "geometry_score": geometry,
    }


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------
def _clean_polygon(points: list[tuple[float, float]], closed: bool, prof: dict[str, Any]) -> list[tuple[float, float]]:
    pts = remove_duplicate_points(points, prof["dup_tol"])
    if len(pts) < 3:
        return pts
    pts = merge_collinear_points(pts, closed, prof["collinear_tol_deg"])
    pts = remove_short_segments(pts, closed, prof["min_seg"])
    pts = snap_axis_aligned_points(pts, closed, prof["axis_tol_deg"])
    if prof["diagonal"]:
        pts = snap_diagonal_points(pts, closed, prof["diagonal_tol_deg"])
    if prof["rdp_epsilon"] > 0:
        pts = simplify_polygon_preserve_corners(pts, closed, prof["rdp_epsilon"], prof["corner_keep_deg"])
    pts = clean_corner_intersections(pts, closed, prof["axis_tol_deg"])
    pts = merge_collinear_points(pts, closed, prof["collinear_tol_deg"])
    return pts


def cleanup_svg_geometry(
    svg_path: Path,
    mode: str = "geometric_logo",
    aggressiveness: str = "standard",
) -> dict[str, Any]:
    """SVG'deki path geometrisini temizler ve bir geometri raporu döndürür.

    Temizlenmiş SVG aynı dosyanın üzerine yazılır. Hata durumunda dosya
    değiştirilmez.
    """
    svg_path = Path(svg_path)
    prof = _profile(aggressiveness)

    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        logger.warning("SVG parse edilemedi, geometri temizleme atlandı: %s", e)
        return {"status": "failed", "error": f"svg parse failed: {e}", "report": _empty_report()}

    stats = {"nodes_before": 0, "nodes_after": 0, "paths_processed": 0, "paths_skipped": 0}
    cleaned_d_list: list[str] = []
    changed_any = False

    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        try:
            subpaths = extract_points_from_path_data(d)
        except Exception as e:  # noqa: BLE001
            logger.debug("path ayrıştırılamadı, korunuyor: %s", e)
            stats["paths_skipped"] += 1
            cleaned_d_list.append(d)
            continue

        if not any(sp["polygonal"] for sp in subpaths):
            # tamamen eğri tabanlı path -> aynen koru (eğriler bozulmaz)
            stats["paths_skipped"] += 1
            cleaned_d_list.append(d)
            continue

        new_parts: list[str] = []
        for sp in subpaths:
            if sp["polygonal"] and len(sp["points"]) >= 3:
                stats["nodes_before"] += len(sp["points"])
                cleaned = _clean_polygon(sp["points"], sp["closed"], prof)
                if len(cleaned) < 3:
                    cleaned = sp["points"]
                stats["nodes_after"] += len(cleaned)
                new_parts.append(rebuild_path_from_points(cleaned, sp["closed"]))
            else:
                # düz çizgi + eğri karışık alt path -> eğriler korunarak aynen yaz
                new_parts.append(_reserialize_tokens(sp["tokens"]))

        new_d = " ".join(p for p in new_parts if p)
        if new_d and new_d != d:
            el.set("d", new_d)
            changed_any = True
        cleaned_d_list.append(new_d or d)
        stats["paths_processed"] += 1

    report = calculate_geometry_report(cleaned_d_list)

    if changed_any:
        try:
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Temizlenmiş SVG yazılamadı: %s", e)
            return {"status": "failed", "error": f"svg write failed: {e}", "report": report, "stats": stats}

    logger.info(
        "Geometri temizleme tamam (%s/%s): node %s -> %s",
        mode, aggressiveness, stats["nodes_before"], stats["nodes_after"],
    )
    return {
        "status": "completed" if changed_any else "no_change",
        "mode": mode,
        "aggressiveness": aggressiveness,
        "stats": stats,
        "report": report,
    }


def _empty_report() -> dict[str, float]:
    return {
        "straight_edge_score": 0.0,
        "axis_alignment_score": 0.0,
        "corner_cleanliness_score": 0.0,
        "geometry_score": 0.0,
    }


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        return None
    try:
        return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    except ValueError:
        return None


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _path_bbox_weight(d: str) -> float:
    nums = [float(x) for x in _NUM_RE.findall(d or "")]
    if len(nums) < 4:
        return 1.0
    xs = nums[0::2]
    ys = nums[1::2]
    n = min(len(xs), len(ys))
    if n < 2:
        return 1.0
    w = (max(xs[:n]) - min(xs[:n]))
    h = (max(ys[:n]) - min(ys[:n]))
    return max(1.0, w * h)


def _dist2(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def consolidate_svg_palette(
    svg_path: Path,
    max_colors: int,
    merge_tol: float = 12.0,
    canonical: list[tuple[int, int, int]] | None = None,
    snap_tol: float = 42.0,
) -> dict[str, Any]:
    """SVG path ``fill`` renklerini net, düzenlenebilir bir palete indirir.

    Adımlar:
    1. Renkler kapladıkları yaklaşık alana (bbox) göre ağırlıklandırılır.
    2. Birbirine ``merge_tol`` mesafesinden yakın renkler tek kümede birleşir
       (VTracer'ın kenarlarda ürettiği ±1 ara-ton dilimleri temizlenir).
    3. ``max_colors`` en ağır küme korunur.
    4. ``canonical`` verilirse, korunan renkler ``snap_tol`` içindeyse tam
       kanonik değere (ör. saf #000000/#ffffff/#ff0000) yaslanır.
    5. Her path en yakın korunan renge atanır. ``d`` geometrisi değişmez.
    """
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "error": str(e)}

    weights: dict[tuple[int, int, int], float] = {}
    path_els: list[tuple[Any, tuple[int, int, int]]] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        fill = el.get("fill")
        rgb = _hex_to_rgb(fill) if fill else None
        if rgb is None:
            continue
        w = _path_bbox_weight(el.get("d", ""))
        weights[rgb] = weights.get(rgb, 0.0) + w
        path_els.append((el, rgb))

    before = len(weights)
    if not weights:
        return {"status": "no_change", "colors_before": 0, "colors_after": 0}

    # 1-2) ağırlık sırasına göre yakın renk kümeleme
    merge_t2 = merge_tol * merge_tol
    clusters: list[dict[str, Any]] = []  # {"rgb": repr, "weight": w}
    for rgb, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        placed = False
        for cl in clusters:
            if _dist2(rgb, cl["rgb"]) <= merge_t2:
                cl["weight"] += w
                placed = True
                break
        if not placed:
            clusters.append({"rgb": rgb, "weight": w})

    # 3) en ağır max_colors küme
    clusters.sort(key=lambda c: c["weight"], reverse=True)
    kept = clusters[:max_colors]

    # 4) kanonik yaslama
    snap_t2 = snap_tol * snap_tol
    for cl in kept:
        rep = cl["rgb"]
        if canonical:
            best = min(canonical, key=lambda c: _dist2(rep, c))
            if _dist2(rep, best) <= snap_t2:
                rep = best
        cl["final"] = rep

    # 5) her orijinal rengi en yakın korunan kümenin final değerine eşle
    def _map(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        cl = min(kept, key=lambda c: _dist2(rgb, c["rgb"]))
        return cl["final"]

    changed = False
    final_colors: set[tuple[int, int, int]] = set()
    for el, rgb in path_els:
        new_rgb = _map(rgb)
        final_colors.add(new_rgb)
        new_hex = _rgb_to_hex(new_rgb)
        if (el.get("fill") or "").lower() != new_hex:
            el.set("fill", new_hex)
            changed = True

    if changed:
        try:
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e), "colors_before": before}

    return {"status": "completed", "colors_before": before, "colors_after": len(final_colors)}


def compute_geometry_report_for_svg(svg_path: Path) -> dict[str, float]:
    """Bir SVG dosyasını değiştirmeden geometri skorlarını hesaplar (scoring fallback)."""
    try:
        ET.register_namespace("", SVG_NS)
        root = ET.parse(str(svg_path)).getroot()
    except Exception:  # noqa: BLE001
        return _empty_report()
    d_list: list[str] = []
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if d:
            d_list.append(d)
    return calculate_geometry_report(d_list)
