"""Yol basitleştirme + eğri yeniden-uydurma (literatür aşama 4+5).

Ölçüm: vtracer çıktısı aşırı-segmentli (medyan 4.97px segment, %12'si <2px);
bu aşırı-segmentasyon hem çentik/merdiven etkisinin (her minik segment ayrı
titreşir) hem ağır dosyanın köküdür. Kaynak boru hattının "Eğri Uydurma
(kübik Bézier + optimum parametrelendirme + İkiye Bölme) + Yolları Basitleştir"
adımlarını tek prensipli geçişte uygular:

1) KÖŞE-KORUYAN KOŞU AYRIMI: çapa poligonu pencereli dönüş ekstremiyle köşelere
   bölünür; gerçek köşeler ve açık-yol uçları her zaman korunur (netlik kalır).
   Köşeler arası her "koşu" bağımsız uydurulduğundan köşede teğet doğal kırılır.
2) SCHNEIDER EN KÜÇÜK KARELER KÜBİK BÉZIER (Graphics Gems, 1990): her koşunun
   YOĞUN eğri örneğine hata-kontrollü kübik Bézier uydurulur — optimum
   parametrelendirme (Newton-Raphson) + hata > tolerans olan en-uzak noktadan
   İkiye Bölme (bisection). Orijinal eğriye SADIK kalır (interpolasyon değil),
   teğet uzunluğu taşma-sivrisine karşı kirişin makul katına kıstırılır.

Böylece kontrol noktası sayısı düşer (basitleştirme) ve aşırı-segment titreşimi
akıcı Bézier'lerle giderilir. Yalnız mutlak M/L/C/Z path'leri işlenir; A (yay)
içeren — bütünsel şekle oturtulmuş daire/elips — atlanır. Piksel-metrik gürültülü
JPEG'i birebir ödüllendirdiğinden basitleştirme metriği hafif düşürebilir
(metrik-göz ayrışması) — bu yüzden opt-in edge_cleanup içinde, tolerans
kapısıyla çağrılır.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_FIT_ERR = 1.2          # Schneider Bézier uydurma toleransı (px, alt-piksel gürültü)
_CORNER_DEG = 48.0      # pencereli dönüş bundan büyükse köşe (korunur, keskin)
_CORNER_WIN = 2         # köşe testi pencere yarıçapı (çapa)
_MIN_ANCHORS = 12       # bundan kısa alt yol basitleştirilmez
_MAX_ITER = 4           # Newton-Raphson yeniden-parametrelendirme iterasyonu


# ---------------------------------------------------------------------------
# Schneider en küçük kareler kübik Bézier uydurma (Graphics Gems, 1990) —
# kaynak aşama 4: optimum parametrelendirme + hata-kontrollü İkiye Bölme.
# Orijinal noktalara SADIK kalır (interpolasyon değil), böylece fidelity kaybı
# minimaldir; hata > tolerans olan yerde en-uzak noktadan bölünür (bisection).
# ---------------------------------------------------------------------------
def _bezier_pt(bez, t: float):
    mt = 1.0 - t
    b0 = mt * mt * mt
    b1 = 3 * mt * mt * t
    b2 = 3 * mt * t * t
    b3 = t * t * t
    return (b0 * bez[0][0] + b1 * bez[1][0] + b2 * bez[2][0] + b3 * bez[3][0],
            b0 * bez[0][1] + b1 * bez[1][1] + b2 * bez[2][1] + b3 * bez[3][1])


def _chord_param(pts):
    u = [0.0]
    for i in range(1, len(pts)):
        u.append(u[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    tot = u[-1] or 1.0
    return [x / tot for x in u]


def _gen_bezier(pts, u, t1, t2):
    """u parametrelemesi ve uç teğetleriyle en küçük kareler kübik Bézier."""
    n = len(pts)
    a = [[(0.0, 0.0), (0.0, 0.0)] for _ in range(n)]
    for i in range(n):
        ui = u[i]
        mt = 1.0 - ui
        b1 = 3 * mt * mt * ui
        b2 = 3 * mt * ui * ui
        a[i][0] = (t1[0] * b1, t1[1] * b1)
        a[i][1] = (t2[0] * b2, t2[1] * b2)
    c00 = c01 = c11 = x0 = x1 = 0.0
    p0, p3 = pts[0], pts[-1]
    for i in range(n):
        ai0, ai1 = a[i][0], a[i][1]
        c00 += ai0[0] * ai0[0] + ai0[1] * ai0[1]
        c01 += ai0[0] * ai1[0] + ai0[1] * ai1[1]
        c11 += ai1[0] * ai1[0] + ai1[1] * ai1[1]
        ui = u[i]
        mt = 1.0 - ui
        b0 = mt * mt * mt
        b1 = 3 * mt * mt * ui
        b2 = 3 * mt * ui * ui
        b3 = ui * ui * ui
        tmpx = pts[i][0] - (b0 * p0[0] + b1 * p0[0] + b2 * p3[0] + b3 * p3[0])
        tmpy = pts[i][1] - (b0 * p0[1] + b1 * p0[1] + b2 * p3[1] + b3 * p3[1])
        x0 += ai0[0] * tmpx + ai0[1] * tmpy
        x1 += ai1[0] * tmpx + ai1[1] * tmpy
    det = c00 * c11 - c01 * c01
    seg = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
    if abs(det) < 1e-12:
        al = ar = seg / 3.0
    else:
        al = (x0 * c11 - x1 * c01) / det
        ar = (c00 * x1 - c01 * x0) / det
    # alpha kısıtı: TAŞMA-SİVRİSİ önleme. Serbest en küçük kareler negatif ya da
    # devasa teğet uzunluğu üretip eğriyi fırlatabiliyor (ölçülen spike artefaktı);
    # kiriş uzunluğunun makul katına kıstırılır.
    lo, hi = seg * 0.02, seg * 1.5
    if not (lo <= al <= hi):
        al = min(hi, max(lo, seg / 3.0))
    if not (lo <= ar <= hi):
        ar = min(hi, max(lo, seg / 3.0))
    return [p0, (p0[0] + t1[0] * al, p0[1] + t1[1] * al),
            (p3[0] + t2[0] * ar, p3[1] + t2[1] * ar), p3]


def _max_error(pts, u, bez):
    dmax, idx = 0.0, len(pts) // 2
    for i in range(1, len(pts) - 1):
        q = _bezier_pt(bez, u[i])
        d = (q[0] - pts[i][0]) ** 2 + (q[1] - pts[i][1]) ** 2
        if d >= dmax:
            dmax, idx = d, i
    return math.sqrt(dmax), idx


def _reparam(pts, u, bez):
    """Newton-Raphson ile parametreleri eğriye yakınsatır (optimum param.)."""
    out = []
    for i in range(len(pts)):
        t = u[i]
        q = _bezier_pt(bez, t)
        # 1. ve 2. türev (açık)
        mt = 1 - t
        qx1 = 3 * mt * mt * (bez[1][0] - bez[0][0]) + 6 * mt * t * (bez[2][0] - bez[1][0]) + 3 * t * t * (bez[3][0] - bez[2][0])
        qy1 = 3 * mt * mt * (bez[1][1] - bez[0][1]) + 6 * mt * t * (bez[2][1] - bez[1][1]) + 3 * t * t * (bez[3][1] - bez[2][1])
        qx2 = 6 * mt * (bez[2][0] - 2 * bez[1][0] + bez[0][0]) + 6 * t * (bez[3][0] - 2 * bez[2][0] + bez[1][0])
        qy2 = 6 * mt * (bez[2][1] - 2 * bez[1][1] + bez[0][1]) + 6 * t * (bez[3][1] - 2 * bez[2][1] + bez[1][1])
        num = (q[0] - pts[i][0]) * qx1 + (q[1] - pts[i][1]) * qy1
        den = qx1 * qx1 + qy1 * qy1 + (q[0] - pts[i][0]) * qx2 + (q[1] - pts[i][1]) * qy2
        out.append(t if abs(den) < 1e-12 else t - num / den)
    return out


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def _fit_cubic(pts, t1, t2, err, out, depth=0):
    if len(pts) == 2:
        d = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) / 3.0
        out.append([pts[0], (pts[0][0] + t1[0] * d, pts[0][1] + t1[1] * d),
                    (pts[1][0] + t2[0] * d, pts[1][1] + t2[1] * d), pts[1]])
        return
    u = _chord_param(pts)
    bez = _gen_bezier(pts, u, t1, t2)
    maxerr, split = _max_error(pts, u, bez)
    if maxerr < err:
        out.append(bez)
        return
    if maxerr < err * err and depth < 8:
        for _ in range(_MAX_ITER):
            u = _reparam(pts, u, bez)
            bez = _gen_bezier(pts, u, t1, t2)
            maxerr, split = _max_error(pts, u, bez)
            if maxerr < err:
                out.append(bez)
                return
    if depth > 24 or split <= 0 or split >= len(pts) - 1:
        out.append(bez)  # güvenlik: daha fazla bölme
        return
    tc = _unit(pts[split - 1], pts[split + 1])
    _fit_cubic(pts[:split + 1], t1, (-tc[0], -tc[1]), err, out, depth + 1)
    _fit_cubic(pts[split:], tc, t2, err, out, depth + 1)


def _win_turn_signed(pts: list, i: int, n: int, closed: bool, w: int) -> float:
    if closed:
        a, b, c = pts[(i - w) % n], pts[i], pts[(i + w) % n]
    else:
        if i - w < 0 or i + w >= n:
            return 0.0
        a, b, c = pts[i - w], pts[i], pts[i + w]
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cang))


def _fit_run(run: list, out_segs: list) -> None:
    """Köşe-arası bir koşuyu (>=2 nokta) Schneider ile Bézier'lere uydurur.

    Uç teğetler koşunun kendi yönünden alınır (köşede teğet doğal olarak kırılır,
    çünkü her koşu bağımsız uydurulur -> köşe keskin kalır).
    """
    if len(run) < 2:
        return
    t1 = _unit(run[0], run[1])
    t2 = _unit(run[-1], run[-2])
    beziers: list = []
    _fit_cubic(run, t1, t2, _FIT_ERR, beziers)
    for bez in beziers:
        out_segs.append(("C", (bez[1][0], bez[1][1]), (bez[2][0], bez[2][1]),
                         (bez[3][0], bez[3][1])))


def _refit_subpath(sp: dict[str, Any]) -> bool:
    """Bir alt yolu köşe-koruyan Schneider Bézier uydurma ile yeniden kurar.

    Çapa poligonu köşelerden koşulara bölünür; her koşu hata-kontrollü (İkiye
    Bölme) en küçük kareler kübik Bézier'e uydurulur. Orijinal noktalara sadık
    kaldığından fidelity kaybı minimal, kontrol noktası sayısı düşük, kenar
    pürüzsüz. sp yerinde güncellenir. Döner: değişti mi.
    """
    segs = sp["segs"]
    if len(segs) < _MIN_ANCHORS:
        return False
    closed = bool(sp["closed"])
    # ÇAPA poligonu (köşe tespiti için) + YOĞUN eğri örneği (uydurma girdisi):
    # vtracer C segmentleri kontrol noktalarında eğrilik taşır; yalnız uçlara
    # uydurmak eğriliği kaybettirip taşma-sivrisi/sapma üretir (ölçüldü). Gerçek
    # eğriyi örnekleyip ona uydururuz -> fidelity korunur.
    anchors = [tuple(sp["start"])] + [tuple(s[-1]) for s in segs]
    dup_last = closed and math.hypot(anchors[-1][0] - anchors[0][0], anchors[-1][1] - anchors[0][1]) < 1e-6
    ring = anchors[:-1] if dup_last else anchors
    n = len(ring)
    if n < _MIN_ANCHORS:
        return False

    corner = [False] * n
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            corner[i] = True
            continue
        if _win_turn_signed(ring, i, n, closed, _CORNER_WIN) > _CORNER_DEG:
            corner[i] = True

    # yoğun örnek + her çapanın yoğun-dizideki indeksi
    dense: list = [tuple(sp["start"])]
    anchor_at = [0]
    cur = tuple(sp["start"])
    for s in segs:
        end = tuple(s[-1])
        if s[0] == "L":
            dense.append(end)
        else:
            c1, c2 = tuple(s[1]), tuple(s[2])
            seglen = (math.hypot(c1[0]-cur[0], c1[1]-cur[1]) + math.hypot(c2[0]-c1[0], c2[1]-c1[1])
                      + math.hypot(end[0]-c2[0], end[1]-c2[1]))
            m = max(1, min(10, int(seglen / 2.0)))
            for k in range(1, m + 1):
                dense.append(_bezier_pt([cur, c1, c2, end], k / m))
        anchor_at.append(len(dense) - 1)
        cur = end
    if dup_last:
        anchor_at = anchor_at[:-1]

    forced = [i for i in range(n) if corner[i]]
    new_segs: list = []

    def _fit_dense(di0: int, di1: int) -> None:
        run = dense[di0:di1 + 1] if di1 >= di0 else (dense[di0:] + dense[:di1 + 1])
        if len(run) >= 2:
            _fit_run(run, new_segs)

    if closed:
        if len(forced) <= 1:
            # köşesiz kapalı halka: tam döngüyü tek koşu olarak uydur
            loop = dense + [dense[0]]
            _fit_run(loop, new_segs)
            start = ring[0]
        else:
            for a, b in zip(forced, forced[1:] + [forced[0]]):
                _fit_dense(anchor_at[a], anchor_at[b])
            start = ring[forced[0]]
    else:
        for a, b in zip(forced, forced[1:]):
            _fit_dense(anchor_at[a], anchor_at[b])
        start = ring[0]

    if not new_segs or len(new_segs) >= len(segs):
        return False  # basitleşme yoksa bırak (fidelity riskine girme)
    sp["start"] = start
    sp["segs"] = new_segs
    sp["closed"] = closed
    return True


def refit_svg_curves(svg_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    """SVG konturlarını Schneider kübik Bézier uydurmayla basitleştirir.

    ``out_path`` verilmezse yerinde yazılır. A içeren path'ler atlanır. Dönen:
    {"refit_paths", "seg_before", "seg_after"}. Benimseme çağırana (ölçüm).
    """
    from app.curve_fairing import _parse_subpaths, _serialize_subpaths  # noqa: PLC0415

    svg_path = Path(svg_path)
    dst = Path(out_path) if out_path is not None else svg_path
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"refit_paths": 0, "error": f"parse: {e}"}

    refit = 0
    seg_before = seg_after = 0
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        sps = _parse_subpaths(d)   # A / mutlak-olmayan -> None (atla)
        if not sps:
            continue
        changed = False
        for sp in sps:
            b = len(sp["segs"])
            try:
                if _refit_subpath(sp):
                    changed = True
                    seg_before += b
                    seg_after += len(sp["segs"])
            except Exception:  # noqa: BLE001
                continue
        if changed:
            el.set("d", _serialize_subpaths(sps))
            refit += 1

    if refit == 0:
        return {"refit_paths": 0, "seg_before": 0, "seg_after": 0}
    try:
        tree.write(str(dst), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"refit_paths": 0, "error": f"yazma: {e}"}
    return {"refit_paths": refit, "seg_before": seg_before, "seg_after": seg_after}
