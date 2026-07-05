"""Kontur gürültü-giderme: tırtıklı/kırık kenarları özellik-koruyarak yumuşatma.

vtracer, düz olması gereken organik konturu (güneş ışını, tepe eğrisi, harf
kenarı) ±0.5-1px zikzak yapan çok sayıda kısa segmentle izler — teğet jitter
ölçümünde segmentlerin yarısından fazlası >40° dönüyordu. curve_fairing yalnız
C-C eklemindeki KÜÇÜK açı kinklerini (1.5-25°) hizalar; bu yüksek-frekanslı
gürültüye dokunmaz.

YÖNTEM — Taubin λ|μ yumuşatma (Taubin 1995): çapa dizisine ardışık λ (pozitif)
ve μ (negatif, |μ|>λ) Laplasyen adımı uygulanır. Bu, alçak-geçiren bir filtredir:
yüksek-frekanslı zikzağı söker ama düşük-frekanslı GERÇEK eğri şeklini (ve μ
adımı sayesinde hacmi/ölçeği) korur — Laplasyen yumuşatmanın küçülme kusuru
olmaz. Özellik koruma:

* KÖŞELER (pencereli dönüş açısı > köşe eşiği: harf köşesi, ışın ucu) dondurulur
  — keskinliği bozulmaz.
* AÇIK alt yolların uçları dondurulur (bitişik path'le hizası kaymasın).
* Segment kontrol noktaları çapalarıyla birlikte ötelenir (yerel eğri şekli
  korunur; boundary_refit ile aynı teknik).

Yalnız mutlak M/L/C/Z path'leri işlenir; A (yay) içeren — yani bütünsel şekle
oturtulmuş daire/elips — path'ler parser tarafından atlanır (dokunulmaz).
Benimseme kararı ÇAĞIRANDA: ölçülen fidelity düşerse eski çıktı korunur.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_LAMBDA = 0.33          # Taubin pozitif adım
_MU = -0.34             # Taubin negatif adım (|μ|>λ: küçülme önleme)
_ITERS = 5              # yumuşatma iterasyonu (az: ince yapı erimesin)
_CORNER_DEG = 50.0      # pencereli dönüş bundan büyükse köşe (dondurulur)
_CORNER_WIN = 2         # köşe testi pencere yarıçapı (çapa)
_MIN_ANCHORS = 10       # bundan kısa alt yol yumuşatılmaz (zaten şekil/kısa)
_MAX_MOVE = 1.1         # çapa başına azami toplam kayma (px) — ince yapı koruma
_MIN_BBOX = 22.0        # bbox min boyutu bundan küçükse (ince şerit) atla


def _win_turn_deg(pts: list, i: int, n: int, closed: bool, w: int) -> float:
    """i çapasında pencereli dönüş açısı (gürültüye dayanıklı köşe tespiti)."""
    if closed:
        a = pts[(i - w) % n]
        b = pts[i]
        c = pts[(i + w) % n]
    else:
        if i - w < 0 or i + w >= n:
            return 0.0
        a, b, c = pts[i - w], pts[i], pts[i + w]
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cang))


def _smooth_subpath(sp: dict[str, Any]) -> int:
    """Bir alt yolun çapalarını Taubin ile yumuşatır (köşe/uç korumalı).

    Döner: taşınan çapa sayısı. Segment kontrol noktaları çapa deltasıyla
    birlikte ötelenir.
    """
    segs = sp["segs"]
    n_seg = len(segs)
    if n_seg < _MIN_ANCHORS:
        return 0
    closed = bool(sp["closed"])
    # çapa dizisi
    pts = [list(sp["start"])] + [list(s[-1]) for s in segs]
    # kapalı yolda son çapa == start ise tekrarı çıkar (halka)
    dup_last = closed and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 1e-6
    ring = pts[:-1] if dup_last else pts
    n = len(ring)
    if n < _MIN_ANCHORS:
        return 0
    # İNCE ŞERİT KORUMASI: güneş ışını / ince kontur gibi dar yapılar Taubin'de
    # iki kenarı birbirine çekildiğinden erir. bbox min boyutu eşiğin altındaysa
    # atla (ölçüldü: ışınlar inceliyordu).
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if min(max(xs) - min(xs), max(ys) - min(ys)) < _MIN_BBOX:
        return 0

    # hareketlilik: köşe/uç = 0 (dondur), diğer = 1
    mob = [1.0] * n
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            mob[i] = 0.0
            continue
        if _win_turn_deg(ring, i, n, closed, _CORNER_WIN) > _CORNER_DEG:
            mob[i] = 0.0

    orig = [tuple(p) for p in ring]
    cur = [list(p) for p in ring]

    def _laplacian(i: int) -> tuple[float, float]:
        if closed:
            a, b = cur[(i - 1) % n], cur[(i + 1) % n]
        else:
            a = cur[i - 1] if i > 0 else cur[i]
            b = cur[i + 1] if i < n - 1 else cur[i]
        return ((a[0] + b[0]) * 0.5 - cur[i][0], (a[1] + b[1]) * 0.5 - cur[i][1])

    for _ in range(_ITERS):
        for factor in (_LAMBDA, _MU):
            deltas = [_laplacian(i) for i in range(n)]
            for i in range(n):
                if mob[i] <= 0.0:
                    continue
                cur[i][0] += factor * deltas[i][0]
                cur[i][1] += factor * deltas[i][1]

    # toplam kayma sınırı
    moved = 0
    anchor_delta: list[tuple[float, float]] = []
    for i in range(n):
        dx, dy = cur[i][0] - orig[i][0], cur[i][1] - orig[i][1]
        d = math.hypot(dx, dy)
        if d > _MAX_MOVE:
            s = _MAX_MOVE / d
            dx, dy = dx * s, dy * s
        anchor_delta.append((dx, dy))
        if d > 0.05:
            moved += 1
    if moved == 0:
        return 0

    # çapa indeksini (halka) -> pts indeksine eşle
    full_delta = list(anchor_delta)
    if dup_last:
        full_delta.append(anchor_delta[0])  # kapanış çapası start ile aynı hareket

    def _shift(p: tuple[float, float], d: tuple[float, float]) -> tuple[float, float]:
        return (p[0] + d[0], p[1] + d[1])

    sp["start"] = _shift(sp["start"], full_delta[0])
    for i, seg in enumerate(segs):
        d_start, d_end = full_delta[i], full_delta[i + 1]
        if seg[0] == "L":
            segs[i] = ("L", _shift(seg[1], d_end))
        else:
            _, c1, c2, end = seg
            segs[i] = ("C", _shift(c1, d_start), _shift(c2, d_end), _shift(end, d_end))
    return moved


def smooth_svg_contours(svg_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    """SVG'deki organik konturları Taubin ile yumuşatır (özellik korumalı).

    ``out_path`` verilmezse yerinde yazılır. Yalnız ``d`` yeniden yazılır;
    fill/transform/sıra korunur. A (yay) içeren path'ler atlanır. Dönen rapor:
    {"smoothed_paths", "anchors_moved"}; hata/uygun-değil durumunda 0.
    """
    from app.curve_fairing import _parse_subpaths, _serialize_subpaths  # noqa: PLC0415

    svg_path = Path(svg_path)
    dst = Path(out_path) if out_path is not None else svg_path
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"smoothed_paths": 0, "error": f"parse: {e}"}

    smoothed = 0
    moved_total = 0
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d:
            continue
        sps = _parse_subpaths(d)   # A içeren / mutlak-olmayan path -> None (atla)
        if not sps:
            continue
        pm = 0
        for sp in sps:
            try:
                pm += _smooth_subpath(sp)
            except Exception:  # noqa: BLE001
                continue
        if pm:
            el.set("d", _serialize_subpaths(sps))
            smoothed += 1
            moved_total += pm

    if smoothed == 0:
        return {"smoothed_paths": 0, "anchors_moved": 0}
    try:
        tree.write(str(dst), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"smoothed_paths": 0, "error": f"yazma: {e}"}
    return {"smoothed_paths": smoothed, "anchors_moved": moved_total}
