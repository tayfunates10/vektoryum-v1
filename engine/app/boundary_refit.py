"""SVG path sınırlarını orijinal görüntünün alt-piksel kenarlarına oturtma.

Tavan analizi + faz-korelasyonu ölçümü, renk refit sonrası kalan izleme
kaybının GLOBAL değil KENAR-BAŞINA YEREL yarım-piksel sapmalardan geldiğini
gösterdi: izleyici bölge sınırını kuantize ızgaraya yerleştirir; oysa
orijinaldeki anti-alias rampası gerçek kenarın alt-piksel konumunu kodlar
(klasik alt-piksel kenar lokalizasyonu — profil orta-nokta geçişi).

FORMÜLASYON — kenar-örneklemeli en küçük kareler (DiffVG geometri adımının
tek Gauss-Newton iterasyonu, kapalı formda): her segment üzerinde çok noktada
yerel normal boyunca alt-piksel kenar ofseti ölçülür; parametre u'daki örnek,
segmentin iki uç çapasının ağırlıklı düzeltme kombinasyonunu kısıtlar
(L: ağırlıklar 1-u/u; C: Bernstein grupları — c1 baş çapayla, c2 uç çapayla
sürüklenir varsayımıyla wA=(1-u)²(1+2u), wB=u²(3-2u)). Alt yol başına
2n bilinmeyenli (çapa başına dx,dy) regülarize LSQ çözülür. Çapa-başına
doğrudan oturtma köşelerde yanılır (açıortay normali + köşe AA profili
doğrusal değildir); kenar-içi örnekler köşe çapalarını iki kenarın kesişimi
olarak DOĞRU konuma çeker — ölçülen sentetik doğrulama bunu gösterdi.

Kurallar:
* Yalnız mutlak M/L/C/Z path'leri işlenir (curve_fairing parser'ı); başka
  komut içeren alt yol aynen korunur.
* transform kapsamındaki path'lere dokunulmaz (çift dönüşüm riski).
* Yaka kontrastı düşükse (düz bölge, örtülen sınır) ya da profili birden çok
  kez kesiyorsa (paralel komşu kenar) örnek atılır; çözüm ``_MAX_SHIFT_PX``
  ile kısıtlanır; açık alt yol uçları sabittir.
* Benimseme kararı ÇAĞIRANDA: modül yalnız yeni SVG'yi yazar; pipeline ölçülen
  fidelity artmadıysa eski çıktıyı korur (renk refit ile aynı sözleşme).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_MAX_SHIFT_PX = 1.0     # karşılaştırma pikselinde azami çapa kayması
_MIN_SHIFT_PX = 0.05    # bundan küçük kayma gürültüdür
_MIN_CONTRAST = 30.0    # iki yaka arasındaki asgari RGB farkı (kenar var mı?)
_PROFILE_T = np.arange(-1.6, 1.61, 0.2)  # normal boyunca örnekleme ofsetleri (px)


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 görüntüden bilinear örnekleme; koordinatlar kenara kilitlenir."""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    p00 = img[y0, x0]
    p01 = img[y0, x0 + 1]
    p10 = img[y0 + 1, x0]
    p11 = img[y0 + 1, x0 + 1]
    return p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy) + p10 * (1 - fx) * fy + p11 * fx * fy


def _edge_offset(ref: np.ndarray, cx: float, cy: float, nx: float, ny: float) -> float | None:
    """(cx,cy) çevresinde normal (nx,ny) boyunca alt-piksel kenar ofseti.

    Profilin iki ucundaki yaka renkleri arasındaki 0.5 projeksiyon geçişini
    arar. Kenar yoksa / belirsizse (0 ya da 2+ geçiş) None döner.
    """
    xs = cx + _PROFILE_T * nx
    ys = cy + _PROFILE_T * ny
    prof = _bilinear(ref, xs, ys)  # (17, 3)
    side_a = prof[_PROFILE_T <= -1.2].mean(axis=0)
    side_b = prof[_PROFILE_T >= 1.2].mean(axis=0)
    dirv = side_b - side_a
    contrast = float(np.linalg.norm(dirv))
    if contrast < _MIN_CONTRAST:
        return None
    s = (prof - side_a) @ dirv / float(dirv @ dirv)  # 0 (A yakası) -> 1 (B yakası)
    c = s - 0.5
    crossings: list[float] = []
    for i in range(len(_PROFILE_T) - 1):
        if c[i] == 0.0:
            crossings.append(float(_PROFILE_T[i]))
        elif c[i] * c[i + 1] < 0:
            f = c[i] / (c[i] - c[i + 1])
            crossings.append(float(_PROFILE_T[i] + f * (_PROFILE_T[i + 1] - _PROFILE_T[i])))
    crossings = [t for t in crossings if abs(t) <= _MAX_SHIFT_PX]
    if len(crossings) != 1:
        return None  # kenar yok ya da paralel komşu kenar belirsizliği
    t = crossings[0]
    if abs(t) < _MIN_SHIFT_PX:
        return None
    return t


_SAMPLE_U = (0.12, 0.3, 0.5, 0.7, 0.88)  # segment-içi örnekleme parametreleri
_REG_LAMBDA = 0.25                        # LSQ regülarizasyonu (küçük düzeltme tercihi)


def _seg_point_tangent(
    p0: tuple[float, float], seg: tuple, u: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Segment üzerinde u'daki nokta ve teğet (L: doğru; C: kübik Bezier)."""
    if seg[0] == "L":
        p1 = seg[1]
        pt = (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
        return pt, (p1[0] - p0[0], p1[1] - p0[1])
    _, c1, c2, p3 = seg
    v = 1.0 - u
    pt = (
        v * v * v * p0[0] + 3 * v * v * u * c1[0] + 3 * v * u * u * c2[0] + u * u * u * p3[0],
        v * v * v * p0[1] + 3 * v * v * u * c1[1] + 3 * v * u * u * c2[1] + u * u * u * p3[1],
    )
    tg = (
        3 * v * v * (c1[0] - p0[0]) + 6 * v * u * (c2[0] - c1[0]) + 3 * u * u * (p3[0] - c2[0]),
        3 * v * v * (c1[1] - p0[1]) + 6 * v * u * (c2[1] - c1[1]) + 3 * u * u * (p3[1] - c2[1]),
    )
    return pt, tg


def _anchor_weights(seg: tuple, u: float) -> tuple[float, float]:
    """u'daki örneğin baş/uç çapa düzeltmelerine duyarlılığı.

    C segmentinde c1 baş çapayla, c2 uç çapayla sürüklenir varsayılır
    (uygulama da böyle taşır): wA=(1-u)²(1+2u), wB=u²(3-2u). L: 1-u / u.
    """
    if seg[0] == "L":
        return 1.0 - u, u
    v = 1.0 - u
    return v * v * (1.0 + 2.0 * u), u * u * (3.0 - 2.0 * u)


def _snap_subpath(
    sp: dict[str, Any],
    ref: np.ndarray,
    to_px: tuple[float, float],
    to_user: tuple[float, float],
    offset: tuple[float, float] = (0.0, 0.0),
) -> int:
    """Alt yolun çapa düzeltmelerini kenar-örneklemeli LSQ ile çözer ve uygular.

    Döner: taşınan çapa sayısı. Örnek kısıtı: n·(wA·dA + wB·dB) = t
    (t = alt-piksel kenar ofseti, karşılaştırma pikselinde; d'ler de piksel
    uzayında çözülür, uygulanırken kullanıcı uzayına ölçeklenir).
    ``offset``: path'in kendi translate transform'u — örnekleme belge uzayında
    yapılır, çözülen delta translate altında değişmediğinden path uzayında
    uygulanır.
    """
    pts: list[tuple[float, float]] = [sp["start"]]
    for seg in sp["segs"]:
        pts.append(seg[-1])
    n_pts = len(pts)
    if n_pts < 2:
        return 0
    closed = bool(sp["closed"]) and n_pts >= 3
    dup_last = closed and abs(pts[-1][0] - pts[0][0]) < 1e-6 and abs(pts[-1][1] - pts[0][1]) < 1e-6
    n_anchor = n_pts - 1 if dup_last else n_pts

    def _aidx(i: int) -> int:
        return i % n_anchor if dup_last else i

    # örneklenecek segmentler: gerçek segmentler + (kapalı ve son nokta start'a
    # eşit değilse) Z'nin ÖRTÜK kapanış kenarı — aksi halde o kenar hiç ölçülmez
    # ve düzeltilmez (sentetik doğrulamada dikdörtgenin sol kenarı böyle kaçtı)
    sample_segs: list[tuple[int, tuple[float, float], tuple, int, int]] = []
    for i, seg in enumerate(sp["segs"]):
        sample_segs.append((i, pts[i], seg, _aidx(i), _aidx(i + 1)))
    if closed and not dup_last:
        sample_segs.append((len(sp["segs"]), pts[-1], ("L", pts[0]), _aidx(n_pts - 1), 0))

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for _i, p0, seg, a_start, a_end in sample_segs:
        for u in _SAMPLE_U:
            pt, tg = _seg_point_tangent(p0, seg, u)
            # teğet/normal karşılaştırma-piksel uzayında (anizotropik ölçek olasılığına karşı)
            tx, ty = tg[0] * to_px[0], tg[1] * to_px[1]
            norm = (tx * tx + ty * ty) ** 0.5
            if norm < 1e-9:
                continue
            nx, ny = -ty / norm, tx / norm
            # piksel-merkezi konvansiyonu: sürekli koordinat c, piksel indeksinde
            # c*scale - 0.5'e düşer (indeks i'nin merkezi i+0.5'tir). Bu yarım
            # piksel atlanırsa tüm ofsetler sistematik -0.5 önyargı alır.
            cx = (pt[0] + offset[0]) * to_px[0] - 0.5
            cy = (pt[1] + offset[1]) * to_px[1] - 0.5
            t = _edge_offset(ref, cx, cy, nx, ny)
            if t is None:
                continue
            wa, wb = _anchor_weights(seg, u)
            row = np.zeros(2 * n_anchor, np.float64)
            row[2 * a_start] += wa * nx
            row[2 * a_start + 1] += wa * ny
            row[2 * a_end] += wb * nx
            row[2 * a_end + 1] += wb * ny
            rows.append(row)
            rhs.append(t)
    if len(rows) < 3:
        return 0

    a = np.array(rows)
    b = np.array(rhs)
    # regülarizasyon: desteklenmeyen çapalar yerinde kalsın; açık yol uçları sabit
    reg = np.eye(2 * n_anchor) * _REG_LAMBDA
    if not closed:
        reg[0, 0] = reg[1, 1] = 1e3
        reg[-2, -2] = reg[-1, -1] = 1e3
    a_full = np.vstack([a, reg])
    b_full = np.concatenate([b, np.zeros(2 * n_anchor)])
    sol, *_ = np.linalg.lstsq(a_full, b_full, rcond=None)
    d_px = sol.reshape(n_anchor, 2)
    # kayma sınırı (px)
    mag = np.linalg.norm(d_px, axis=1)
    over = mag > _MAX_SHIFT_PX
    d_px[over] *= (_MAX_SHIFT_PX / mag[over])[:, None]

    deltas: list[tuple[float, float] | None] = [None] * n_pts
    moved = 0
    for k in range(n_anchor):
        dx, dy = d_px[k]
        if (dx * dx + dy * dy) ** 0.5 < _MIN_SHIFT_PX:
            continue
        deltas[k] = (dx * to_user[0], dy * to_user[1])
        moved += 1
    if dup_last and deltas[0] is not None:
        deltas[n_pts - 1] = deltas[0]
    if moved == 0:
        return 0

    def _shift(p: tuple[float, float], d: tuple[float, float]) -> tuple[float, float]:
        return (p[0] + d[0], p[1] + d[1])

    if deltas[0] is not None:
        sp["start"] = _shift(sp["start"], deltas[0])
    for i, seg in enumerate(sp["segs"]):
        d_start, d_end = deltas[i], deltas[i + 1]
        if seg[0] == "L":
            if d_end is not None:
                sp["segs"][i] = ("L", _shift(seg[1], d_end))
        else:
            _, c1, c2, end = seg
            if d_start is not None:
                c1 = _shift(c1, d_start)  # c1 baş çapayla sürüklenir
            if d_end is not None:
                c2 = _shift(c2, d_end)    # c2 uç çapayla sürüklenir
                end = _shift(end, d_end)
            sp["segs"][i] = ("C", c1, c2, end)
    return moved


def refit_svg_boundaries(
    svg_path: Path,
    original_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """SVG path çapalarını orijinalin alt-piksel kenarlarına oturtup out_path'e yazar.

    Dönen rapor: {"moved": taşınan çapa, "anchors": bakılan çapa, ...};
    başarısızlıkta {"moved": 0, "error": ...} (çökme yok). Benimseme kararı
    çağırana aittir (ölçülen fidelity artmalı).
    """
    from app.curve_fairing import _parse_subpaths, _serialize_subpaths  # noqa: PLC0415
    from app.fidelity import load_reference_rgb  # noqa: PLC0415

    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
    except Exception as e:  # noqa: BLE001
        return {"moved": 0, "error": f"parse: {e}"}
    root = tree.getroot()

    try:
        ref_u8, (w, h) = load_reference_rgb(Path(original_path))
    except Exception as e:  # noqa: BLE001
        return {"moved": 0, "error": f"referans: {e}"}
    ref = ref_u8.astype(np.float32)

    # kullanıcı uzayı <-> karşılaştırma pikseli ölçekleri (color_refit ile aynı)
    vb = root.get("viewBox")
    if vb:
        try:
            _, _, vbw, vbh = (float(x) for x in vb.replace(",", " ").split())
        except ValueError:
            vbw, vbh = float(w), float(h)
    else:
        try:
            vbw = float(re.sub(r"[a-z%]+$", "", root.get("width", str(w))))
            vbh = float(re.sub(r"[a-z%]+$", "", root.get("height", str(h))))
        except ValueError:
            vbw, vbh = float(w), float(h)
    to_px = (float(w) / vbw, float(h) / vbh)
    to_user = (vbw / float(w), vbh / float(h))

    # EBEVEYN (grup) transform'u taşıyan path'lere dokunma; path'in KENDİ
    # transform'u yalnız translate ise desteklenir: örnekleme koordinatına ofset
    # eklenir, çözülen delta translate altında değişmeden path uzayında uygulanır
    # (vtracer path'leri translate taşır — bunlar dondurulursa hiçbir renkli
    # aday oturtulamaz).
    frozen: set[int] = set()

    def _mark(el: ET.Element, ancestor_has_xf: bool) -> None:
        if ancestor_has_xf and el.tag.split("}")[-1] == "path":
            frozen.add(id(el))
        has = ancestor_has_xf or (el.get("transform") is not None)
        for ch in list(el):
            _mark(ch, has)

    # kökün kendi transform'u da alt path'leri etkiler; path'in KENDİ transform'u
    # ise yalnız alt öğelerini etkilerdi (path'in path çocuğu olmaz -> sorun yok)
    for child in list(root):
        _mark(child, root.get("transform") is not None)

    _TRANSLATE_RE = re.compile(
        r"^\s*translate\(\s*([-+0-9.eE]+)(?:[\s,]+([-+0-9.eE]+))?\s*\)\s*$"
    )

    moved = 0
    anchors = 0
    paths_changed = 0
    for el in root.iter():
        if el.tag.split("}")[-1] != "path" or id(el) in frozen:
            continue
        xf = el.get("transform")
        offset = (0.0, 0.0)
        if xf is not None:
            m = _TRANSLATE_RE.match(xf)
            if m is None:
                continue  # translate dışı transform: dokunma
            offset = (float(m.group(1)), float(m.group(2) or 0.0))
        d = el.get("d")
        if not d:
            continue
        subpaths = _parse_subpaths(d)
        if subpaths is None:
            continue  # desteklenmeyen komut (ör. A yayı): aynen bırak
        p_moved = 0
        for sp in subpaths:
            anchors += len(sp["segs"]) + 1
            p_moved += _snap_subpath(sp, ref, to_px, to_user, offset)
        if p_moved:
            el.set("d", _serialize_subpaths(subpaths))
            moved += p_moved
            paths_changed += 1

    if moved == 0:
        return {"moved": 0, "anchors": anchors}
    try:
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"moved": 0, "error": f"yazma: {e}"}
    return {"moved": moved, "anchors": anchors, "paths_changed": paths_changed}
