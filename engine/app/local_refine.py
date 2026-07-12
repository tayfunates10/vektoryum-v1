"""Kritik küçük bileşenler için kaynak-uzayında YEREL alt-piksel refit.

Küçük ama anlamlı bileşenler (® halkası, sayaçlı harfler, ince ikonlar)
global aday skorunda görünmez; izleme rasterinden (ör. 2200) gelen ~0.3 px
sapmalar küçük bileşende büyük oransal IoU kaybı yaratır. Bu modül kazanan
SVG (artık KAYNAK koordinat uzayında) üzerinde yalnız kritik bileşen
bölgelerini kaynağın anti-alias coverage'ına yeniden oturtur:

* DAİRE alt-yolları (yaylardan kurulu; boundary_refit yayları bilinçli
  dondurur): radyal alt-piksel kenar geçişlerinden analitik en-küçük-kareler
  daire oturtması. Merkez/yarıçap düzeltmesi ≤ 2 px ile sınırlıdır.
* GENEL alt-yollar (C/L): boundary_refit._snap_subpath, kaynak görüntüye
  to_px=1 ile doğrudan uygulanır (SVG kaynak uzayında olduğundan ölçek 1:1).

Kabul ÖLÇÜM KAPILIDIR: bileşen kırpımında kaynak-render sınıf uyumu
iyileşmezse o bileşenin tüm değişikliği geri alınır; kırpım dışında maddi
render farkı oluşursa da geri alınır. En çok _MAX_ROUNDS tur. Determinist:
sabit örnekleme, rastgelelik yok. ``VEKTORYUM_LOCAL_REFINE=off`` kapatır.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_MAX_ROUNDS = 3            # bileşen başına en çok refit turu (kesin sınır)
_MAX_COMPONENTS = 8        # tek görselde işlenecek en çok kritik küme
_MIN_IMPROVE = 0.0005      # kırpım uyum kazancı tabanı (altı gürültü)
_CIRCLE_MAX_ADJ_PX = 2.0   # daire merkez/yarıçap düzeltme sınırı
_CRIT_AREA_FRAC = 0.005    # bileşen alanı < tuvalin %0.5'i -> küçük
_CRIT_MIN_AREA = 60.0      # bundan küçüğü gürültü
_PAD_PX = 12               # bileşen kırpım/aidiyet payı

try:
    from svgpathtools import parse_path
except ImportError:  # pragma: no cover
    parse_path = None


def is_available() -> bool:
    return parse_path is not None


# ---------------------------------------------------------------------------
# Alt-piksel kenar örneklemesi (geniş pencere)
# ---------------------------------------------------------------------------
def _edge_cross(ref: np.ndarray, cx: float, cy: float, nx: float, ny: float,
                tmax: float = 2.5, step: float = 0.25) -> float | None:
    """(cx,cy)'den normal boyunca 0.5 coverage geçişinin alt-piksel ofseti.

    boundary_refit._edge_offset'in geniş-pencereli türevi; koordinatlar
    PİKSEL-İNDEKS uzayındadır (kullanıcı koordinatı x -> indeks x-0.5).
    """
    from app.boundary_refit import _bilinear  # noqa: PLC0415

    ts = np.arange(-tmax, tmax + 1e-9, step)
    prof = _bilinear(ref, cx + ts * nx, cy + ts * ny)
    side_a = prof[ts <= -tmax + 2 * step].mean(axis=0)
    side_b = prof[ts >= tmax - 2 * step].mean(axis=0)
    dirv = side_b - side_a
    if float(np.linalg.norm(dirv)) < 30.0:
        return None
    s = (prof - side_a) @ dirv / float(dirv @ dirv)
    c = s - 0.5
    crossings: list[float] = []
    for i in range(len(ts) - 1):
        if c[i] == 0.0:
            crossings.append(float(ts[i]))
        elif c[i] * c[i + 1] < 0:
            f = c[i] / (c[i] - c[i + 1])
            crossings.append(float(ts[i] + f * (ts[i + 1] - ts[i])))
    if len(crossings) != 1:
        return None
    return crossings[0]


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float] | None:
    """Kasa cebirsel daire oturtması: (cx, cy, r). Determinist."""
    if pts.shape[0] < 8:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _subpath_samples(d_sub: str, n: int = 128) -> np.ndarray | None:
    try:
        p = parse_path(d_sub)
        return np.array([[p.point(i / n).real, p.point(i / n).imag] for i in range(n)])
    except Exception:  # noqa: BLE001
        return None


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_circle_subpath(d_sub: str) -> tuple[float, float, float] | None:
    """Alt-yol yay tabanlı ve daireye oturuyorsa (cx, cy, r) döndürür."""
    if not re.search(r"[Aa]", d_sub):
        return None
    pts = _subpath_samples(d_sub)
    if pts is None:
        return None
    fit = _fit_circle(pts)
    if fit is None:
        return None
    cx, cy, r = fit
    if r < 3.0:
        return None
    resid = np.abs(np.linalg.norm(pts - np.array([cx, cy]), axis=1) - r)
    if float(resid.max()) > max(0.75, 0.03 * r):
        return None  # daire değil (elips/serbest yay): dokunma
    return fit


def _refit_circle_to_source(ref: np.ndarray, cx: float, cy: float, r: float,
                            k: int = 96) -> tuple[float, float, float] | None:
    """Kaynak coverage'ından radyal geçişlerle daireyi yeniden oturtur."""
    pts = []
    for i in range(k):
        ang = 2.0 * np.pi * i / k
        nx, ny = np.cos(ang), np.sin(ang)
        px = cx + r * nx - 0.5  # kullanıcı koordinatı -> piksel indeksi
        py = cy + r * ny - 0.5
        t = _edge_cross(ref, px, py, nx, ny)
        if t is None:
            continue
        pts.append([cx + (r + t) * nx, cy + (r + t) * ny])
    if len(pts) < int(0.6 * k):
        return None  # kenarın çoğunluğu ölçülemedi: güvenme
    fit = _fit_circle(np.array(pts))
    if fit is None:
        return None
    ncx, ncy, nr = fit
    if abs(ncx - cx) > _CIRCLE_MAX_ADJ_PX or abs(ncy - cy) > _CIRCLE_MAX_ADJ_PX \
            or abs(nr - r) > _CIRCLE_MAX_ADJ_PX:
        return None  # büyük sapma: bu bir daire düzeltmesi değil
    return fit


def _circle_d(cx: float, cy: float, r: float, clockwise: bool) -> str:
    sf = 1 if clockwise else 0
    return (f"M {cx + r:.2f},{cy:.2f} "
            f"A {r:.2f},{r:.2f} 0 1,{sf} {cx - r:.2f},{cy:.2f} "
            f"A {r:.2f},{r:.2f} 0 1,{sf} {cx + r:.2f},{cy:.2f} Z")


# ---------------------------------------------------------------------------
# Kritik bileşen tespiti
# ---------------------------------------------------------------------------
def _critical_clusters(src: np.ndarray, fills_rgb: np.ndarray,
                       width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Kaynaktan kritik küçük bileşen kümelerinin bbox listesi (determinist)."""
    from app.palette_ops import classify_rgb  # noqa: PLC0415

    cls = classify_rgb(src, fills_rgb)  # bant bazlı: bellek sınırlı, bit-birebir
    canvas = float(width * height)
    boxes: list[tuple[int, int, int, int, int]] = []  # (alan, x0,y0,x1,y1)
    for ci in range(fills_rgb.shape[0]):
        mask = (cls == ci).astype(np.uint8)
        if int(mask.sum()) < _CRIT_MIN_AREA:
            continue
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < _CRIT_MIN_AREA or area > _CRIT_AREA_FRAC * canvas:
                continue
            if max(ww, hh) > 0.2 * max(width, height):
                continue  # ince uzun büyük şerit: küçük bileşen değil
            boxes.append((int(area),
                          max(0, x - _PAD_PX), max(0, y - _PAD_PX),
                          min(width, x + ww + _PAD_PX), min(height, y + hh + _PAD_PX)))
    # örtüşen kutuları birleştir (® = halka+R+disk tek küme olmalı)
    boxes.sort(key=lambda b: (-b[0], b[1], b[2]))
    clusters: list[list[int]] = []
    for _a, x0, y0, x1, y1 in boxes:
        placed = False
        for cl in clusters:
            if not (x1 < cl[0] or x0 > cl[2] or y1 < cl[1] or y0 > cl[3]):
                cl[0], cl[1] = min(cl[0], x0), min(cl[1], y0)
                cl[2], cl[3] = max(cl[2], x1), max(cl[3], y1)
                placed = True
                break
        if not placed:
            clusters.append([x0, y0, x1, y1])
        if len(clusters) >= _MAX_COMPONENTS:
            break
    return [tuple(int(v) for v in c) for c in clusters]


# ---------------------------------------------------------------------------
# Ana giriş
# ---------------------------------------------------------------------------
def refine_critical_components(
    svg_path: Path,
    source_rgb: np.ndarray,
    width: int,
    height: int,
    render_fn: Callable[[Path, int, int], np.ndarray | None],
    cache: Any = None,
) -> dict[str, Any]:
    """Kritik küçük bileşenleri kaynak coverage'ına yerel olarak oturtur."""
    if not is_available():
        return {"status": "skipped", "reason": "svgpathtools yok"}
    from app.boundary_refit import (  # noqa: PLC0415
        _parse_subpaths_arc,
        _serialize_subpaths_arc,
        _snap_subpath,
    )

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

    hex_fills = sorted({
        (el.get("fill") or "").lower()
        for el in root.iter()
        if el.tag.split("}")[-1] == "path" and re.fullmatch(r"#[0-9a-f]{6}", (el.get("fill") or "").lower())
    })
    if not hex_fills:
        return {"status": "no_change", "reason": "hex dolgu yok"}
    fills_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                          for h in hex_fills], dtype=np.float32)

    clusters = _critical_clusters(source_rgb, fills_rgb, width, height)
    if not clusters:
        return {"status": "no_change", "reason": "kritik küçük bileşen yok"}

    ref = source_rgb.astype(np.float32)

    from app.palette_ops import classify_rgb  # noqa: PLC0415

    def classify(img: np.ndarray) -> np.ndarray:
        if cache is not None:
            return cache.classify(img, fills_rgb)
        return classify_rgb(img, fills_rgb)  # bant bazlı: bellek sınırlı, bit-birebir

    before_rgb = render_fn(svg_path, width, height)
    if before_rgb is None:
        return {"status": "skipped", "reason": "render backend yok"}
    src_cls = cache.classify_source(fills_rgb) if cache is not None else classify(source_rgb)

    # path bbox'ları (svgpathtools; yay bayrakları güvenli)
    path_els: list[ET.Element] = [el for el in root.iter()
                                  if el.tag.split("}")[-1] == "path" and el.get("d")]
    bboxes: dict[int, tuple[float, float, float, float]] = {}
    for el in path_els:
        try:
            x0, x1, y0, y1 = parse_path(el.get("d")).bbox()
            bboxes[id(el)] = (x0, y0, x1, y1)
        except Exception:  # noqa: BLE001
            continue

    tmp = svg_path.with_suffix(".lrefine.svg")
    report_comps: list[dict[str, Any]] = []
    total_changed = 0
    for cx0, cy0, cx1, cy1 in clusters:
        # kümedeki path'ler: bbox'ı küme kutusunun (paylı) içinde olanlar
        members = [el for el in path_els
                   if id(el) in bboxes
                   and bboxes[id(el)][0] >= cx0 - _PAD_PX and bboxes[id(el)][1] >= cy0 - _PAD_PX
                   and bboxes[id(el)][2] <= cx1 + _PAD_PX and bboxes[id(el)][3] <= cy1 + _PAD_PX]
        if not members:
            continue
        crop = (slice(cy0, cy1), slice(cx0, cx1))
        agree0 = float((src_cls[crop] == classify(before_rgb)[crop]).mean())
        comp_rep = {"bbox": [cx0, cy0, cx1 - cx0, cy1 - cy0], "paths": len(members),
                    "agree_before": round(agree0, 5), "rounds": 0, "applied": False}
        best_agree = agree0
        backup = {id(el): el.get("d") for el in members}
        cur_render = before_rgb
        for rnd_i in range(_MAX_ROUNDS):
            changed = 0
            for el in members:
                d = el.get("d")
                subpaths = _parse_subpaths_arc(d)
                if subpaths is None:
                    continue
                new_subs: list[str] = []
                el_changed = False
                for sp in subpaths:
                    sp_d = _serialize_subpaths_arc([sp])
                    circ = _is_circle_subpath(sp_d)
                    if circ is not None:
                        refit = _refit_circle_to_source(ref, *circ)
                        if refit is not None:
                            samples = _subpath_samples(sp_d)
                            cw = _shoelace(samples) > 0 if samples is not None else True
                            nd = _circle_d(*refit, clockwise=cw)
                            if nd != sp_d:
                                new_subs.append(nd)
                                el_changed = True
                                continue
                        new_subs.append(sp_d)
                        continue
                    moved = _snap_subpath(sp, ref, (1.0, 1.0), (1.0, 1.0))
                    if moved:
                        el_changed = True
                    new_subs.append(_serialize_subpaths_arc([sp]))
                if el_changed:
                    el.set("d", " ".join(new_subs))
                    changed += 1
            if changed == 0:
                break
            comp_rep["rounds"] = rnd_i + 1
            tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
            after_rgb = render_fn(tmp, width, height)
            if after_rgb is None:
                break
            # kırpım dışı maddi fark: sızıntı yok garantisi
            from app.palette_ops import abs_diff_sum  # noqa: PLC0415

            outside = abs_diff_sum(after_rgb, cur_render) > 30
            outside[crop] = False
            agree1 = float((src_cls[crop] == classify(after_rgb)[crop]).mean())
            if outside.any() or agree1 < best_agree + _MIN_IMPROVE:
                break  # bu tur kazanç yok: tur sonunda geri alınacak
            best_agree = agree1
            cur_render = after_rgb
            backup = {id(el): el.get("d") for el in members}  # kabul edilen durum
        # kabul edilen son duruma dön (son başarısız tur geri alınır)
        for el in members:
            if el.get("d") != backup[id(el)]:
                el.set("d", backup[id(el)])
        # --- 4x/8x süperörnekleme kaçışı: 1x turlar doyduğunda kırpım hâlâ
        # kusurluysa (uyum < 0.9995) kritik bileşen 4x yerel rasterde refit
        # edilir; 8x yalnız 4x anlamlı kazanç veremediyse denenir. Tüm görsel
        # asla büyütülmez; kapı 1x turlarla aynıdır. VEKTORYUM_SS_REFINE=off.
        import os as _os  # noqa: PLC0415

        ss_on = _os.environ.get("VEKTORYUM_SS_REFINE", "on").strip().lower() not in {
            "off", "0", "false"}
        if ss_on and best_agree < 0.9995:
            gain_4x = 0.0
            for scale in (4, 8):
                if scale == 8 and gain_4x >= 2 * _MIN_IMPROVE:
                    break  # 4x yeterli kazandı; 8x maliyetine gerek yok
                ss_changed = False
                for el in members:
                    nd = _snap_scaled(el.get("d"), ref, (cx0, cy0, cx1, cy1), scale)
                    if nd is not None and nd != el.get("d"):
                        el.set("d", nd)
                        ss_changed = True
                if not ss_changed:
                    continue
                tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
                after_rgb = render_fn(tmp, width, height)
                accept = False
                if after_rgb is not None:
                    from app.palette_ops import abs_diff_sum as _ads  # noqa: PLC0415

                    outside = _ads(after_rgb, cur_render) > 30
                    outside[crop] = False
                    agree_s = float((src_cls[crop] == classify(after_rgb)[crop]).mean())
                    # SS kapısı: kırpımda gerçek-pozitif kazanç yeter (1e-4 ≈
                    # 200² kırpımda ~9 px); ölçüldü: 4x, R IoU'yu 0.9691 ->
                    # 0.9723 taşırken uyum +0.00039 kazanıyor — _MIN_IMPROVE
                    # (5e-4) bu gerçek kazancı kıl payı reddediyordu
                    if not outside.any() and agree_s > best_agree + 1e-4:
                        accept = True
                if accept:
                    if scale == 4:
                        gain_4x = agree_s - best_agree
                    best_agree = agree_s
                    cur_render = after_rgb
                    backup = {id(el): el.get("d") for el in members}
                    comp_rep[f"ss{scale}x"] = round(agree_s, 5)
                else:
                    for el in members:
                        if el.get("d") != backup[id(el)]:
                            el.set("d", backup[id(el)])
        if best_agree > agree0 + 1e-9:
            comp_rep["applied"] = True
            total_changed += 1
            before_rgb = cur_render
        comp_rep["agree_after"] = round(best_agree, 5)
        report_comps.append(comp_rep)

    tmp.unlink(missing_ok=True)
    if total_changed:
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        status = "completed"
    else:
        status = "no_change"
    return {"status": status, "clusters": len(clusters),
            "improved": total_changed, "components": report_comps}


def _snap_scaled(d: str, ref: np.ndarray,
                 box: tuple[int, int, int, int], scale: int) -> str | None:
    """Path'i kırpım uzayında x{scale} büyütüp dar pencereyle snap eder.

    Kaynak kırpımı bikübik büyütülür (coverage modeli; palet etiketi İÇİN
    kullanılmaz — sınıflandırma her zaman 1x kaynakta kalır). Kenar arama
    penceresi ±1.6 büyütülmüş piksel = ±1.6/scale kaynak pikseli: 4x'te
    0.4 px, 8x'te 0.2 px etkin hassasiyet. Sonuç kaynak uzayına float
    hassasiyetle geri taşınır; yuvarlama canonical serileştirmede (0.01).
    Daire alt-yollarına dokunulmaz (analitik yol zaten alt-piksel).
    """
    from app.boundary_refit import (  # noqa: PLC0415
        _parse_subpaths_arc,
        _serialize_subpaths_arc,
        _snap_subpath,
    )

    x0, y0, x1, y1 = box
    try:
        p = parse_path(d)
        # Z kapanışları svgpathtools .d()'de düşer; alt-yol bazında korunur
        parts = []
        for sub in p.continuous_subpaths():
            s2 = sub.translated(complex(-x0, -y0)).scaled(scale)
            parts.append(_round_scaled_d(s2.d(), 4) + (" Z" if sub.isclosed() else ""))
        crop = ref[y0:y1, x0:x1].astype(np.uint8)
        ref_s = cv2.resize(crop, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC).astype(np.float32)
        subs = _parse_subpaths_arc(" ".join(parts))
        if subs is None:
            return None
        moved = 0
        for _ss_round in range(2):
            for sp in subs:
                sp_d = _serialize_subpaths_arc([sp])
                if _is_circle_subpath(sp_d) is not None:
                    continue  # daireler analitik yolda; SS snap uygulanmaz
                moved += _snap_subpath(sp, ref_s, (1.0, 1.0), (1.0, 1.0))
        if not moved:
            return None
        out_parts = []
        for sp in subs:
            sp_d = _serialize_subpaths_arc([sp])
            closed = sp_d.rstrip().endswith(("Z", "z"))
            b = parse_path(sp_d).scaled(1.0 / scale).translated(complex(x0, y0))
            out_parts.append(
                re.sub(r"-?\d+\.\d+",
                       lambda m: f"{float(m.group()):.2f}".rstrip("0").rstrip("."),
                       b.d()) + (" Z" if closed else ""))
        return " ".join(out_parts)
    except Exception:  # noqa: BLE001
        return None


def _round_scaled_d(d: str, nd: int) -> str:
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.{nd}f}", d)


# ---------------------------------------------------------------------------
# Hata-güdümlü render-and-refine (büyük bileşenlerdeki yerel sapmalar; ör. G)
# ---------------------------------------------------------------------------
_ERR_MIN_BLOB = 40         # maddi hata blob tabanı (px)
_ERR_MAX_REGIONS = 8       # işlenecek en çok hata bölgesi
_ERR_PAD = 12              # bölge kutusu payı (px)
_WIDE_TMAX = 4.0           # geniş kenar arama penceresi (px)
_WIDE_MAX_SHIFT = 2.5      # tur başına azami çapa kayması (px)
_WIDE_MIN_SHIFT = 0.05
_WIDE_REG = 0.25


def _snap_subpath_wide(
    sp: dict[str, Any], ref: np.ndarray,
    region: tuple[float, float, float, float] | None = None,
) -> int:
    """boundary_refit._snap_subpath'in GENİŞ pencereli türevi (kaynak uzay 1:1).

    İz-uzayı refit'i ±1.6 px pencereyle örnekler; 2-6 px'lik yerel sapmalarda
    (G iç gövdesi ölçümü) geçiş pencere DIŞINDA kalır ve hiçbir çapa desteği
    oluşmaz. Burada pencere ±4 px, kayma sınırı 2.5 px/turdur. Yay uçları yine
    donuktur (daireler analitik yolda düzeltilir).

    ``region`` (x0,y0,x1,y1) verilirse örnekleme ve çapa hareketleri bu
    kutuyla sınırlanır: kutu dışındaki çapalar YERİNDE kalır (cusp_refine'ın
    blob-kapsamlı kullanımı; bölge dışı render değişimi seam korumasına
    takılıyordu — ölçüldü).
    """
    from app.boundary_refit import _anchor_weights, _seg_point_tangent  # noqa: PLC0415

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

    sample_segs: list[tuple[tuple[float, float], tuple, int, int]] = []
    for i, seg in enumerate(sp["segs"]):
        sample_segs.append((pts[i], seg, _aidx(i), _aidx(i + 1)))
    if closed and not dup_last:
        sample_segs.append((pts[-1], ("L", pts[0]), _aidx(n_pts - 1), 0))

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for p0, seg, a_start, a_end in sample_segs:
        for u in (0.12, 0.3, 0.5, 0.7, 0.88):
            pt, tg = _seg_point_tangent(p0, seg, u)
            norm = (tg[0] * tg[0] + tg[1] * tg[1]) ** 0.5
            if norm < 1e-9:
                continue
            if region is not None and not (
                region[0] <= pt[0] <= region[2] and region[1] <= pt[1] <= region[3]
            ):
                continue
            nx, ny = -tg[1] / norm, tg[0] / norm
            t = _edge_cross(ref, pt[0] - 0.5, pt[1] - 0.5, nx, ny,
                            tmax=_WIDE_TMAX, step=0.4)
            if t is None or abs(t) < _WIDE_MIN_SHIFT:
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
    reg = np.eye(2 * n_anchor) * _WIDE_REG
    if not closed:
        reg[0, 0] = reg[1, 1] = 1e3
        reg[-2, -2] = reg[-1, -1] = 1e3
    for i, seg in enumerate(sp["segs"]):
        if seg[0] == "A":
            for ai in (_aidx(i), _aidx(i + 1)):
                reg[2 * ai, 2 * ai] = reg[2 * ai + 1, 2 * ai + 1] = 1e3
    sol, *_ = np.linalg.lstsq(np.vstack([a, reg]),
                              np.concatenate([b, np.zeros(2 * n_anchor)]), rcond=None)
    d_px = sol.reshape(n_anchor, 2)
    mag = np.linalg.norm(d_px, axis=1)
    over = mag > _WIDE_MAX_SHIFT
    d_px[over] *= (_WIDE_MAX_SHIFT / mag[over])[:, None]

    deltas: list[tuple[float, float] | None] = [None] * n_pts
    moved = 0
    for k in range(n_anchor):
        dx, dy = d_px[k]
        if (dx * dx + dy * dy) ** 0.5 < _WIDE_MIN_SHIFT:
            continue
        if region is not None:
            ax, ay = pts[k]
            if not (region[0] <= ax <= region[2] and region[1] <= ay <= region[3]):
                continue  # bölge dışı çapa yerinde kalır
        deltas[k] = (float(dx), float(dy))
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
        elif seg[0] == "A":
            if d_end is not None:
                _, rx, ry, xrot, laf, sf, end = seg
                sp["segs"][i] = ("A", rx, ry, xrot, laf, sf, _shift(end, d_end))
        else:
            _, c1, c2, end = seg
            if d_start is not None:
                c1 = _shift(c1, d_start)
            if d_end is not None:
                c2 = _shift(c2, d_end)
                end = _shift(end, d_end)
            sp["segs"][i] = ("C", c1, c2, end)
    return moved


def refine_error_regions(
    svg_path: Path,
    source_rgb: np.ndarray,
    width: int,
    height: int,
    render_fn: Callable[[Path, int, int], np.ndarray | None],
    cache: Any = None,
) -> dict[str, Any]:
    """Kaynak-render sınıf uyuşmazlığı bloblarını yerel çapa-kaydırma ile azaltır.

    Kazanan SVG render edilir, palet sınıfı uyuşmazlık haritası çıkarılır
    (3x3 açma ile AA çizgileri elenir), blob bölgeleriyle kesişen path'lerin
    çapaları kaynak coverage'ına oturtulur. Kabul kapıları: toplam maddi hata
    AZALMALI ve bölgeler dışında YENİ hata doğmamalı (seam koruması); aksi tur
    geri alınır. En çok _MAX_ROUNDS tur; determinist.
    """
    if not is_available():
        return {"status": "skipped", "reason": "svgpathtools yok"}
    from app.boundary_refit import (  # noqa: PLC0415
        _parse_subpaths_arc,
        _serialize_subpaths_arc,
    )

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

    hex_fills = sorted({
        (el.get("fill") or "").lower()
        for el in root.iter()
        if el.tag.split("}")[-1] == "path" and re.fullmatch(r"#[0-9a-f]{6}", (el.get("fill") or "").lower())
    })
    if not hex_fills:
        return {"status": "no_change", "reason": "hex dolgu yok"}
    fills_rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                          for h in hex_fills], dtype=np.float32)

    from app.palette_ops import classify_rgb  # noqa: PLC0415

    def classify(img: np.ndarray) -> np.ndarray:
        if cache is not None:
            return cache.classify(img, fills_rgb)
        return classify_rgb(img, fills_rgb)  # bant bazlı: bellek sınırlı, bit-birebir

    def err_mask(rnd: np.ndarray) -> np.ndarray:
        e = (src_cls != classify(rnd)).astype(np.uint8)
        return cv2.morphologyEx(e, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    src_cls = cache.classify_source(fills_rgb) if cache is not None else classify(source_rgb)
    cur = render_fn(svg_path, width, height)
    if cur is None:
        return {"status": "skipped", "reason": "render backend yok"}
    e0 = err_mask(cur)
    err_before = int(e0.sum())
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(e0, 8)
    regions: list[list[int]] = []
    blob_boxes = sorted(
        ((int(a), int(x), int(y), int(ww), int(hh))
         for x, y, ww, hh, a in (stats[i] for i in range(1, n)) if a >= _ERR_MIN_BLOB),
        reverse=True,
    )
    if not blob_boxes:
        return {"status": "no_change", "reason": "maddi hata blobu yok",
                "err_px": err_before}
    for _a, x, y, ww, hh in blob_boxes:
        x0, y0 = max(0, x - _ERR_PAD), max(0, y - _ERR_PAD)
        x1, y1 = min(width, x + ww + _ERR_PAD), min(height, y + hh + _ERR_PAD)
        placed = False
        for r in regions:
            if not (x1 < r[0] or x0 > r[2] or y1 < r[1] or y0 > r[3]):
                r[0], r[1] = min(r[0], x0), min(r[1], y0)
                r[2], r[3] = max(r[2], x1), max(r[3], y1)
                placed = True
                break
        if not placed:
            regions.append([x0, y0, x1, y1])
        if len(regions) >= _ERR_MAX_REGIONS:
            break
    region_mask = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in regions:
        region_mask[y0:y1, x0:x1] = 1
    out_before = int((e0 & (region_mask == 0)).sum())

    ref = source_rgb.astype(np.float32)
    path_els = [el for el in root.iter()
                if el.tag.split("}")[-1] == "path" and el.get("d")]

    def _intersects(bb: tuple[float, float, float, float]) -> bool:
        return any(not (bb[2] < r[0] or bb[0] > r[2] or bb[3] < r[1] or bb[1] > r[3])
                   for r in regions)

    members = []
    for el in path_els:
        try:
            x0, x1, y0, y1 = parse_path(el.get("d")).bbox()
        except Exception:  # noqa: BLE001
            continue
        if _intersects((x0, y0, x1, y1)):
            members.append(el)
    if not members:
        return {"status": "no_change", "reason": "bölgeyle kesişen path yok",
                "err_px": err_before}

    tmp = svg_path.with_suffix(".erefine.svg")
    backup = {id(el): el.get("d") for el in members}
    best_err = err_before
    rounds_done = 0
    for _round in range(_MAX_ROUNDS):
        changed = 0
        for el in members:
            subpaths = _parse_subpaths_arc(el.get("d"))
            if subpaths is None:
                continue
            moved = 0
            for sp in subpaths:
                moved += _snap_subpath_wide(sp, ref)
            if moved:
                el.set("d", _serialize_subpaths_arc(subpaths))
                changed += 1
        if changed == 0:
            break
        tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
        after = render_fn(tmp, width, height)
        if after is None:
            break
        e1 = err_mask(after)
        err_new = int(e1.sum())
        out_new = int((e1 & (region_mask == 0)).sum())
        # kapılar: toplam maddi hata anlamlı azalmalı; bölge dışına yeni hata
        # taşmamalı (komşu sınırlarda seam doğurma koruması); hiçbir bölgede
        # MAKS sınır sapması derinleşmemeli (dar kama tepesinde snap piksel
        # sayısını düşürürken sahte kenar açabiliyordu — ölçüldü: 7.8px çentik)
        if err_new > best_err - _ERR_MIN_BLOB or out_new > out_before + 8:
            break  # kazanç yok: tur sonunda geri alınır
        after_cls = classify(after)
        cur_cls = classify(cur)
        deepened = False
        for rx0, ry0, rx1, ry1 in regions:
            from app.cusp_refine import _crop_max_dev  # noqa: PLC0415

            m0 = _crop_max_dev(src_cls[ry0:ry1, rx0:rx1], cur_cls[ry0:ry1, rx0:rx1])
            m1 = _crop_max_dev(src_cls[ry0:ry1, rx0:rx1], after_cls[ry0:ry1, rx0:rx1])
            # yalnız MADDİ derinleşme veto eder (>0.5px artış VE >3px mutlak):
            # AA ölçekli titreşimi veto etmek iyi turları topluca kesip genel
            # sınıf IoU'sunu geriletiyordu (ölçüldü: sarı 0.9944 -> 0.9924)
            if m1 > m0 + 0.5 and m1 > 3.0:
                deepened = True
                break
        if deepened:
            break  # tur sonunda geri alınır
        rounds_done += 1
        best_err = err_new
        cur = after
        backup = {id(el): el.get("d") for el in members}
    for el in members:
        if el.get("d") != backup[id(el)]:
            el.set("d", backup[id(el)])
    tmp.unlink(missing_ok=True)
    if rounds_done:
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        status = "completed"
    else:
        status = "no_change"
    return {"status": status, "rounds": rounds_done, "regions": len(regions),
            "paths": len(members), "err_px_before": err_before, "err_px_after": best_err,
            "outside_before": out_before}
