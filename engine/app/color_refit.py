"""SVG dolgu renklerini orijinal görüntüye yeniden oturtma (color refit).

Tavan analizi bulgusu: izleme sonrası kaybın en büyük bileşeni RENK — vtracer
bölge rengini kendi kuantize katman merkezinden alır, palet konsolidasyonu da
yakın tonları birleştirir; sonuç orijinalden ΔE 3-5 uzağa kayabilir. Dolgular
SABİT renk olduğundan bu, DiffVG-tarzı türevlenebilir vektör optimizasyonunun
renk adımının KAPALI FORMDA çözülebildiği özel haldir: her path'in görünür
bölgesi üzerinde en küçük kareler çözümü, orijinal piksellerin ortalamasıdır
(aykırı değerlere karşı medyan kullanırız). Gradyan uzanımı da aynı bölge
üzerinde doğrusal model c(x,y) = c0 + gx·x + gy·y en küçük kareler oturtmasıdır
("Segmentation-guided Layer-wise Image Vectorization with Gradient Fills"
yaklaşımının bölge-bazlı hali).

Görünür bölge tespiti ID-RENDER ile yapılır: her path'e benzersiz bir ID rengi
verilip belge bir kez render edilir; piksel başına en üstteki path birebir
okunur (painter's algorithm'i yeniden kurmaya gerek kalmaz, transform/fill-rule
dahil gerçek render semantiği geçerlidir). Anti-alias karışım pikselleri hiçbir
ID'ye denk gelmez ve 1px erozyonla zaten dışlanır.

Benimseme kararı ÇAĞIRANDA: bu modül yalnız yeni SVG'yi yazar ve rapor döner;
pipeline ölçülen fidelity artmadıysa eski çıktıyı korur (geri alınabilirlik).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_MIN_REGION_PX = 25          # bundan az görünür pikseli olan path'e dokunma
_MIN_SHIFT_DE = 0.5          # bundan küçük düzeltme gürültüdür, uygulama
_MAX_SHIFT_DE = 30.0         # bundan büyük sapma şüphelidir (ID/maske hatası)
_NAMED = {"black": "#000000", "white": "#ffffff", "red": "#ff0000"}

# Gradyan uzanımı (isteğe bağlı, refit_svg_colors(gradients=True)):
_GRAD_MIN_PX = 900           # doğrusal model ancak yeterince büyük bölgede anlamlı
_GRAD_MIN_GAIN_DE = 1.2     # sabit renge göre ortalama ΔE bu kadar düşmeli
_GRAD_MIN_SPAN_DE = 3.0     # iki uç stop arasındaki fark algılanabilir olmalı


def _parse_fill(value: str | None) -> tuple[int, int, int] | None:
    """fill değerini RGB'ye çevirir; url()/none/tanınmayan için None."""
    if value is None:
        return (0, 0, 0)  # SVG varsayılanı: siyah
    v = value.strip().lower()
    v = _NAMED.get(v, v)
    if v.startswith("#"):
        if len(v) == 7:
            try:
                return (int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16))
            except ValueError:
                return None
        if len(v) == 4:
            try:
                return (int(v[1] * 2, 16), int(v[2] * 2, 16), int(v[3] * 2, 16))
            except ValueError:
                return None
        return None
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", v)
    if m:
        return tuple(min(255, int(g)) for g in m.groups())  # type: ignore[return-value]
    return None


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) uint8 RGB -> (N,3) float32 LAB."""
    arr = rgb.reshape(1, -1, 3).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)


def _delta_e(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la = _lab(np.array([a], dtype=np.uint8))[0]
    lb = _lab(np.array([b], dtype=np.uint8))[0]
    return float(np.linalg.norm(la - lb))


def _iter_paths(root: ET.Element) -> list[ET.Element]:
    return [el for el in root.iter() if el.tag.split("}")[-1] == "path"]


def _elem_opaque(el: ET.Element) -> bool:
    for attr in ("opacity", "fill-opacity"):
        v = el.get(attr)
        if v is not None:
            try:
                if float(v) < 0.999:
                    return False
            except ValueError:
                return False
    return True


def _build_id_svg(tree: ET.ElementTree) -> tuple[ET.ElementTree, list[ET.Element]]:
    """Her path'e benzersiz ID rengi atanmış bir kopya ağaç üretir.

    ID rengi = (indeks+1) 24-bit kodlaması; 0xFFFFFF (beyaz zemin) atlanır.
    Konturlu path'lerin stroke'u da kendi ID rengine boyanır ki örtme
    (occlusion) gerçek render ile aynı kalsın.
    """
    import copy

    id_tree = copy.deepcopy(tree)
    src_paths = _iter_paths(tree.getroot())
    id_paths = _iter_paths(id_tree.getroot())
    for i, el in enumerate(id_paths):
        code = i + 1
        if code >= 0xFFFFFF:
            code += 1  # beyazla çakışma (pratikte erişilmez)
        color = _hex(((code >> 16) & 255, (code >> 8) & 255, code & 255))
        el.set("fill", color)
        if el.get("stroke") not in (None, "none"):
            el.set("stroke", color)
        # stil özniteliği fill/stroke içeriyorsa ID rengini ezmesin
        style = el.get("style")
        if style:
            style = re.sub(r"(?:fill|stroke)\s*:\s*[^;]+;?", "", style).strip()
            if style:
                el.set("style", style)
            else:
                del el.attrib["style"]
    return id_tree, src_paths


def _decode_id_map(id_rgb: np.ndarray) -> np.ndarray:
    return (
        id_rgb[:, :, 0].astype(np.int32) << 16
    ) | (id_rgb[:, :, 1].astype(np.int32) << 8) | id_rgb[:, :, 2].astype(np.int32)


def _fit_linear_gradient(
    ys: np.ndarray, xs: np.ndarray, pix: np.ndarray, base_rgb: tuple[int, int, int]
) -> dict[str, Any] | None:
    """Bölge pikselleri üzerinde c(x,y) = c0 + gx·x + gy·y doğrusal LSQ modeli.

    Sabit renge göre ortalama ΔE kazancı yeterliyse iki-stop'lu doğrusal
    gradyan tanımı döner (x1,y1,x2,y2 kullanıcı uzayında; renkler uint8 RGB).
    """
    n = len(xs)
    if n < _GRAD_MIN_PX:
        return None
    a = np.stack([np.ones(n, np.float64), xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    tgt = pix.astype(np.float64)
    coef, *_ = np.linalg.lstsq(a, tgt, rcond=None)  # (3 param, 3 kanal)
    pred = a @ coef

    lab_t = _lab(pix)
    lab_c = _lab(np.tile(np.array(base_rgb, np.uint8), (1, 1)))[0]
    de_const = float(np.mean(np.linalg.norm(lab_t - lab_c[None, :], axis=1)))
    lab_p = _lab(np.clip(pred, 0, 255).astype(np.uint8))
    de_grad = float(np.mean(np.linalg.norm(lab_t - lab_p, axis=1)))
    if de_const - de_grad < _GRAD_MIN_GAIN_DE:
        return None

    # gradyan yönü: kanal eğimlerinin en büyük varyanslı birleşimi (PCA yerine
    # basitçe L* eğimi — algısal olarak baskın eksen)
    g = coef[1:, :]  # (2, 3) [d/dx; d/dy] her kanal
    # luminance ağırlıkları (Rec.601)
    w = np.array([0.299, 0.587, 0.114])
    gx, gy = float(g[0] @ w), float(g[1] @ w)
    norm = (gx * gx + gy * gy) ** 0.5
    if norm < 1e-9:
        # renk-yönlü gradyan (luma sabit): en büyük kanal eğimini kullan
        mags = np.linalg.norm(g, axis=1)
        k = int(np.argmax(np.abs(g).sum(axis=1)))
        gx, gy = float(g[0][k]), float(g[1][k])
        norm = (gx * gx + gy * gy) ** 0.5
        if norm < 1e-9 or mags.max() < 1e-9:
            return None
    ux, uy = gx / norm, gy / norm

    t = xs * ux + ys * uy
    t1, t2 = float(np.percentile(t, 1)), float(np.percentile(t, 99))
    if t2 - t1 < 2.0:
        return None
    # uç noktalar: projeksiyon eksenindeki bölge merkez hattı üzerinde
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    p1 = (cx + (t1 - (cx * ux + cy * uy)) * ux, cy + (t1 - (cx * ux + cy * uy)) * uy)
    p2 = (cx + (t2 - (cx * ux + cy * uy)) * ux, cy + (t2 - (cx * ux + cy * uy)) * uy)
    c1 = np.clip(coef[0] + coef[1] * p1[0] + coef[2] * p1[1], 0, 255).astype(np.uint8)
    c2 = np.clip(coef[0] + coef[1] * p2[0] + coef[2] * p2[1], 0, 255).astype(np.uint8)
    if _delta_e(tuple(int(v) for v in c1), tuple(int(v) for v in c2)) < _GRAD_MIN_SPAN_DE:
        return None
    return {
        "x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1],
        "c1": tuple(int(v) for v in c1), "c2": tuple(int(v) for v in c2),
        "gain_de": round(de_const - de_grad, 2),
    }


def _adjacent_region_pairs(id_map: np.ndarray, n_bins: int) -> list[tuple[int, int, int]]:
    """ID haritasından komşu bölge çiftlerini (a, b, sınır_uzunluğu) çıkarır."""
    acc: dict[int, int] = {}
    for a, b in ((id_map[:, :-1], id_map[:, 1:]), (id_map[:-1, :], id_map[1:, :])):
        m = (a != b) & (a >= 1) & (a < n_bins) & (b >= 1) & (b < n_bins)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys, cnts = np.unique(lo * n_bins + hi, return_counts=True)
        for k, c in zip(keys.tolist(), cnts.tolist()):
            acc[k] = acc.get(k, 0) + c
    return [(k // n_bins, k % n_bins, c) for k, c in acc.items()]


def _same_tone_family(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """İki dolgu aynı ton ailesinden mi? (bant-birleştirme adayı)

    Yakın ΔE + benzer ton açısı: bir gradyanın kuantize bantları böyledir;
    farklı nesnelerin renkleri (kırmızı domates / yeşil yaprak) elenir.
    """
    la = _lab(np.array([a], dtype=np.uint8))[0]
    lb = _lab(np.array([b], dtype=np.uint8))[0]
    # cv2 LAB ölçeği: L 0-255'e ölçeklenir (gerçek ΔE76'nın ~2.55 katı L'de);
    # 45 ≈ gerçek ΔE ~18 — kaba posterizasyon adımlarını (erdogmus bandı,
    # 30-RGB'lik gri merdiveni) kapsar, farklı nesne renklerini dışarıda tutar
    if float(np.linalg.norm(la - lb)) > 45.0:
        return False
    ca = float(np.hypot(la[1] - 128.0, la[2] - 128.0))
    cb = float(np.hypot(lb[1] - 128.0, lb[2] - 128.0))
    if ca <= 10.0 and cb <= 10.0:
        return True  # ikisi de nötr (gri merdiveni)
    if ca <= 10.0 or cb <= 10.0:
        return ca <= 14.0 and cb <= 14.0  # nötr-kromatik sınırı: ancak ikisi de soluksa
    ha = float(np.degrees(np.arctan2(la[2] - 128.0, la[1] - 128.0)))
    hb = float(np.degrees(np.arctan2(lb[2] - 128.0, lb[1] - 128.0)))
    dh = abs(ha - hb) % 360.0
    return min(dh, 360.0 - dh) <= 30.0


_BAND_MIN_GAIN_DE = 0.8  # çok-bantlı mevcut gösterime göre asgari ΔE kazancı


def _fit_linear_gradient_multi(
    ys: np.ndarray, xs: np.ndarray, pix: np.ndarray, cur_rgb: np.ndarray
) -> dict[str, Any] | None:
    """Bant kümesi üzerinde doğrusal gradyan; kazanç MEVCUT bant renklerine göre.

    _fit_linear_gradient tek sabit renge karşı ölçer; burada her pikselin
    mevcut rengi kendi bandının dolgusudur (çok-bantlı posterizasyon).
    """
    n = len(xs)
    if n < _GRAD_MIN_PX:
        return None
    a = np.stack([np.ones(n, np.float64), xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(a, pix.astype(np.float64), rcond=None)
    pred = a @ coef

    lab_t = _lab(pix.astype(np.uint8))
    lab_c = _lab(np.clip(cur_rgb, 0, 255).astype(np.uint8))
    de_cur = float(np.mean(np.linalg.norm(lab_t - lab_c, axis=1)))
    lab_p = _lab(np.clip(pred, 0, 255).astype(np.uint8))
    de_grad = float(np.mean(np.linalg.norm(lab_t - lab_p, axis=1)))
    if de_cur - de_grad < _BAND_MIN_GAIN_DE:
        return None

    wl = np.array([0.299, 0.587, 0.114])
    g = coef[1:, :]
    gx, gy = float(g[0] @ wl), float(g[1] @ wl)
    norm = (gx * gx + gy * gy) ** 0.5
    if norm < 1e-9:
        k = int(np.argmax(np.abs(g).sum(axis=1)))
        gx, gy = float(g[0][k]), float(g[1][k])
        norm = (gx * gx + gy * gy) ** 0.5
        if norm < 1e-9:
            return None
    ux, uy = gx / norm, gy / norm
    t = xs * ux + ys * uy
    t1, t2 = float(np.percentile(t, 1)), float(np.percentile(t, 99))
    if t2 - t1 < 4.0:
        return None
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    p1 = (cx + (t1 - (cx * ux + cy * uy)) * ux, cy + (t1 - (cx * ux + cy * uy)) * uy)
    p2 = (cx + (t2 - (cx * ux + cy * uy)) * ux, cy + (t2 - (cx * ux + cy * uy)) * uy)
    c1 = np.clip(coef[0] + coef[1] * p1[0] + coef[2] * p1[1], 0, 255).astype(np.uint8)
    c2 = np.clip(coef[0] + coef[1] * p2[0] + coef[2] * p2[1], 0, 255).astype(np.uint8)
    if _delta_e(tuple(int(v) for v in c1), tuple(int(v) for v in c2)) < _GRAD_MIN_SPAN_DE:
        return None
    return {
        "x1": p1[0], "y1": p1[1], "x2": p2[0], "y2": p2[1],
        "c1": tuple(int(v) for v in c1), "c2": tuple(int(v) for v in c2),
        "gain_de": round(de_cur - de_grad, 2),
    }


def refit_svg_colors_per_path(
    svg_path: Path,
    original_path: Path,
    out_path: Path,
    max_side: int = 2048,
    min_region_px: int = 12,
) -> dict[str, Any]:
    """FOTO-YOĞUN çıktı için per-path bağımsız renk refit'i (vektörize).

    Palet-koruyan grup modu (refit_svg_colors) aynı kaynak rengi paylaşan TÜM
    path'leri birlikte taşır — logolarda doğru (marka rengi bölünmez) ama
    foto-yoğun içerikte YANLIŞ: aynı kuantize renk, görüntünün uzak ve farklı
    tonlu bölgelerine dağılır; global havuz medyanı yerel tonu bozar (ölçüldü:
    mangal −0.5). Foto-yoğunda detay ürünün kendisidir (editability de bu
    rejimde kapalıdır): her path kendi görünür bölgesinin ORİJİNAL rengini alır
    — izlemenin/konsolidasyonun yuttuğu tonlar (kahverengi→turuncu kayması,
    kaybolan sebze renkleri) geri gelir.

    Binlerce path'te hızlı olması için bölge istatistikleri tek geçişte
    vektörize hesaplanır (bincount ortalaması; AA pikselleri komşuluk-uniform
    iç-bölge maskesiyle dışlanır). Benimseme kararı çağırandadır.
    """
    from app.fidelity import load_reference_rgb, render_svg_to_rgb  # noqa: PLC0415

    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"parse: {e}"}
    paths = _iter_paths(tree.getroot())
    if not paths:
        return {"changed": 0, "error": "path yok"}
    # foto-yoğun çıktıda binlerce KÜÇÜK bölge vardır; 512px'te çoğu ölçüm
    # eşiğinin altında kalır (5002 path'ten 96'sı ölçülebildi — gerçek ölçüm).
    # Yüksek çözünürlüklü ID-render bölge alanlarını büyütüp kapsamayı artırır
    # (2048 + min 12 iç piksel: 1862 path ölçülür; süre ~1s, vektörize).
    try:
        ref, (w, h) = load_reference_rgb(Path(original_path), max_side=max_side)
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"referans: {e}"}

    id_tree, _ = _build_id_svg(tree)
    id_svg = Path(out_path).with_suffix(".idmap.svg")
    try:
        id_tree.write(str(id_svg), encoding="utf-8", xml_declaration=True)
        id_rgb = render_svg_to_rgb(id_svg, w, h)
    finally:
        id_svg.unlink(missing_ok=True)
    if id_rgb is None:
        return {"changed": 0, "error": "render backend yok"}
    id_map = _decode_id_map(id_rgb)

    # iç-bölge maskesi: 4-komşusu aynı ID olan pikseller (AA karışımı dışarıda)
    interior = np.ones_like(id_map, dtype=bool)
    interior[1:, :] &= id_map[1:, :] == id_map[:-1, :]
    interior[:-1, :] &= id_map[:-1, :] == id_map[1:, :]
    interior[:, 1:] &= id_map[:, 1:] == id_map[:, :-1]
    interior[:, :-1] &= id_map[:, :-1] == id_map[:, 1:]

    ids = id_map[interior]
    pix = ref[interior].astype(np.float64)
    n_bins = len(paths) + 2
    valid = (ids >= 1) & (ids < n_bins)
    ids, pix = ids[valid], pix[valid]
    counts = np.bincount(ids, minlength=n_bins)
    means = np.zeros((n_bins, 3), np.float64)
    for c in range(3):
        s = np.bincount(ids, weights=pix[:, c], minlength=n_bins)
        np.divide(s, counts, out=means[:, c], where=counts > 0)

    changed = 0
    shifts: list[float] = []
    new_fill: dict[int, tuple[int, int, int]] = {}  # path idx -> sabit renk
    for i, el in enumerate(paths):
        code = i + 1
        if code >= 0xFFFFFF:
            code += 1
        if code >= n_bins or counts[code] < min_region_px:
            continue
        old = _parse_fill(el.get("fill"))
        if old is None or not _elem_opaque(el):
            continue
        new = tuple(int(round(v)) for v in np.clip(means[code], 0, 255))
        new_fill[i] = new
        de = _delta_e(old, new)
        if de < _MIN_SHIFT_DE or de > _MAX_SHIFT_DE:
            continue
        el.set("fill", _hex(new))
        changed += 1
        shifts.append(de)

    # GRADYAN UZANIMI (lekesizlik): parlak yansımalı/yumuşak tonlu BÜYÜK
    # bölgelerde sabit renk, posterizasyon "lekeleri" bırakır (domates üzerinde
    # krem adalar — gerçek kullanıcı şikâyeti). En büyük bölgelere doğrusal
    # gradyan LSQ ile oturtulur; kazanç eşiği geçilmezse bölge sabit kalır.
    grad_applied = 0
    root = tree.getroot()
    vb = root.get("viewBox")
    try:
        _, _, vbw, vbh = (float(x) for x in (vb or "").replace(",", " ").split())
    except ValueError:
        vbw, vbh = float(w), float(h)
    sx, sy = vbw / float(w), vbh / float(h)
    # Gradyan + transform: userSpaceOnUse uçları, path'in KENDİ kullanıcı
    # uzayında yorumlanır. vtracer HER path'e translate(...) verir — bunlar
    # dondurulursa hiçbir gerçek çıktı gradyan alamaz (ölçüldü: gradients hep
    # 0'dı). Çözüm: yalnız-translate transform'larda uçlar path başına ters
    # ofsetlenir (üye başına ayrı def, aynı doğru üzerinde -> görsel süreklilik
    # birebir). Translate-dışı transform'lu path gradyan almaz.
    xf_offset: dict[int, tuple[float, float] | None] = {}

    def _mark_xf(el: ET.Element, inherited_blocked: bool) -> None:
        if el.tag.split("}")[-1] == "path":
            if inherited_blocked:
                xf_offset[id(el)] = None
            else:
                t = el.get("transform")
                if t is None:
                    xf_offset[id(el)] = (0.0, 0.0)
                else:
                    m = re.match(r"^\s*translate\(\s*([-+0-9.eE]+)(?:[\s,]+([-+0-9.eE]+))?\s*\)\s*$", t)
                    xf_offset[id(el)] = (float(m.group(1)), float(m.group(2) or 0.0)) if m else None
        blocked = inherited_blocked or (
            el.get("transform") is not None and el.tag.split("}")[-1] != "path"
        )
        for ch in list(el):
            _mark_xf(ch, blocked)

    _mark_xf(root, False)
    defs_el: ET.Element | None = None

    def _grad_ok(i: int) -> bool:
        return xf_offset.get(id(paths[i])) is not None

    def _emit_gradient(gid: str, grad: dict[str, Any], members: list[int]) -> None:
        nonlocal defs_el, grad_applied, changed
        if defs_el is None:
            defs_el = root.find(f"{{{SVG_NS}}}defs")
            if defs_el is None:
                defs_el = ET.Element(f"{{{SVG_NS}}}defs")
                root.insert(0, defs_el)
        for i in members:
            tx, ty = xf_offset[id(paths[i])] or (0.0, 0.0)
            mid = gid if (tx == 0.0 and ty == 0.0) else f"{gid}_m{i}"
            if defs_el.find(f"{{{SVG_NS}}}linearGradient[@id='{mid}']") is None:
                g_el = ET.SubElement(defs_el, f"{{{SVG_NS}}}linearGradient", {
                    "id": mid, "gradientUnits": "userSpaceOnUse",
                    "x1": f"{grad['x1'] * sx - tx:.2f}", "y1": f"{grad['y1'] * sy - ty:.2f}",
                    "x2": f"{grad['x2'] * sx - tx:.2f}", "y2": f"{grad['y2'] * sy - ty:.2f}",
                })
                ET.SubElement(g_el, f"{{{SVG_NS}}}stop",
                              {"offset": "0", "stop-color": _hex(grad["c1"])})
                ET.SubElement(g_el, f"{{{SVG_NS}}}stop",
                              {"offset": "1", "stop-color": _hex(grad["c2"])})
            paths[i].set("fill", f"url(#{mid})")
            grad_applied += 1
            changed += 1

    # 3a) BANT-BİRLEŞTİRME (banding giderme): kuantizasyon, yumuşak gradyanı
    # komşu ton bantlarına böler (siyah bandın griye akışı, domates gölgesi).
    # Benzer tonlu KOMŞU bölgeler kümelege (union-find) toplanır ve kümeye TEK
    # doğrusal gradyan oturtulur; tüm üyeler AYNI url(#) dolguyu alır. Geometri
    # hiç değişmez (z-sırası/pyclipper riski yok) — renk bant sınırında sürekli
    # olduğundan bantlaşma görsel olarak kaybolur. Kazanç, mevcut ÇOK-BANTLI
    # gösterimin hatasına göre ölçülür.
    in_cluster: set[int] = set()
    pairs = _adjacent_region_pairs(id_map, n_bins)
    parent = list(range(n_bins))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, blen in pairs:
        ia, ib = int(a) - 1, int(b) - 1
        if ia not in new_fill or ib not in new_fill or blen < 15:
            continue
        if not _grad_ok(ia) or not _grad_ok(ib):
            continue
        if not _same_tone_family(new_fill[ia], new_fill[ib]):
            continue
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            parent[rb] = ra

    clusters: dict[int, list[int]] = {}
    for code in range(1, n_bins):
        i = code - 1
        if i in new_fill:
            clusters.setdefault(_find(code), []).append(code)
    big_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:40]
    for rank, (_rt, codes) in enumerate(big_clusters):
        if len(codes) < 2 or len(codes) > 12:
            continue
        total_px = int(counts[codes].sum())
        if total_px < 1500:
            continue
        mask = np.isin(id_map, codes) & interior
        ys, xs = np.nonzero(mask)
        cur_rgb = np.zeros((n_bins, 3), np.float64)
        for c in codes:
            cur_rgb[c] = new_fill[c - 1]
        grad = _fit_linear_gradient_multi(ys, xs, ref[ys, xs], cur_rgb[id_map[mask]])
        if grad is None:
            continue
        members = [c - 1 for c in codes]
        _emit_gradient(f"refit_band_{rank}", grad, members)
        in_cluster.update(members)

    # 3b) tek-bölge gradyan uzanımı (kümede olmayan büyük bölgeler)
    big_codes = np.nonzero(counts >= _GRAD_MIN_PX)[0]
    big_codes = big_codes[np.argsort(-counts[big_codes])][:80]  # en büyük 80 bölge
    for code in big_codes:
        i = int(code) - 1
        if i < 0 or i >= len(paths) or i not in new_fill or i in in_cluster:
            continue
        if not _grad_ok(i):
            continue
        mask = (id_map == code) & interior
        ys, xs = np.nonzero(mask)
        grad = _fit_linear_gradient(ys, xs, ref[ys, xs], new_fill[i])
        if grad is None:
            continue
        _emit_gradient(f"refit_grad_{i}", grad, [i])

    if changed == 0:
        return {"changed": 0}
    try:
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"yazma: {e}"}
    return {
        "changed": changed,
        "mode": "per_path",
        "gradients": grad_applied,
        "paths_measured": int((counts[1:len(paths) + 1] >= min_region_px).sum()),
        "mean_shift_de": round(float(np.mean(shifts)), 2) if shifts else 0.0,
        "max_shift_de": round(float(np.max(shifts)), 2) if shifts else 0.0,
    }


def refit_svg_colors(
    svg_path: Path,
    original_path: Path,
    out_path: Path,
    gradients: bool = False,
) -> dict[str, Any]:
    """SVG dolgularını orijinal görüntünün bölge medyanlarına yeniden oturtur.

    ``gradients=True`` ile büyük bölgelerde doğrusal gradyan uzanımı da denenir.
    Yalnız ``out_path`` yazılır; benimseme kararı (fidelity artışı ölçümü)
    çağırana aittir. Dönen rapor: değişen path sayısı, ortalama kayma vb.
    Başarısızlıkta {"changed": 0, "error": ...} döner (çökme yok).
    PALET-KORUYAN grup modudur (logo/marka işleri); foto-yoğun içerik için
    ``refit_svg_colors_per_path`` kullanılır.
    """
    from app.fidelity import load_reference_rgb, render_svg_to_rgb  # noqa: PLC0415

    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"parse: {e}"}

    paths = _iter_paths(tree.getroot())
    if not paths:
        return {"changed": 0, "error": "path yok"}

    try:
        ref, (w, h) = load_reference_rgb(Path(original_path))
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"referans: {e}"}

    # ID haritası: benzersiz renkli kopyayı bir kez render et
    id_tree, _ = _build_id_svg(tree)
    id_svg = Path(out_path).with_suffix(".idmap.svg")
    try:
        id_tree.write(str(id_svg), encoding="utf-8", xml_declaration=True)
        id_rgb = render_svg_to_rgb(id_svg, w, h)
    finally:
        id_svg.unlink(missing_ok=True)
    if id_rgb is None:
        return {"changed": 0, "error": "render backend yok"}
    id_map = _decode_id_map(id_rgb)

    # ölçek: SVG kullanıcı uzayı -> karşılaştırma pikseli (gradyan uçları için)
    root = tree.getroot()
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
    sx, sy = vbw / float(w), vbh / float(h)

    kernel = np.ones((3, 3), np.uint8)

    # transform kapsamındaki path'lere gradyan uygulanmaz: userSpaceOnUse uçları
    # referans alan öğenin (transform dahil) kullanıcı uzayında yorumlanır;
    # kök-uzayı koordinatlarımız orada çift dönüşüme uğrardı
    no_grad: set[int] = set()

    def _mark_transform_scope(el: ET.Element, inherited: bool) -> None:
        has = inherited or (el.get("transform") is not None)
        if has and el.tag.split("}")[-1] == "path":
            no_grad.add(id(el))
        for ch in list(el):
            _mark_transform_scope(ch, has)

    _mark_transform_scope(root, False)

    # 1) path başına görünür-iç piksellerden medyan renk
    per_path: list[dict[str, Any] | None] = [None] * len(paths)
    for i, el in enumerate(paths):
        old = _parse_fill(el.get("fill"))
        if old is None or not _elem_opaque(el):
            continue  # gradyan/none/desteklenmeyen dolgu ya da yarı saydam
        code = i + 1
        if code >= 0xFFFFFF:
            code += 1
        mask = (id_map == code).astype(np.uint8)
        n_vis = int(mask.sum())
        if n_vis < _MIN_REGION_PX:
            continue
        interior = cv2.erode(mask, kernel)
        if int(interior.sum()) < _MIN_REGION_PX:
            interior = mask  # ince şekil: erozyon her şeyi silmesin
        ys, xs = np.nonzero(interior)
        pix = ref[ys, xs]
        med = tuple(int(v) for v in np.median(pix, axis=0))
        per_path[i] = {
            "el": el, "old": old, "median": med, "n": len(ys),
            "ys": ys, "xs": xs, "pix": pix,
        }

    # 2) PALET-KORUYAN atama: aynı eski dolguyu paylaşan tüm path'ler TEK havuz
    #    medyanına taşınır. Böylece çıktı paletindeki benzersiz renk sayısı kaynak
    #    paletini ASLA aşamaz (kaynak zaten renk-sayısı kalite geçidini geçmişti);
    #    düzenlenebilirlik ve marka-rengi bütünlüğü korunur, yalnız her ton
    #    orijinaline ΔE olarak yaklaşır. (Path-başına bağımsız medyan paleti
    #    onlarca yakın-ama-ayrık tona şişiriyordu — düzenlenemez çıktı.)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, rec in enumerate(per_path):
        if rec is not None:
            groups.setdefault(rec["old"], []).append(i)

    assignments: dict[int, tuple[int, int, int]] = {}
    group_new: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for old, idxs in groups.items():
        pooled = np.concatenate([per_path[i]["pix"] for i in idxs])
        pooled_med = tuple(int(v) for v in np.median(pooled, axis=0))
        de = _delta_e(old, pooled_med)
        new = old if (de < _MIN_SHIFT_DE or de > _MAX_SHIFT_DE) else pooled_med
        group_new[old] = new
        for i in idxs:
            assignments[i] = new

    # iki eski renk aynı yeni renge (ΔE <= 1.5) yakınsıyorsa tek merkeze çekilir:
    # palet yalnız KÜÇÜLEBİLİR, asla büyümez
    order = sorted(group_new, key=lambda o: -sum(per_path[i]["n"] for i in groups[o]))
    centers: list[tuple[int, int, int]] = []
    remap: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for old in order:
        c = group_new[old]
        snapped = next((k for k in centers if _delta_e(c, k) <= 1.5), None)
        if snapped is None:
            centers.append(c)
            remap[old] = c
        else:
            remap[old] = snapped
    for i, rec in enumerate(per_path):
        if rec is not None and i in assignments:
            assignments[i] = remap[rec["old"]]

    changed = 0
    grad_applied = 0
    shifts: list[float] = []
    defs_el: ET.Element | None = None
    grad_paths: set[int] = set()  # gradyan verilen öğe id'leri (remap dışı bırak)

    # 3) isteğe bağlı gradyan uzanımı (yalnız ölçülen büyük bölgelerde): sabit
    #    renk yerine doğrusal model. Gradyanlı path artık remap'e girmez.
    if gradients:
        for i, new in assignments.items():
            rec = per_path[i]
            el = rec["el"]
            if rec["n"] < _GRAD_MIN_PX or id(el) in no_grad:
                continue
            grad = _fit_linear_gradient(rec["ys"], rec["xs"], rec["pix"], new)
            if grad is None:
                continue
            if defs_el is None:
                defs_el = root.find(f"{{{SVG_NS}}}defs")
                if defs_el is None:
                    defs_el = ET.Element(f"{{{SVG_NS}}}defs")
                    root.insert(0, defs_el)
            gid = f"refit_grad_{i}"
            g_el = ET.SubElement(defs_el, f"{{{SVG_NS}}}linearGradient", {
                "id": gid, "gradientUnits": "userSpaceOnUse",
                "x1": f"{grad['x1'] * sx:.2f}", "y1": f"{grad['y1'] * sy:.2f}",
                "x2": f"{grad['x2'] * sx:.2f}", "y2": f"{grad['y2'] * sy:.2f}",
            })
            ET.SubElement(g_el, f"{{{SVG_NS}}}stop",
                          {"offset": "0", "stop-color": _hex(grad["c1"])})
            ET.SubElement(g_el, f"{{{SVG_NS}}}stop",
                          {"offset": "1", "stop-color": _hex(grad["c2"])})
            el.set("fill", f"url(#{gid})")
            grad_paths.add(id(el))
            grad_applied += 1
            changed += 1

    # 4) sabit-renk remap'ini TÜM path'lere RENK DEĞERİNE göre uygula: ölçülen
    #    büyük path'ler kadar, aynı kaynak rengi paylaşan ölçülemeyen küçük
    #    path'ler de birlikte taşınır. Böylece bir kaynak renk ASLA bölünmez
    #    (küçükler eski, büyükler yeni renkte kalıp paleti şişirmez) ve çıktı
    #    paleti kaynak paletini geçemez.
    for el in paths:
        if id(el) in grad_paths:
            continue
        old = _parse_fill(el.get("fill"))
        if old is None or old not in remap:
            continue
        new = remap[old]
        if new != old:
            el.set("fill", _hex(new))
            changed += 1
            shifts.append(_delta_e(old, new))

    if changed == 0:
        return {"changed": 0}

    # bellek tutmayalım: piksel dizileri rapora girmez
    try:
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"changed": 0, "error": f"yazma: {e}"}

    return {
        "changed": changed,
        "gradients": grad_applied,
        "paths_measured": sum(1 for r in per_path if r is not None),
        "mean_shift_de": round(float(np.mean(shifts)), 2) if shifts else 0.0,
        "max_shift_de": round(float(np.max(shifts)), 2) if shifts else 0.0,
    }
