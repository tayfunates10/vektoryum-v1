"""Gradyan-farkındalıklı vektörleştirme (Faz 2).

Klasik akış, gradyanları az sayıda DÜZ renk bandına böler (posterize) — bu da
renkli logolarda görünür "bantlaşma" ve yüksek ΔE üretir. Bu modül, görseli kaba
bölgelere ayırır ve her bölgeyi şu şekilde modeller:

* **Düz bölge** → tek `fill="#rrggbb"` (mevcut davranışla aynı).
* **Gradyan bölge** → tek path + SVG `<linearGradient>` (bantlaşma yok).

Çıktı, diğer adaylar gibi puanlanır. Algısal sadakat (app/fidelity.py) yargıç
olduğundan bu aday yalnızca GERÇEKTEN daha sadıksa seçilir; aksi halde mevcut
adaylar kazanır (mevcut kaliteye regresyon riski yok).

Tasarım: orijinal (posterize EDİLMEMİŞ) pikseller üzerinde çalışır — gradyanı
ancak ham veriden modelleyebiliriz. Yeni ağır bağımlılık yok (cv2+numpy).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# İşleme çözünürlüğü üst sınırı (vektör sonsuz ölçeklenir; hız için sınırla).
_MAX_SIDE = 900
# Bir bölgenin gradyan sayılması için iki uç stop arasındaki min renk mesafesi.
_GRADIENT_COLOR_DELTA = 26.0
# Gradyan örneklemesinde stop sayısı.
_GRADIENT_STOPS = 6


def _load_rgb(image_path: Path) -> np.ndarray:
    """Görseli beyaz zemine indirip RGB (H, W, 3) uint8 döndürür, boyut sınırlı."""
    with Image.open(image_path) as im:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        rgb = np.asarray(bg.convert("RGB"))
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest > _MAX_SIDE:
        scale = _MAX_SIDE / float(longest)
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return rgb


def _edge_based_segments(rgb: np.ndarray) -> np.ndarray:
    """Kenar-bazlı bölge segmentasyonu (watershed).

    Renk kümeleme YERİNE gerçek kenarlarla bölgeler oluşturur. Böylece düz bir
    gradyan (içinde güçlü kenar yoktur) TEK bölge olarak kalır — bantlara
    bölünmez — ve bölge sınırları görseldeki gerçek kenarlara hizalanır (pürüzsüz,
    keskin). Bu, gradyan adayının sınır kalitesini VTracer seviyesine çıkarır.

    Döner: her piksel için tamsayı etiket haritası (arka plan dahil).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # hafif düzleştirme: anti-alias gürültüsünü bastır, gerçek kenarları koru
    smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=30, sigmaSpace=30)
    gray_s = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_s, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # kenar-olmayan iç bölgeler -> watershed markörleri
    non_edge = (edges == 0).astype(np.uint8)
    sure = cv2.erode(non_edge, np.ones((3, 3), np.uint8), iterations=1)
    num, markers = cv2.connectedComponents(sure)
    markers = markers + 1            # arka plan 0 olmasın
    markers[edges > 0] = 0           # bilinmeyen (kenar) bölgeler

    markers = cv2.watershed(rgb, markers)  # 0/kenarlar -1 olur, etiketler büyür

    # watershed sınır pikselleri (-1): en yakın etikete yasla (1px dilate)
    boundary = markers == -1
    if boundary.any():
        filled = markers.copy()
        filled[boundary] = 0
        # birkaç dilate iterasyonu ile etiketleri sınıra doğru büyüt
        for _ in range(3):
            zero = filled == 0
            if not zero.any():
                break
            dil = cv2.dilate(filled.astype(np.int32).astype(np.float32), np.ones((3, 3), np.uint8))
            filled = np.where(zero, dil.astype(np.int32), filled)
        markers = filled

    return markers.astype(np.int32)


def _mask_to_path(mask: np.ndarray, epsilon: float = 1.2, min_area: float = 12.0) -> str:
    """İkili maskeden delikleri koruyan (evenodd) polygon path 'd' üretir."""
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        eps = max(0.5, epsilon * peri / 100.0)
        poly = cv2.approxPolyDP(contour, eps, True)
        if len(poly) < 3:
            continue
        parts.append("M " + " ".join(f"{int(p[0][0])} {int(p[0][1])}" for p in poly) + " Z")
    return " ".join(parts)


def _fit_linear_gradient(
    ys: np.ndarray, xs: np.ndarray, colors: np.ndarray
) -> tuple[float, float, float, float, list[tuple[float, tuple[int, int, int]]]] | None:
    """Bölge piksellerine lineer gradyan oturtur.

    Yön: en çok değişen renk kanalının uzamsal düzlem-fit'i (color = a·x + b·y + c)
    ile bulunur. Stop'lar bu eksen boyunca medyan renklerden örneklenir. İki uç
    renk yeterince farklı değilse (düz bölge) ``None`` döner.
    """
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    A = np.stack([xs_f, ys_f, np.ones_like(xs_f)], axis=1)

    best_dir = None
    best_mag = 0.0
    for ch in range(3):
        coef, *_ = np.linalg.lstsq(A, colors[:, ch].astype(np.float32), rcond=None)
        mag = float(np.hypot(coef[0], coef[1]))
        if mag > best_mag:
            best_mag = mag
            best_dir = (float(coef[0]), float(coef[1]))

    if best_dir is None or best_mag < 1e-6:
        return None

    norm = float(np.hypot(*best_dir))
    dx, dy = best_dir[0] / norm, best_dir[1] / norm
    t = xs_f * dx + ys_f * dy
    t_min, t_max = float(t.min()), float(t.max())
    if t_max - t_min < 1.0:
        return None

    # eksen uç noktaları (bölge centroid'inden geçen doğru üzerinde)
    cx, cy = float(xs_f.mean()), float(ys_f.mean())
    t_c = cx * dx + cy * dy
    x1, y1 = cx + dx * (t_min - t_c), cy + dy * (t_min - t_c)
    x2, y2 = cx + dx * (t_max - t_c), cy + dy * (t_max - t_c)

    # eksen boyunca medyan renkten stop'lar
    edges = np.linspace(t_min, t_max, _GRADIENT_STOPS + 1)
    stops: list[tuple[float, tuple[int, int, int]]] = []
    for i in range(_GRADIENT_STOPS):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if int(sel.sum()) < 3:
            continue
        col = np.median(colors[sel], axis=0)
        offset = (0.5 * (edges[i] + edges[i + 1]) - t_min) / (t_max - t_min)
        stops.append((float(offset), (int(col[0]), int(col[1]), int(col[2]))))

    if len(stops) < 2:
        return None

    c_first = np.array(stops[0][1], dtype=np.float32)
    c_last = np.array(stops[-1][1], dtype=np.float32)
    if float(np.linalg.norm(c_first - c_last)) < _GRADIENT_COLOR_DELTA:
        return None  # uçlar benzer -> düz bölge, gradyan değil

    return x1, y1, x2, y2, stops


def vectorize_with_gradients(
    input_path: Path,
    output_path: Path,
    params: dict[str, Any] | None = None,
) -> None:
    """Görseli gradyan-farkındalıklı SVG'ye dönüştürür.

    Bölgeler kenar-bazlı (watershed) segmentasyonla bulunur; her bölge düz renk
    veya lineer gradyan olarak modellenir. ``params``: ``epsilon`` (polygon
    sadeleştirme, vars. 1.0), ``min_area`` (vars. görsel alanının ~%0.04'ü).
    """
    params = params or {}
    rgb = _load_rgb(Path(input_path))
    height, width = rgb.shape[:2]
    total_px = height * width
    epsilon = float(params.get("epsilon", 1.0))
    min_area = float(params.get("min_area", max(12.0, total_px * 0.0004)))

    labels = _edge_based_segments(rgb)

    defs: list[str] = []
    paths: list[str] = []
    grad_id = 0

    for label in np.unique(labels):
        region = (labels == label).astype(np.uint8) * 255
        coverage = float(np.count_nonzero(region)) / max(total_px, 1)
        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            continue
        region_colors = rgb[ys, xs].astype(np.float32)
        mean_color = region_colors.mean(axis=0)

        # büyük ve neredeyse beyaz alan = arka plan -> atla
        if coverage > 0.35 and float(mean_color.min()) > 244:
            continue

        path_d = _mask_to_path(region, epsilon=epsilon, min_area=min_area)
        if not path_d:
            continue

        grad = _fit_linear_gradient(ys, xs, region_colors)
        if grad is not None:
            x1, y1, x2, y2, stops = grad
            gid = f"g{grad_id}"
            grad_id += 1
            stop_tags = "".join(
                f'<stop offset="{off:.3f}" stop-color="#{r:02x}{g:02x}{b:02x}"/>'
                for off, (r, g, b) in stops
            )
            defs.append(
                f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">{stop_tags}</linearGradient>'
            )
            paths.append(f'<path fill="url(#{gid})" fill-rule="evenodd" d="{path_d}"/>')
        else:
            med = np.median(region_colors, axis=0).astype(int)
            hex_color = f"#{int(med[0]):02x}{int(med[1]):02x}{int(med[2]):02x}"
            paths.append(f'<path fill="{hex_color}" fill-rule="evenodd" d="{path_d}"/>')

    if not paths:
        raise RuntimeError("gradient vectorizer hiç path üretemedi.")

    defs_block = f"<defs>{''.join(defs)}</defs>" if defs else ""
    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{defs_block}{"".join(paths)}</svg>'
    )
    Path(output_path).write_text(svg, encoding="utf-8")
    logger.info("Gradient vectorizer ile '%s' üretildi (%d path, %d gradyan).",
                Path(output_path).name, len(paths), len(defs))
