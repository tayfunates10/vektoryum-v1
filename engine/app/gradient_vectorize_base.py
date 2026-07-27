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
    """Kenar-bazlı bölge segmentasyonu (connected components).

    Renk kümeleme YERİNE gerçek kenarlarla bölgeler oluşturur: kenarlar görseli
    kapalı eğrilerle alanlara böler, kenar-olmayan bağlı bileşenler = bölgeler.
    Böylece düz bir gradyan (içinde güçlü kenar yoktur) TEK bölge olarak kalır —
    bantlara/yarılara bölünmez — ve sınırlar gerçek kenarlara hizalanır.

    Döner: her piksel için tamsayı etiket haritası (0 dahil değil; kenar
    pikselleri en yakın bölgeye yaslanır).
    """
    # hafif düzleştirme: anti-alias gürültüsünü bastır, gerçek kenarları koru
    smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=30, sigmaSpace=30)
    gray_s = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_s, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # kenar-olmayan alanın bağlı bileşenleri = bölgeler (kenar pikselleri = etiket 0)
    non_edge = (edges == 0).astype(np.uint8)
    _num, labels = cv2.connectedComponents(non_edge, connectivity=4)
    labels = labels.astype(np.int32)

    # Kenar bandını en yakın gerçek bölge TOHUMUNA yasla. Önceki ``dilate +
    # max(label)`` yöntemi etiketi büyük olan tarafı her kenarda sistematik
    # olarak 1px şişiriyordu (rounded-rect bbox 70..410 yerine 69..410);
    # vektör ölçeklenince bu gerçek bir çerçeve/yarıçap hatasına dönüşüyordu.
    # Distance-transform etiketi numerik öncelik kullanmaz ve iki taraftan eşit
    # uzaklıkta geometrik Voronoi sınırı kurar.
    edge_pixels = labels == 0
    if edge_pixels.any() and (~edge_pixels).any():
        _distance, nearest = cv2.distanceTransformWithLabels(
            edge_pixels.astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        seed_to_region = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
        seed_to_region[nearest[~edge_pixels]] = labels[~edge_pixels]
        labels[edge_pixels] = seed_to_region[nearest[edge_pixels]]

    return labels


def _mask_to_path(mask: np.ndarray, epsilon: float = 0.3, min_area: float = 12.0) -> str:
    """İkili maskeden delikleri koruyan (evenodd) polygon path 'd' üretir.

    Maske önce hafifçe Gaussian ile yumuşatılır: connected-components sınırındaki
    1px merdivenlenme giderilir, polygon gerçek (anti-alias) kenara daha iyi oturur
    (edge_f1 belirgin artar). Düşük ``epsilon`` ile yuvarlak köşeler korunur.
    """
    smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    smoothed = ((smoothed > 127).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(smoothed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        eps = max(0.3, epsilon * peri / 100.0)
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

    # Eksen uç renkleri: bölgenin tamamında doğrusal LSQ. Yalnız bin
    # merkezlerine stop yazmak 0..ilk-merkez ve son-merkez..1 aralığında renk
    # platosu oluşturuyordu. Gerçek 0/1 uç stopları bu banding'i kaldırır;
    # ara medyan stoplar doğrusal olmayan yumuşak geçişleri korur.
    line_a = np.stack([np.ones_like(t, dtype=np.float64), t.astype(np.float64)], axis=1)
    line_coef, *_ = np.linalg.lstsq(line_a, colors.astype(np.float64), rcond=None)
    endpoint_first = np.clip(line_coef[0] + line_coef[1] * t_min, 0, 255).astype(np.uint8)
    endpoint_last = np.clip(line_coef[0] + line_coef[1] * t_max, 0, 255).astype(np.uint8)

    # eksen boyunca medyan renkten ara stop'lar
    edges = np.linspace(t_min, t_max, _GRADIENT_STOPS + 1)
    stops: list[tuple[float, tuple[int, int, int]]] = [
        (0.0, tuple(int(v) for v in endpoint_first)),
    ]
    for i in range(_GRADIENT_STOPS):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if int(sel.sum()) < 3:
            continue
        col = np.median(colors[sel], axis=0)
        offset = (0.5 * (edges[i] + edges[i + 1]) - t_min) / (t_max - t_min)
        stops.append((float(offset), (int(col[0]), int(col[1]), int(col[2]))))
    stops.append((1.0, tuple(int(v) for v in endpoint_last)))

    if len(stops) < 2:
        return None

    c_first = endpoint_first.astype(np.float32)
    c_last = endpoint_last.astype(np.float32)
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
    epsilon = float(params.get("epsilon", 0.3))
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
