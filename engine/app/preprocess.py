"""Profil bazlı ön işleme katmanı.

Her vektörleştirme profili için ayrı bir ön işleme fonksiyonu vardır. Tek
giriş noktası ``preprocess_for_mode`` bu fonksiyonlara dağıtım yapar ve
``(çıktı_yolu, rapor)`` döndürür.

Tasarım ilkeleri:
* Geometrik logolarda düz çizgi ve köşeleri korumak için yalnızca çok hafif
  kenar-koruyan filtre + flat palet indirgeme uygulanır (Gaussian blur yok).
* Palet sertleştirme yalnızca kanonik siyah/beyaz/kırmızıya çok yakın renkleri
  yaslar; kurumsal diğer renkler korunur.
* Çizgi kalınlığını değiştiren morfoloji kullanılmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Kanonik renkler (RGB)
_CANON = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "dark_gray": (45, 45, 45),
}


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _rgba_to_rgb_on_white(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 3:
        return arr
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    rgb = arr[..., :3].astype(np.float32)
    white = np.full_like(rgb, 255.0)
    return (rgb * alpha + white * (1 - alpha)).astype(np.uint8)


def _foreground_mask_from_alpha(arr: np.ndarray) -> np.ndarray | None:
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    alpha = arr[..., 3]
    if np.all(alpha > 250):
        return None
    return (alpha > 128).astype(np.uint8) * 255


def _quantize_flat(rgb: np.ndarray, colors: int, dither: bool = False) -> np.ndarray:
    pil = Image.fromarray(rgb)
    q = pil.quantize(
        colors=max(2, colors),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE,
    )
    return np.array(q.convert("RGB"))


def _harden_palette(rgb: np.ndarray, tol: int = 42) -> np.ndarray:
    """Kanonik siyah/beyaz/kırmızıya yakın renkleri tam değere yaslar."""
    out = rgb.copy()
    flat = out.reshape(-1, 3).astype(np.int32)

    def _near(target: tuple[int, int, int], t: int) -> np.ndarray:
        diff = flat - np.array(target, dtype=np.int32)
        return np.sqrt((diff * diff).sum(axis=1)) <= t

    # beyaz (geniş tolerans), siyah, kırmızı
    flat[_near(_CANON["white"], tol + 18)] = _CANON["white"]
    flat[_near(_CANON["black"], tol)] = _CANON["black"]
    # kırmızı: doygun VE açık (anti-alias pembe) kırmızılar. Mutlak g/b sınırı
    # yerine bağıl baskınlık kullanılır; böylece ince kırmızı çizgiler (anti-alias
    # nedeniyle pembeleşmiş) kaybolmadan kırmızıya yaslanır.
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    red_like = (r >= 130) & (r > g + 45) & (r > b + 45) & (np.abs(g.astype(np.int32) - b.astype(np.int32)) <= 60)
    flat[red_like] = _CANON["red"]
    return flat.reshape(out.shape).astype(np.uint8)


def _reduce_to_dominant(rgb: np.ndarray, k: int) -> np.ndarray:
    """Her pikseli en sık görülen k renkten en yakınına atar (sert palet).

    Çıktıda yalnızca k (veya daha az) düz renk kalır; ara/anti-alias tonları
    yok olur. Sert kenarlar oluştuğundan VTracer renkleri yeniden çoğaltamaz.
    """
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    if len(colors) <= k:
        return rgb
    dominant = colors[np.argsort(counts)[::-1][:k]].astype(np.int32)
    # her piksel için en yakın dominant renk (parça parça, bellek dostu)
    out = np.empty_like(flat)
    chunk = 200000
    for start in range(0, len(flat), chunk):
        block = flat[start:start + chunk].astype(np.int32)
        d = ((block[:, None, :] - dominant[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk] = dominant[np.argmin(d, axis=1)]
    return out.reshape(rgb.shape)


def _remove_speckles(rgb: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Renk bölgelerindeki çok küçük izole lekeleri komşuya gömerek temizler."""
    out = rgb.copy()
    colors = np.unique(out.reshape(-1, 3), axis=0)
    if len(colors) > 24:
        return out  # çok renkli görselde atla
    for color in colors:
        mask = cv2.inRange(out, color, color)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                labels_mask = (labels == i).astype(np.uint8) * 255
                dil = cv2.dilate(labels_mask, np.ones((3, 3), np.uint8))
                ring = cv2.subtract(dil, labels_mask)
                ys, xs = np.where(ring > 0)
                if len(xs) > 0:
                    repl = out[ys, xs].reshape(-1, 3)
                    fill = np.median(repl, axis=0).astype(np.uint8)
                    out[labels == i] = fill
    return out


# ---------------------------------------------------------------------------
# Profil fonksiyonları
# ---------------------------------------------------------------------------
def preprocess_geometric_logo(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    # ince çizgileri birbirine karıştırmamak için çok hafif kenar-koruyan filtre
    filtered = cv2.bilateralFilter(rgb, d=3, sigmaColor=18, sigmaSpace=18)
    report["steps"].append("bilateral_very_light")
    quant = _quantize_flat(filtered, colors=8)
    report["steps"].append("quantize_8")
    hard = _harden_palette(quant, tol=46)
    report["steps"].append("palette_harden_bwr")
    # küçük lekeleri temizle (ince çizgileri korumak için küçük eşik)
    cleaned = _remove_speckles(hard, min_area=3)
    report["steps"].append("despeckle")
    # SON adım: sert palet -> en baskın 4 renk; ara/median tonları garanti elenir
    reduced = _reduce_to_dominant(cleaned, k=4)
    report["steps"].append("reduce_to_dominant_4")
    report["palette"] = _palette_list(reduced)
    return reduced


def preprocess_minimal_ai(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    filtered = cv2.bilateralFilter(rgb, d=5, sigmaColor=35, sigmaSpace=35)
    report["steps"].append("bilateral_light")
    quant = _quantize_flat(filtered, colors=6)
    report["steps"].append("quantize_6")
    hard = _harden_palette(quant, tol=38)
    report["steps"].append("palette_harden_bwr")
    reduced = _reduce_to_dominant(hard, k=5)
    report["steps"].append("reduce_to_dominant_5")
    report["palette"] = _palette_list(reduced)
    return reduced


def _kmeans_quantize_lab(rgb: np.ndarray, k: int, edge_preserve: bool = True) -> np.ndarray:
    """LAB renk uzayında k-means ile algısal kuantizasyon.

    RGB'de değil LAB'de kümelendiği için renkler insan algısına göre ayrılır;
    çıktı az sayıda DÜZ ve temiz renk bölgesidir (Vectorizer.AI benzeri).
    """
    img = rgb
    if edge_preserve:
        # kenar-koruyan güçlü düzleştirme -> düz renk bölgeleri (gürültü/gradyan azalır)
        img = cv2.bilateralFilter(rgb, d=9, sigmaColor=45, sigmaSpace=45)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    unique_n = len(np.unique(samples.astype(np.uint8), axis=0))
    K = max(2, min(int(k), unique_n))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # HIZ: merkezleri alt-örneklemde bul, sonra TÜM pikselleri en yakın merkeze ata
    if len(samples) > 60000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(samples), 60000, replace=False)
        fit = samples[idx]
    else:
        fit = samples
    _compactness, _labels, centers = cv2.kmeans(fit, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.float32)

    labels_full = np.empty(len(samples), dtype=np.int32)
    chunk = 200000
    for s in range(0, len(samples), chunk):
        block = samples[s:s + chunk]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_full[s:s + chunk] = np.argmin(d2, axis=1)
    quant_lab = centers.astype(np.uint8)[labels_full].reshape(lab.shape)
    out = cv2.cvtColor(quant_lab, cv2.COLOR_LAB2RGB)
    return out


def preprocess_logo_color(arr: np.ndarray, report: dict, n_colors: int = 20) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    # LAB k-means: algısal, temiz, düz renk bölgeleri
    quant = _kmeans_quantize_lab(rgb, k=n_colors, edge_preserve=True)
    report["steps"].append(f"lab_kmeans_{n_colors}")
    # çok küçük renk lekelerini komşuya gömerek bölge sınırlarını temizle
    cleaned = _remove_speckles(quant, min_area=8) if len(np.unique(quant.reshape(-1, 3), axis=0)) <= 28 else quant
    report["steps"].append("despeckle")
    report["palette"] = _palette_list(cleaned, limit=24)
    return cleaned


def preprocess_lineart(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Otsu; ince çizgileri korumak için morfoloji yok
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    report["steps"].append("otsu_threshold")
    return binary


def preprocess_single_color(arr: np.ndarray, report: dict) -> np.ndarray:
    fg = _foreground_mask_from_alpha(arr)
    if fg is not None:
        report["steps"].append("alpha_foreground_mask")
        binary = 255 - fg  # konu siyah, zemin beyaz
    else:
        rgb = _rgba_to_rgb_on_white(arr)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        report["steps"].append("otsu_threshold")
    # küçük lekeleri temizle
    inv = 255 - binary
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 8:
            inv[labels == i] = 0
    report["steps"].append("despeckle")
    return 255 - inv


def preprocess_centerline(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    report["steps"].append("otsu_threshold_for_skeleton")
    return binary


def preprocess_photo_poster(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    quant = _kmeans_quantize_lab(rgb, k=16, edge_preserve=True)
    report["steps"].append("lab_kmeans_16")
    report["palette"] = _palette_list(quant, limit=16)
    report["note"] = "Fotoğraf modu: çıktı posterize bir yaklaşımdır, tam sadakat garanti edilmez."
    return quant


def _palette_list(rgb: np.ndarray, limit: int = 12) -> list[str]:
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in colors[order]]


_DISPATCH = {
    "geometric_logo": preprocess_geometric_logo,
    "minimal_ai": preprocess_minimal_ai,
    "flat_logo": preprocess_minimal_ai,
    "logo_color": preprocess_logo_color,
    "lineart": preprocess_lineart,
    "single_color": preprocess_single_color,
    "centerline": preprocess_centerline,
    "photo_poster": preprocess_photo_poster,
}


def _auto_color_count(analysis: dict[str, Any] | None) -> int:
    """Analizdeki renk zenginliğine göre logo_color için k seçer (14-28)."""
    if not analysis:
        return 20
    est = int(analysis.get("estimated_color_count", 14))
    return int(max(14, min(28, est + 6)))


def preprocess_for_mode(
    image_path: Path,
    mode: str,
    output_dir: Path,
    analysis: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Görseli seçilen moda göre ön işler ve PNG olarak kaydeder."""
    image = Image.open(image_path).convert("RGBA")

    # PERFORMANS: çok büyük girdileri trace öncesi küçült (vektör çıktı sonsuz
    # ölçeklenir; 4K+ görselde k-means/VTracer aşırı yavaşlar). Uzun kenar <= 1500.
    max_side = max(image.size)
    report_resize = None
    if max_side > 1500:
        scale = 1500 / max_side
        new_size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
        image = image.resize(new_size, Image.LANCZOS)
        report_resize = {"from": [max_side, max_side], "to": list(image.size)}

    arr = np.array(image)

    report: dict[str, Any] = {"mode": mode, "steps": []}
    if report_resize:
        report["resized"] = report_resize
    func = _DISPATCH.get(mode, preprocess_minimal_ai)
    if mode == "logo_color":
        n_colors = _auto_color_count(analysis)
        report["auto_color_count"] = n_colors
        processed = preprocess_logo_color(arr, report, n_colors=n_colors)
    else:
        processed = func(arr, report)

    output_path = output_dir / f"preprocessed_{mode}.png"
    if processed.ndim == 2:
        Image.fromarray(processed, mode="L").save(output_path)
    else:
        Image.fromarray(processed).save(output_path)

    report["output"] = str(output_path)
    return output_path, report
