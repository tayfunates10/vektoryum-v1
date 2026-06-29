"""Algısal sadakat (perceptual fidelity) ölçüm katmanı.

Bir vektör adayının orijinal raster görsele ne kadar sadık olduğunu **algısal**
olarak ölçer. Eski yaklaşımın (gri tonlama + MSE) aksine üç tamamlayıcı sinyal
birleştirilir:

1. **SSIM**  — yapısal benzerlik (parlaklık/kontrast/yapı). Gözün algıladığı
   bozulmayı MSE'den çok daha iyi yakalar.
2. **Renk farkı (ΔE)** — CIELAB uzayında ortalama renk sapması. Renk logolarında
   bantlaşma/kayma bunu doğrudan cezalandırır.
3. **Kenar uyumu (edge-F1)** — Canny kenarlarının toleranslı eşleşmesi. Çizgi
   keskinliği / merdivenlenme bunda görünür.

Tasarım ilkesi: **yeni ağır bağımlılık yok.** Her şey zaten kurulu olan
``cv2 + numpy + scipy`` ile yapılır (scikit-image gerekmez). Rasterizer olarak
CairoSVG kullanılır; yoksa fonksiyonlar ``None`` döner ve çağıran taraf yapısal
skorlara güvenle düşer (projenin "çökme yok" felsefesi).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Karşılaştırma çözünürlüğü: hız/doğruluk dengesi. Vektör sonsuz ölçeklenir;
# 512px algısal farkları yakalamak için yeterli, k-means/SSIM hızlı kalır.
_COMPARE_MAX_SIDE = 512


# ---------------------------------------------------------------------------
# Görsel yükleme / render
# ---------------------------------------------------------------------------
def _rgb_on_white(image: Image.Image) -> np.ndarray:
    """PIL görselini beyaz zemine indirip (H, W, 3) uint8 RGB döndürür."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return np.asarray(background.convert("RGB"))


def load_reference_rgb(original_path: Path, max_side: int = _COMPARE_MAX_SIDE) -> tuple[np.ndarray, tuple[int, int]]:
    """Orijinal görseli RGB (beyaz zemin) olarak yükler ve hedef boyutu döndürür.

    Dönen boyut (genişlik, yükseklik) SVG'nin aynı oranda render edileceği
    karşılaştırma çözünürlüğüdür.
    """
    with Image.open(original_path) as im:
        rgb = _rgb_on_white(im)
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        w2, h2 = max(1, round(w * scale)), max(1, round(h * scale))
        rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    return rgb, (w, h)


def _render_pymupdf(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """PyMuPDF (MuPDF) backend. Kendi içinde render motoru barındırır; Windows'ta
    harici DLL gerektirmez. Hedef platformda birincil çalışan rasterizer budur.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415

        doc = fitz.open(str(svg_path))
        try:
            page = doc[0]
            rect = page.rect
            if rect.width <= 0 or rect.height <= 0:
                return None
            matrix = fitz.Matrix(width / rect.width, height / rect.height)
            pix = page.get_pixmap(matrix=matrix, alpha=True)
            img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            return _rgb_on_white(img)
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("pymupdf render atlandı (%s): %s", svg_path.name, e)
        return None


def _render_cairosvg(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """CairoSVG backend. Windows'ta cairo DLL yoksa import/render başarısız olur."""
    try:
        import cairosvg  # noqa: PLC0415

        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=int(width),
            output_height=int(height),
            background_color="white",
        )
        return _rgb_on_white(Image.open(io.BytesIO(png_bytes)))
    except Exception as e:  # noqa: BLE001
        logger.debug("cairosvg render atlandı (%s): %s", svg_path.name, e)
        return None


def _render_svglib(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """svglib + reportlab renderPM backend. Saf-Python; Windows'ta DLL gerektirmez.

    CairoSVG'nin cairo DLL bağımlılığı olmadığı için Windows'ta birincil çalışan
    rasterizer budur. SVG'ler path tabanlı olduğundan (metin yok) renderPM yeterli.
    """
    try:
        from reportlab.graphics import renderPM  # noqa: PLC0415
        from svglib.svglib import svg2rlg  # noqa: PLC0415

        drawing = svg2rlg(str(svg_path))
        if drawing is None or drawing.width <= 0 or drawing.height <= 0:
            return None
        scale_x = width / float(drawing.width)
        scale_y = height / float(drawing.height)
        drawing.scale(scale_x, scale_y)
        drawing.width, drawing.height = width, height
        pil = renderPM.drawToPIL(drawing, dpi=72, bg=0xFFFFFF)
        return _rgb_on_white(pil)
    except Exception as e:  # noqa: BLE001
        logger.debug("svglib render atlandı (%s): %s", svg_path.name, e)
        return None


def _render_resvg(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """Opsiyonel resvg CLI backend (RESVG_PATH env ile). En sağlam, ama kurulum ister."""
    import os
    import shutil
    import subprocess

    resvg = os.environ.get("RESVG_PATH") or shutil.which("resvg")
    if not resvg:
        return None
    out_png = Path(svg_path).with_suffix(".resvg.png")
    try:
        subprocess.run(
            [resvg, "--width", str(int(width)), "--height", str(int(height)),
             "--background", "white", str(svg_path), str(out_png)],
            check=True, capture_output=True, timeout=60,
        )
        arr = _rgb_on_white(Image.open(out_png))
        return arr
    except Exception as e:  # noqa: BLE001
        logger.debug("resvg render atlandı (%s): %s", svg_path.name, e)
        return None
    finally:
        if out_png.exists():
            try:
                out_png.unlink()
            except OSError:
                pass


# Render backend sırası: en sağlam/taşınabilir olandan opsiyonel fallback'lere.
# PyMuPDF Windows'ta DLL'siz çalışır; CairoSVG/resvg varsa onlar da denenir.
_RENDER_BACKENDS = (_render_pymupdf, _render_cairosvg, _render_svglib, _render_resvg)


def render_svg_to_rgb(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """SVG'yi verilen boyutta beyaz zeminli RGB diziye render eder.

    Birden çok backend sırayla denenir; hiçbiri çalışmazsa ``None`` döner ve
    çağıran yapısal skorlara güvenle düşer (çökme yok).
    """
    for backend in _RENDER_BACKENDS:
        arr = backend(Path(svg_path), int(width), int(height))
        if arr is None:
            continue
        if arr.shape[0] != height or arr.shape[1] != width:
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
        return arr
    return None


# ---------------------------------------------------------------------------
# Metrik bileşenleri
# ---------------------------------------------------------------------------
def _ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Gaussian pencereli SSIM (gri tonlama, 0-1). scipy ile, scikit-image'sız."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = gaussian_filter(a * a, sigma) - mu_a2
    sigma_b2 = gaussian_filter(b * b, sigma) - mu_b2
    sigma_ab = gaussian_filter(a * b, sigma) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _ms_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Hafif çok-ölçekli SSIM: tam ve yarı çözünürlükte ortalama."""
    full = _ssim(gray_a, gray_b)
    h, w = gray_a.shape
    if min(h, w) >= 64:
        half_a = cv2.resize(gray_a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        half_b = cv2.resize(gray_b, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return 0.6 * full + 0.4 * _ssim(half_a, half_b)
    return full


def _mean_delta_e(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """CIELAB uzayında ortalama ΔE76 (Öklid). Düşük = renk olarak sadık."""
    lab_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = lab_a - lab_b
    delta = np.sqrt(np.sum(diff * diff, axis=2))
    return float(np.mean(delta))


def _edge_f1(gray_a: np.ndarray, gray_b: np.ndarray, tolerance: int = 2) -> float:
    """Toleranslı kenar uyumu (F1). a=render, b=orijinal kenarları.

    Precision: render kenarlarının kaçı orijinale yakın.
    Recall:    orijinal kenarlarının kaçı render'da var.
    """
    edges_a = cv2.Canny(gray_a.astype(np.uint8), 80, 160) > 0
    edges_b = cv2.Canny(gray_b.astype(np.uint8), 80, 160) > 0

    if not edges_a.any() and not edges_b.any():
        return 1.0  # iki tarafta da kenar yok -> tam uyum (düz alan)
    if not edges_a.any() or not edges_b.any():
        return 0.0

    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    edges_a_d = cv2.dilate(edges_a.astype(np.uint8), k) > 0
    edges_b_d = cv2.dilate(edges_b.astype(np.uint8), k) > 0

    precision = float(np.sum(edges_a & edges_b_d)) / float(np.sum(edges_a))
    recall = float(np.sum(edges_b & edges_a_d)) / float(np.sum(edges_b))
    if precision + recall < 1e-9:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Birleşik sadakat
# ---------------------------------------------------------------------------
def compute_fidelity(original_rgb: np.ndarray, rendered_rgb: np.ndarray) -> dict[str, Any]:
    """İki RGB dizi (aynı boyut) arasında algısal sadakat raporu üretir.

    Döner: ``fidelity_score`` (0-100) ve bileşenleri + hata haritası özetleri
    (Faz 1 refinement geçişi bunları kullanacak).
    """
    if original_rgb.shape != rendered_rgb.shape:
        h, w = original_rgb.shape[:2]
        rendered_rgb = cv2.resize(rendered_rgb, (w, h), interpolation=cv2.INTER_AREA)

    gray_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2GRAY)

    ssim = _ms_ssim(gray_r, gray_o)
    mean_de = _mean_delta_e(rendered_rgb, original_rgb)
    edge_f1 = _edge_f1(gray_r, gray_o)

    # 0-100 alt skorlar
    ssim_score = round(ssim * 100.0, 2)
    # ΔE76: ~2.3 algı eşiği (JND). Doğrusal ceza; ΔE 0->100, 20+->0.
    color_score = round(max(0.0, 100.0 - mean_de * 5.0), 2)
    edge_score = round(edge_f1 * 100.0, 2)

    fidelity_score = round(
        0.40 * ssim_score + 0.35 * color_score + 0.25 * edge_score, 2
    )

    # bölgesel hata haritası özeti (refinement için): ΔE eşiğini aşan piksel oranı
    lab_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    per_pixel_de = np.sqrt(np.sum((lab_o - lab_r) ** 2, axis=2))
    high_error_ratio = round(float(np.mean(per_pixel_de > 18.0)), 4)

    return {
        "fidelity_score": fidelity_score,
        "ssim": round(ssim, 4),
        "ssim_score": ssim_score,
        "mean_delta_e": round(mean_de, 3),
        "color_score": color_score,
        "edge_f1": round(edge_f1, 4),
        "edge_score": edge_score,
        "high_error_ratio": high_error_ratio,
    }


def score_svg_fidelity(
    svg_path: Path,
    original_path: Path,
    max_side: int = _COMPARE_MAX_SIDE,
) -> dict[str, Any] | None:
    """SVG'yi render edip orijinalle algısal sadakatini ölçer.

    Render mümkün değilse (CairoSVG yok / bozuk SVG) ``None`` döner.
    """
    try:
        reference, (w, h) = load_reference_rgb(Path(original_path), max_side=max_side)
    except Exception as e:  # noqa: BLE001
        logger.debug("Referans görsel yüklenemedi: %s", e)
        return None

    rendered = render_svg_to_rgb(Path(svg_path), w, h)
    if rendered is None:
        return None

    try:
        return compute_fidelity(reference, rendered)
    except Exception as e:  # noqa: BLE001
        logger.debug("Sadakat hesaplanamadı (%s): %s", Path(svg_path).name, e)
        return None
