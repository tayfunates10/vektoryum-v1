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


def _render_resvg_py(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """resvg (Rust) Python binding. Referans kalite SVG render motoru — gradyan,
    pattern, clip dahil tam destek. Gradyan adaylarının doğru puanlanması için
    BİRİNCİL backend budur (PyMuPDF gradyanları siyah render ediyor).
    """
    try:
        import resvg_py  # noqa: PLC0415

        data = bytes(resvg_py.svg_to_bytes(
            svg_path=str(svg_path), width=int(width), height=int(height),
        ))
        return _rgb_on_white(Image.open(io.BytesIO(data)))
    except Exception as e:  # noqa: BLE001
        logger.debug("resvg_py render atlandı (%s): %s", svg_path.name, e)
        return None


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


# Render backend sırası: en doğru (gradyan dahil) olandan sağlam fallback'lere.
# resvg gradyanı doğru render eder; PyMuPDF DLL'siz ama gradyanları siyah çizer
# (gradyan adaylarında yalnız fallback olarak iş görür).
_RENDER_BACKENDS = (_render_resvg_py, _render_pymupdf, _render_cairosvg, _render_svglib, _render_resvg)


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
def _component_class_report(
    original_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    k: int = 6,
) -> dict[str, Any] | None:
    """Palet-sınıfı bağlı-bileşen raporu: EN KÖTÜ bileşen IoU'su.

    Global skorlar (SSIM/ΔE/edge) alan ağırlıklıdır: ® gibi küçük ama anlamlı
    bir bileşen tamamen bozulsa bile genel skor ~%99 kalır ve aday seçimi
    hatayı görmez (LEGO ® vakası: global %99.5, ® bölgesi IoU %46). Bu rapor
    orijinali küçük K ile (LAB) kümeler, her iki görüntüyü en yakın merkeze
    sınıflar ve orijinal sınıf maskelerinin HER bağlı bileşeni için render
    maskesiyle bileşen-yerel IoU ölçer.

    Yalnız düz-renk/palet karakterli görsellerde anlamlıdır: kuantalama artığı
    (medyan LAB uzaklığı) yüksekse ``None`` döner ve çağıran eski formüle düşer
    (foto/gradyan girdiler cezalandırılmaz). AA kırıntıları alan tabanıyla elenir.
    """
    try:
        h, w = original_rgb.shape[:2]
        lab_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_r = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        samples = lab_o.reshape(-1, 3)
        # hızlı kümeleme için alt örnekleme (deterministik adım)
        step = max(1, samples.shape[0] // 40000)
        sub = samples[::step]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(7)  # kmeans deterministik olsun (skorlama tekrarlanabilir)
        _compact, _labels, centers = cv2.kmeans(
            sub, k, None, crit, 3, cv2.KMEANS_PP_CENTERS
        )
        def _classify(lab: np.ndarray) -> np.ndarray:
            d = np.linalg.norm(lab[:, :, None, :] - centers[None, None, :, :], axis=3)
            return np.argmin(d, axis=2)
        cls_o = _classify(lab_o)
        cls_r = _classify(lab_r)
        # palet karakteri kontrolü: kuantalama artığı büyükse (foto) rapor üretme
        resid = np.take(centers, cls_o.reshape(-1), axis=0).reshape(lab_o.shape)
        med_err = float(np.median(np.linalg.norm(lab_o - resid, axis=2)))
        if med_err > 10.0:
            return None
        min_area = max(48, int(0.0001 * h * w))
        ious: list[float] = []
        worst = None
        weak: list[dict[str, Any]] = []
        for ci in range(centers.shape[0]):
            mo = (cls_o == ci).astype(np.uint8)
            if int(mo.sum()) < min_area:
                continue
            mr = cls_r == ci
            n, lab_map, stats, _ = cv2.connectedComponentsWithStats(mo, 8)
            for i in range(1, n):
                x, y, ww, hh, area = stats[i]
                if area < min_area:
                    continue
                pad = 6
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + ww + pad), min(h, y + hh + pad)
                mm = lab_map[y0:y1, x0:x1] == i
                rr = mr[y0:y1, x0:x1]
                # yerellik: aynı sınıfın KOMŞU bileşenleri (ör. çerçeve bbox'ı
                # tüm tuvali kapsar, içine harfler girer) birleşimi şişirmesin.
                # Render pikselleri yalnız kaynak bileşenin ~4px komşuluğunda
                # sayılır; daha büyük kayma zaten kesişim kaybıyla cezalanır.
                near = cv2.dilate(mm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                rr = rr & near
                union = int((mm | rr).sum())
                if union == 0:
                    continue
                iou = float((mm & rr).sum()) / union
                ious.append(iou)
                if worst is None or iou < worst[0]:
                    worst = (iou, [int(x), int(y), int(ww), int(hh)])
                # zayıf KÜÇÜK bileşen: hizalama refiti adayı. Kestirim GRUP
                # bazlıdır (penceredeki sınıfın tümü — ör. halka + R birlikte):
                # render tarafında bileşen ayrımı olmadığından tek bileşenle
                # centroid karşılaştırmak yanıltır. Ölçek, alan yerine RMS
                # yarıçap oranından kestirilir; AA incelmesi (kaynak küçültme
                # yumuşatması ince şekillerin sınıf alanını düşürür) alanı
                # bozar ama RMS yarıçapı korur.
                small = max(ww, hh) <= 0.18 * max(w, h)
                if iou < 0.88 and small and len(weak) < 6:
                    pad2 = max(8, int(0.35 * max(ww, hh)))
                    ax0, ay0 = max(0, x - pad2), max(0, y - pad2)
                    ax1, ay1 = min(w, x + ww + pad2), min(h, y + hh + pad2)
                    grp_o = cls_o[ay0:ay1, ax0:ax1] == ci
                    grp_r = mr[ay0:ay1, ax0:ax1]
                    if grp_r.sum() >= min_area * 0.4 and grp_o.sum() >= min_area * 0.4:
                        ys, xs = np.nonzero(grp_o)
                        yr, xr = np.nonzero(grp_r)
                        dx = float(xs.mean() - xr.mean())
                        dy = float(ys.mean() - yr.mean())
                        rms_o = float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))
                        rms_r = float(np.sqrt(((xr - xr.mean()) ** 2 + (yr - yr.mean()) ** 2).mean()))
                        s = rms_o / max(rms_r, 1e-6)
                        weak.append({
                            "iou": round(iou, 4),
                            "bbox": [int(x), int(y), int(ww), int(hh)],
                            "dx": round(dx, 2), "dy": round(dy, 2),
                            "scale": round(s, 4),
                        })
        if not ious:
            return None
        return {
            "component_min_iou": round(min(ious), 4),
            "component_mean_iou": round(float(np.mean(ious)), 4),
            "components_measured": len(ious),
            "worst_component_bbox": worst[1] if worst else None,
            "weak_components": weak,
        }
    except Exception as e:  # noqa: BLE001 (ölçüm başarısızsa eski davranışa dön)
        logger.debug("Bileşen raporu hesaplanamadı: %s", e)
        return None


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


def score_structure_integrity(
    svg_path: Path,
    original_path: Path,
    bg_rgb: tuple[int, int, int] | None = None,
    max_side: int = 1024,
) -> dict[str, Any] | None:
    """Vektör çıktının YAPI bütünlüğünü ölçer: kopan/kaybolan çizgi var mı?

    Orijinal rasterdeki mürekkep (zeminden ayrışan) piksellerin render'da
    karşılanma oranı (``ink_recall``) düşükse çizgiler KIRIK/eksik demektir;
    ``ink_precision`` düşükse çıktıda hayalet çizik/leke vardır. Bağlı bileşen
    farkı (``component_delta``) pozitifse şekiller parçalanmış, çok negatifse
    ayrık şekiller birbirine yapışmıştır. Ölçüm 1px toleranslıdır (dilate).

    Render backend'i yoksa ``None`` döner (çökme yok). Zemin rengi verilmezse
    orijinalin köşe medyanı kullanılır.
    """
    try:
        reference, (w, h) = load_reference_rgb(Path(original_path), max_side=max_side)
    except Exception as e:  # noqa: BLE001
        logger.debug("Yapı ölçümü: referans yüklenemedi: %s", e)
        return None

    rendered = render_svg_to_rgb(Path(svg_path), w, h)
    if rendered is None:
        return None

    try:
        if bg_rgb is None:
            pw, ph = max(4, w // 12), max(4, h // 12)
            corners = np.concatenate([
                reference[:ph, :pw].reshape(-1, 3), reference[:ph, -pw:].reshape(-1, 3),
                reference[-ph:, :pw].reshape(-1, 3), reference[-ph:, -pw:].reshape(-1, 3),
            ]).astype(np.float32)
            bg = np.median(corners, axis=0)
        else:
            bg = np.array(bg_rgb, dtype=np.float32)

        def _ink(rgb: np.ndarray) -> np.ndarray:
            dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
            return (dist > 60.0).astype(np.uint8)

        mo = _ink(reference)
        mr = _ink(rendered)
        n_o_ink = int(mo.sum())
        if n_o_ink < 30:
            return None  # ölçülecek anlamlı mürekkep yok
        k = np.ones((3, 3), np.uint8)
        recall = float((mo & (cv2.dilate(mr, k) > 0)).sum()) / float(n_o_ink)
        precision = float((mr & (cv2.dilate(mo, k) > 0)).sum()) / max(1.0, float(mr.sum()))
        n_o, _ = cv2.connectedComponents(mo, connectivity=8)
        n_r, _ = cv2.connectedComponents(mr, connectivity=8)
        return {
            "ink_recall": round(recall, 4),
            "ink_precision": round(precision, 4),
            "components_original": int(n_o - 1),
            "components_rendered": int(n_r - 1),
            "component_delta": int(n_r - n_o),
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("Yapı ölçümü hesaplanamadı (%s): %s", Path(svg_path).name, e)
        return None


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
        report = compute_fidelity(reference, rendered)
    except Exception as e:  # noqa: BLE001
        logger.debug("Sadakat hesaplanamadı (%s): %s", Path(svg_path).name, e)
        return None

    # Bileşen-ağırlıklı ceza: küçük ama anlamlı bir bileşen ciddi bozuksa
    # (ör. ® halkası dikeyde şişmişse) global skor bunu maskelememeli.
    # 512'de ince şekiller (glif konturları) sınıflandırma asimetrisiyle
    # YANLIŞ düşük ölçülebilir; bu yüzden şüphe (min<0.92) yalnızca daha
    # yüksek çözünürlükte (1024, kalınlık 2x) İKİNCİ ölçümle doğrulanırsa
    # cezaya dönüşür — ekstra render yalnız şüphe durumunda yapılır.
    try:
        comp = _component_class_report(reference, rendered)
        if comp is not None and comp["component_min_iou"] < 0.92:
            ref_hi, (w2, h2) = load_reference_rgb(Path(original_path), max_side=1024)
            rnd_hi = render_svg_to_rgb(Path(svg_path), w2, h2)
            if rnd_hi is not None:
                comp_hi = _component_class_report(ref_hi, rnd_hi)
                if comp_hi is not None:
                    comp = comp_hi
        if comp is not None:
            shortfall = max(0.0, 0.92 - comp["component_min_iou"])
            penalty = round(min(12.0, shortfall * 12.0), 2)
            if penalty > 0:
                report["fidelity_score"] = round(
                    max(0.0, report["fidelity_score"] - penalty), 2
                )
            comp["component_penalty"] = penalty
            report["component_report"] = comp
    except Exception as e:  # noqa: BLE001 (ölçüm hatası cezasız eski davranış)
        logger.debug("Bileşen cezası hesaplanamadı (%s): %s", Path(svg_path).name, e)
    return report
