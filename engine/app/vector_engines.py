"""Vektör motorları ve aday üretimi.

Birincil motor VTracer'dir (Python binding). Geometrik logolar için OpenCV
contour polygonization adayı, ikili görseller için opsiyonel Potrace ve
centerline için opsiyonel AutoTrace CLI adayları desteklenir.

Tüm harici CLI'lar opsiyoneldir; bulunmazsa ilgili aday atlanır ve sistem
çökmeden devam eder.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI tespiti (env değişkeni > PATH)
# ---------------------------------------------------------------------------
def _resolve_cli(tool_name: str, env_var: str) -> str | None:
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        logger.info("'%s' env değişkeninden bulundu: %s", tool_name, env_path)
        return env_path
    path = shutil.which(tool_name)
    if path:
        logger.info("'%s' PATH'te bulundu: %s", tool_name, path)
        return path
    logger.warning("'%s' bulunamadı. İlgili adaylar atlanacak.", tool_name)
    return None


def get_potrace_path() -> str | None:
    return _resolve_cli("potrace", "POTRACE_PATH")


def get_autotrace_path() -> str | None:
    return _resolve_cli("autotrace", "AUTOTRACE_PATH")


def get_cli_path(tool_name: str) -> str | None:
    return _resolve_cli(tool_name, f"{tool_name.upper()}_PATH")


# Modül yükleme anında bir kez tespit et (geriye dönük uyumluluk için sabitler)
POTRACE_PATH = get_potrace_path()
AUTOTRACE_PATH = get_autotrace_path()


# ---------------------------------------------------------------------------
# Aday tanımları
# ---------------------------------------------------------------------------
def _vt(colormode: str = "color", mode: str = "spline", **kw: Any) -> dict[str, Any]:
    """VTracer parametre sözlüğü kısayolu. length_threshold >= 3.5 olmalı."""
    params: dict[str, Any] = {"colormode": colormode, "mode": mode}
    params.update(kw)
    if "length_threshold" in params:
        params["length_threshold"] = max(3.5, float(params["length_threshold"]))
    return params


def build_vector_candidates(mode: str) -> dict[str, dict[str, Any]]:
    """Verilen profile göre üretilecek adayları ve parametrelerini döndürür.

    Her aday gerçekten farklı bir SVG üretecek şekilde ayarlanmıştır
    (motor / VTracer parametreleri / cleanup seviyesi farklı).
    """
    if mode == "geometric_logo":
        return {
            "geo_clean": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "polygon", color_precision=4, filter_speckle=10,
                                      layer_difference=32, corner_threshold=85, length_threshold=8.0,
                                      path_precision=2),
                "cleanup": "aggressive",
            },
            "geo_standard": {
                # spline: yuvarlak şekiller bezier eğrisi olur; düşük corner_threshold
                # sayesinde GERÇEK köşeler keskin kalır (oval yuvarlatılmaz)
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=5, filter_speckle=6,
                                      layer_difference=24, corner_threshold=32, length_threshold=3.5,
                                      splice_threshold=45, path_precision=6),
                "cleanup": "standard",
            },
            "geo_detail": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=6, filter_speckle=3,
                                      layer_difference=16, corner_threshold=24, length_threshold=3.5,
                                      splice_threshold=45, path_precision=5),
                "cleanup": "light",
            },
            "geo_mixed": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=5, filter_speckle=6,
                                      layer_difference=24, corner_threshold=42, length_threshold=4.5,
                                      path_precision=3),
                "cleanup": "balanced",
            },
            "geo_contour": {
                "engine": "opencv_contour",
                "params": {"epsilon": 1.4, "min_area": 10.0, "palette_mode": "auto"},
                "cleanup": "standard",
            },
            "geo_potrace": {
                "engine": "potrace",
                "params": {"turdsize": 2, "alphamax": 0.7, "opttolerance": 0.2},
                "cleanup": "standard",
                "optional": True,
            },
        }

    if mode in ("minimal_ai", "flat_logo"):
        return {
            "minimal_clean": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "polygon", color_precision=6, filter_speckle=8,
                                      corner_threshold=75, length_threshold=6.0, path_precision=3),
                "cleanup": "standard",
            },
            "minimal_standard": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=7, filter_speckle=5,
                                      corner_threshold=60, length_threshold=4.5, path_precision=4),
                "cleanup": "light",
            },
            "minimal_detail": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=8, filter_speckle=3,
                                      corner_threshold=50, length_threshold=3.5, path_precision=5),
                "cleanup": None,
            },
            "minimal_contour": {
                "engine": "opencv_contour",
                "params": {"epsilon": 1.0, "min_area": 8.0, "palette_mode": "auto"},
                "cleanup": "light",
            },
            "minimal_potrace": {
                "engine": "potrace",
                "params": {"turdsize": 2, "alphamax": 1.0, "opttolerance": 0.2},
                "cleanup": "light",
                "optional": True,
            },
        }

    if mode == "logo_color":
        return {
            "logo_clean": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=4, filter_speckle=8,
                                      layer_difference=32, corner_threshold=60, length_threshold=4.5,
                                      path_precision=4),
                "cleanup": None,
            },
            "logo_standard": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=5, filter_speckle=4,
                                      layer_difference=24, corner_threshold=55, length_threshold=4.0,
                                      path_precision=5),
                "cleanup": None,
            },
            "logo_detail_rich": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=6, filter_speckle=2,
                                      layer_difference=16, corner_threshold=50, length_threshold=3.5,
                                      path_precision=6),
                "cleanup": None,
            },
            "logo_color_preserve": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=6, filter_speckle=3,
                                      corner_threshold=55, length_threshold=3.5, layer_difference=12,
                                      path_precision=5),
                "cleanup": None,
            },
            "logo_smooth": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=4, filter_speckle=6,
                                      layer_difference=32, corner_threshold=70, length_threshold=5.0,
                                      splice_threshold=60, path_precision=4),
                "cleanup": None,
            },
        }

    if mode == "lineart":
        return {
            "lineart_clean": {
                "engine": "vtracer",
                "vtracer_params": _vt("binary", "polygon", filter_speckle=6,
                                      corner_threshold=70, length_threshold=5.0, path_precision=3),
                "cleanup": "standard",
            },
            "lineart_detail": {
                "engine": "vtracer",
                "vtracer_params": _vt("binary", "spline", filter_speckle=3,
                                      corner_threshold=55, length_threshold=3.5, path_precision=4),
                "cleanup": "light",
            },
            "lineart_potrace": {
                "engine": "potrace",
                "params": {"turdsize": 2, "alphamax": 1.0, "opttolerance": 0.2},
                "cleanup": "light",
                "optional": True,
            },
            "lineart_autotrace": {
                "engine": "autotrace",
                "params": {"centerline": False},
                "cleanup": None,
                "optional": True,
            },
        }

    if mode == "single_color":
        return {
            "single_clean": {
                "engine": "vtracer",
                "vtracer_params": _vt("binary", "polygon", filter_speckle=8,
                                      corner_threshold=80, length_threshold=6.0, path_precision=3),
                "cleanup": "standard",
            },
            "single_contour": {
                "engine": "opencv_contour",
                "params": {"epsilon": 1.2, "min_area": 12.0, "palette_mode": "binary"},
                "cleanup": "standard",
            },
            "single_potrace": {
                "engine": "potrace",
                "params": {"turdsize": 3, "alphamax": 1.0, "opttolerance": 0.2},
                "cleanup": "standard",
                "optional": True,
            },
        }

    if mode == "centerline":
        return {
            "centerline_autotrace": {
                "engine": "autotrace",
                "params": {"centerline": True},
                "cleanup": None,
                "optional": True,
            },
            "centerline_skeleton": {
                "engine": "opencv_skeleton",
                "params": {"min_branch": 6},
                "cleanup": None,
            },
        }

    if mode == "photo_poster":
        return {
            "photo_standard": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=6, filter_speckle=8,
                                      corner_threshold=60, length_threshold=4.5, path_precision=4),
                "cleanup": None,
            },
            "photo_detail": {
                "engine": "vtracer",
                "vtracer_params": _vt("color", "spline", color_precision=8, filter_speckle=4,
                                      corner_threshold=55, length_threshold=3.5, path_precision=5),
                "cleanup": None,
            },
        }

    # auto veya bilinmeyen -> güvenli varsayılan
    return {
        "default_standard": {
            "engine": "vtracer",
            "vtracer_params": _vt("color", "spline", color_precision=7, filter_speckle=4),
            "cleanup": None,
        },
        "default_clean": {
            "engine": "vtracer",
            "vtracer_params": _vt("color", "polygon", color_precision=6, filter_speckle=8,
                                  corner_threshold=70, length_threshold=5.0),
            "cleanup": "standard",
        },
    }


# ---------------------------------------------------------------------------
# VTracer
# ---------------------------------------------------------------------------
def vectorize_with_vtracer(input_path: Path, output_path: Path, params: dict[str, Any]) -> None:
    """VTracer Python binding ile görseli SVG'ye dönüştürür."""
    try:
        import vtracer
    except ImportError as e:
        raise RuntimeError("vtracer kütüphanesi kurulu değil.") from e

    kwargs = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        vtracer.convert_image_to_svg_py(str(input_path), str(output_path), **kwargs)
    except TypeError:
        # bilinmeyen parametreleri ele: yalnızca temel anahtarlarla yeniden dene
        safe = {k: kwargs[k] for k in ("colormode", "mode", "color_precision", "filter_speckle",
                                       "corner_threshold", "length_threshold", "path_precision")
                if k in kwargs}
        vtracer.convert_image_to_svg_py(str(input_path), str(output_path), **safe)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("VTracer boş çıktı üretti.")
    logger.info("VTracer ile '%s' üretildi.", output_path.name)


# ---------------------------------------------------------------------------
# OpenCV contour polygonization
# ---------------------------------------------------------------------------
def _quantized_palette(image_bgr: np.ndarray, max_colors: int = 16) -> list[tuple[int, int, int]]:
    """Görseldeki baskın renkleri (BGR) döndürür."""
    data = image_bgr.reshape(-1, 3)
    # zaten palet indirgenmiş bir görsel beklenir; benzersiz renkleri say
    colors, counts = np.unique(data, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    colors = colors[order][:max_colors]
    return [tuple(int(c) for c in col) for col in colors]


def vectorize_geometric_contours_to_svg(
    clean_image_path: Path,
    svg_output_path: Path,
    palette_mode: str = "auto",
    epsilon: float = 1.2,
    min_area: float = 8.0,
) -> None:
    """OpenCV contour detection + approxPolyDP ile renk bölgelerinden SVG üretir.

    Her baskın renk için ayrı maske çıkarılır, contour bulunur, polygon
    sadeleştirilir ve ``fill-rule="evenodd"`` ile delikler korunur.
    """
    image = cv2.imread(str(clean_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Görsel okunamadı: {clean_image_path}")

    # alfa kanalını beyaz zemine indir
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        image = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    total_px = height * width

    if palette_mode == "binary":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        palette = [(0, 0, 0)]
        masks = {(0, 0, 0): binm}
    else:
        palette = _quantized_palette(image, max_colors=16)
        masks = {}
        for color in palette:
            mask = cv2.inRange(image, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))
            masks[color] = mask

    svg_paths: list[str] = []
    for color in palette:
        mask = masks[color]
        coverage = float(np.count_nonzero(mask)) / max(total_px, 1)
        mean_c = float(np.mean(color))
        # büyük ve neredeyse beyaz alan = arka plan -> atla
        if palette_mode != "binary" and coverage > 0.35 and mean_c > 245:
            continue
        if np.count_nonzero(mask) == 0:
            continue

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        path_data = ""
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            eps = max(0.5, epsilon * peri / 100.0)
            poly = cv2.approxPolyDP(contour, eps, True)
            if len(poly) < 3:
                continue
            path_data += f"M {int(poly[0][0][0])} {int(poly[0][0][1])} "
            for pt in poly[1:]:
                path_data += f"L {int(pt[0][0])} {int(pt[0][1])} "
            path_data += "Z "

        if path_data:
            b, g, r = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            svg_paths.append(f'<path fill="{hex_color}" fill-rule="evenodd" d="{path_data.strip()}"/>')

    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'{"".join(svg_paths)}</svg>'
    )
    svg_output_path.write_text(svg_content, encoding="utf-8")
    if not svg_paths:
        raise RuntimeError("OpenCV contour hiç path üretemedi.")
    logger.info("OpenCV contour ile '%s' üretildi (%d path).", svg_output_path.name, len(svg_paths))


# ---------------------------------------------------------------------------
# Potrace (opsiyonel CLI)
# ---------------------------------------------------------------------------
def convert_mask_to_pbm(image_path: Path, pbm_path: Path) -> Path:
    """Bir görseli ikili PBM'e çevirir (Potrace girdisi)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Maske okunamadı: {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Potrace siyahı (0) ön plan kabul eder; koyu konu -> siyah kalsın
    cv2.imwrite(str(pbm_path), binary)
    return pbm_path


def vectorize_with_potrace_cli(input_path: Path, output_path: Path, params: dict[str, Any] | None = None) -> None:
    """Potrace CLI ile ikili görseli SVG'ye dönüştürür. Potrace yoksa hata fırlatır."""
    potrace = get_potrace_path()
    if not potrace:
        raise FileNotFoundError("potrace not found")

    params = params or {}
    pbm_path = Path(output_path).with_suffix(".pbm")
    convert_mask_to_pbm(Path(input_path), pbm_path)
    command = [
        potrace, str(pbm_path), "-s", "-o", str(output_path),
        "--turdsize", str(params.get("turdsize", 2)),
        "--alphamax", str(params.get("alphamax", 1.0)),
        "--opttolerance", str(params.get("opttolerance", 0.2)),
        "--turnpolicy", str(params.get("turnpolicy", "minority")),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"potrace error: {e.stderr.strip() if e.stderr else e}") from e
    finally:
        if pbm_path.exists():
            try:
                pbm_path.unlink()
            except OSError:
                pass
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("potrace boş çıktı üretti.")
    logger.info("Potrace ile '%s' üretildi.", output_path.name)


# ---------------------------------------------------------------------------
# AutoTrace (opsiyonel CLI)
# ---------------------------------------------------------------------------
def vectorize_with_autotrace_cli(input_path: Path, output_path: Path, params: dict[str, Any] | None = None) -> None:
    """AutoTrace CLI ile SVG üretir. centerline=True ise --centerline kullanır."""
    autotrace = get_autotrace_path()
    if not autotrace:
        raise FileNotFoundError("autotrace not found")

    params = params or {}
    command = [autotrace, "-output-format", "svg", "-output-file", str(output_path)]
    if params.get("centerline"):
        command.append("-centerline")
    command.append(str(input_path))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"autotrace error: {e.stderr.strip() if e.stderr else e}") from e
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("autotrace boş çıktı üretti.")
    logger.info("AutoTrace ile '%s' üretildi.", output_path.name)


# ---------------------------------------------------------------------------
# OpenCV skeleton (centerline fallback)
# ---------------------------------------------------------------------------
def vectorize_skeleton_to_svg(input_path: Path, output_path: Path, params: dict[str, Any] | None = None) -> None:
    """Morfolojik skeleton tabanlı basit centerline fallback (placeholder kalite)."""
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Görsel okunamadı: {input_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    for _ in range(2000):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    height, width = binary.shape
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    svg_paths: list[str] = []
    for contour in contours:
        if len(contour) < 2:
            continue
        poly = cv2.approxPolyDP(contour, 1.0, False)
        if len(poly) < 2:
            continue
        d = f"M {int(poly[0][0][0])} {int(poly[0][0][1])} "
        for pt in poly[1:]:
            d += f"L {int(pt[0][0])} {int(pt[0][1])} "
        svg_paths.append(f'<path fill="none" stroke="#000000" stroke-width="1" d="{d.strip()}"/>')

    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{"".join(svg_paths)}</svg>'
    )
    output_path.write_text(svg_content, encoding="utf-8")
    if not svg_paths:
        raise RuntimeError("skeleton hiç path üretemedi.")
    logger.info("Skeleton centerline ile '%s' üretildi.", output_path.name)


# ---------------------------------------------------------------------------
# Aday yürütme dağıtıcısı
# ---------------------------------------------------------------------------
def run_candidate(
    engine: str,
    input_path: Path,
    output_path: Path,
    candidate: dict[str, Any],
    original_path: Path | None = None,
) -> None:
    """Bir adayı motoruna göre çalıştırır. Hata fırlatabilir (main yakalar).

    ``original_path``: gradyan motoru gibi posterize EDİLMEMİŞ pikselleri gereken
    motorlar için ham görsel yolu. Verilmezse ``input_path`` kullanılır.
    """
    if engine == "vtracer":
        vectorize_with_vtracer(input_path, output_path, candidate.get("vtracer_params", {}))
    elif engine == "opencv_contour":
        vectorize_geometric_contours_to_svg(input_path, output_path, **candidate.get("params", {}))
    elif engine == "gradient":
        from app.gradient_vectorize import vectorize_with_gradients
        vectorize_with_gradients(original_path or input_path, output_path, candidate.get("params", {}))
    elif engine == "potrace":
        vectorize_with_potrace_cli(input_path, output_path, candidate.get("params", {}))
    elif engine == "autotrace":
        vectorize_with_autotrace_cli(input_path, output_path, candidate.get("params", {}))
    elif engine == "opencv_skeleton":
        vectorize_skeleton_to_svg(input_path, output_path, candidate.get("params", {}))
    else:
        raise RuntimeError(f"Bilinmeyen motor: {engine}")
