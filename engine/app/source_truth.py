"""Alpha ve color-managed kaynak gerçekliği yardımcıları.

Bu modül tek bir beyaz-kompozit RGB görüntüyü kaynak gerçeği saymaz. Kaynağı
straight RGBA, premultiplied RGBA ve beyaz/siyah/checker görünümü olarak açıkça
ayırır. Final artifact ve transform journal aynı fonksiyonları kullanır.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_EPS = 1.0 / 65535.0


def rgba_sha256(rgba: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def source_rgba_from_white_composite(
    source_rgb: np.ndarray,
    source_alpha: np.ndarray | None,
) -> np.ndarray:
    """Beyaz kompozit RGB + alpha düzleminden straight RGBA oluşturur.

    Mevcut API zinciri şeffaf kaynağı evaluator'a beyaz kompozit RGB ve ayrı
    alpha olarak taşıyor. Alpha > 0 piksellerde straight renk analitik olarak
    geri çözülür. Tam şeffaf piksellerde renk tanımsız olduğundan sıfır yazılır;
    bu pikseller appearance ve halo metriklerinde ağırlık almaz.
    """
    rgb = np.asarray(source_rgb, dtype=np.uint8)
    if source_alpha is None:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        return np.dstack([rgb.copy(), alpha])

    alpha = np.asarray(source_alpha, dtype=np.uint8)
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("source_alpha boyutu source_rgb ile eşleşmiyor")

    af = alpha.astype(np.float32)[:, :, None] / 255.0
    comp = rgb.astype(np.float32)
    straight = np.zeros_like(comp, dtype=np.float32)
    valid = af[:, :, 0] > _EPS
    if np.any(valid):
        straight[valid] = (
            comp[valid] - 255.0 * (1.0 - af[valid])
        ) / np.maximum(af[valid], _EPS)
    straight = np.clip(np.rint(straight), 0, 255).astype(np.uint8)
    return np.dstack([straight, alpha])


def premultiply_rgba(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("RGBA dizi bekleniyor")
    out = arr.astype(np.float32)
    out[:, :, :3] *= out[:, :, 3:4] / 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def unpremultiply_rgba(premultiplied: np.ndarray) -> np.ndarray:
    arr = np.asarray(premultiplied, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("RGBA dizi bekleniyor")
    out = arr.astype(np.float32)
    alpha = out[:, :, 3:4] / 255.0
    valid = alpha[:, :, 0] > _EPS
    rgb = np.zeros_like(out[:, :, :3])
    if np.any(valid):
        rgb[valid] = out[:, :, :3][valid] / np.maximum(alpha[valid], _EPS)
    result = np.dstack([np.clip(np.rint(rgb), 0, 255), out[:, :, 3]])
    return result.astype(np.uint8)


def solid_background(height: int, width: int, value: int | tuple[int, int, int]) -> np.ndarray:
    if isinstance(value, int):
        color = (value, value, value)
    else:
        color = tuple(int(v) for v in value)
    bg = np.empty((height, width, 3), dtype=np.uint8)
    bg[:] = color
    return bg


def checker_background(height: int, width: int, cell: int | None = None) -> np.ndarray:
    cell = max(2, int(cell or max(4, round(max(height, width) / 16))))
    yy, xx = np.indices((height, width))
    mask = ((xx // cell + yy // cell) % 2).astype(bool)
    bg = np.empty((height, width, 3), dtype=np.uint8)
    bg[~mask] = (238, 238, 238)
    bg[mask] = (96, 96, 96)
    return bg


def composite_rgba(rgba: np.ndarray, background: np.ndarray | tuple[int, int, int] | int) -> np.ndarray:
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("RGBA dizi bekleniyor")
    h, w = arr.shape[:2]
    if isinstance(background, np.ndarray):
        bg = np.asarray(background, dtype=np.uint8)
        if bg.shape != (h, w, 3):
            raise ValueError("background boyutu RGBA ile eşleşmiyor")
    else:
        bg = solid_background(h, w, background)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    comp = arr[:, :, :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(comp), 0, 255).astype(np.uint8)


def _soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float32) / 255.0
    bf = np.asarray(b, dtype=np.float32) / 255.0
    denom = float(np.maximum(af, bf).sum())
    if denom <= 1e-12:
        return 1.0
    return float(np.minimum(af, bf).sum() / denom)


def _binary_iou(a: np.ndarray, b: np.ndarray, threshold: int = 1) -> float:
    aa = np.asarray(a, dtype=np.uint8) >= threshold
    bb = np.asarray(b, dtype=np.uint8) >= threshold
    union = int((aa | bb).sum())
    if union == 0:
        return 1.0
    return float((aa & bb).sum() / union)


def alpha_plane_metrics(source_alpha: np.ndarray, render_alpha: np.ndarray) -> dict[str, float]:
    sa = np.asarray(source_alpha, dtype=np.uint8)
    ra = np.asarray(render_alpha, dtype=np.uint8)
    if sa.shape != ra.shape:
        raise ValueError("alpha boyutları eşleşmiyor")
    diff = np.abs(sa.astype(np.float32) - ra.astype(np.float32)) / 255.0
    return {
        "alpha_iou": _soft_iou(sa, ra),
        "alpha_binary_iou": _binary_iou(sa, ra),
        "alpha_mae": float(diff.mean()),
        "alpha_p95": float(np.percentile(diff, 95)),
        "alpha_max": float(diff.max(initial=0.0)),
        "source_coverage": float(sa.astype(np.float32).mean() / 255.0),
        "render_coverage": float(ra.astype(np.float32).mean() / 255.0),
    }


def boundary_halo_metrics(source_rgba: np.ndarray, render_rgba: np.ndarray) -> dict[str, float | None]:
    src = np.asarray(source_rgba, dtype=np.uint8)
    rnd = np.asarray(render_rgba, dtype=np.uint8)
    if src.shape != rnd.shape:
        raise ValueError("RGBA boyutları eşleşmiyor")
    alpha = src[:, :, 3]
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    boundary = (np.hypot(gx, gy) > 1.0) & (alpha > 2)
    if not np.any(boundary):
        return {"halo_rgb_mae": None, "halo_rgb_p95": None, "halo_sample_count": 0}

    src_rgb = src[:, :, :3].astype(np.float32)
    rnd_rgb = rnd[:, :, :3].astype(np.float32)
    weight = np.maximum(alpha.astype(np.float32) / 255.0, 0.05)
    pixel_err = np.mean(np.abs(src_rgb - rnd_rgb), axis=2) / 255.0
    weighted = pixel_err[boundary] * weight[boundary]
    return {
        "halo_rgb_mae": float(weighted.mean()),
        "halo_rgb_p95": float(np.percentile(weighted, 95)),
        "halo_sample_count": int(boundary.sum()),
    }


def roundtrip_metrics(rgba: np.ndarray) -> dict[str, float]:
    arr = np.asarray(rgba, dtype=np.uint8)
    restored = unpremultiply_rgba(premultiply_rgba(arr))
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    valid = alpha > 0.02
    if not np.any(valid):
        return {"premultiplied_roundtrip_mae": 0.0, "premultiplied_roundtrip_max": 0.0}
    diff = np.mean(
        np.abs(arr[:, :, :3].astype(np.float32) - restored[:, :, :3].astype(np.float32)),
        axis=2,
    ) / 255.0
    return {
        "premultiplied_roundtrip_mae": float(diff[valid].mean()),
        "premultiplied_roundtrip_max": float(diff[valid].max(initial=0.0)),
    }


def resize_rgba(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Straight RGBA'yı premultiplied uzayda küçültür; halo üretmez."""
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.shape[:2] == (height, width):
        return arr.copy()
    premul = premultiply_rgba(arr)
    resized = cv2.resize(premul, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return unpremultiply_rgba(resized.astype(np.uint8))


def _png_bytes_to_rgba(data: bytes, width: int, height: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGBA")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def render_svg_to_rgba(svg_path: Path, width: int, height: int) -> np.ndarray | None:
    """SVG'yi şeffaf zeminli straight RGBA olarak render eder.

    resvg_py varsa production renderer ile aynı yolu kullanır; yoksa CairoSVG
    şeffaf PNG fallback'i kullanılır. Hiçbiri çalışmazsa ``None`` döner.
    """
    path = Path(svg_path)
    try:
        import resvg_py  # type: ignore  # noqa: PLC0415

        data = bytes(resvg_py.svg_to_bytes(
            svg_path=str(path), width=int(width), height=int(height),
        ))
        return _png_bytes_to_rgba(data, int(width), int(height))
    except Exception:
        pass
    try:
        import cairosvg  # noqa: PLC0415

        data = cairosvg.svg2png(
            url=str(path), output_width=int(width), output_height=int(height),
        )
        return _png_bytes_to_rgba(data, int(width), int(height))
    except Exception:
        return None


def multibackground_pairs(source_rgba: np.ndarray, render_rgba: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    h, w = source_rgba.shape[:2]
    backgrounds: dict[str, np.ndarray | int] = {
        "white": 255,
        "black": 0,
        "checker": checker_background(h, w),
    }
    return {
        name: (composite_rgba(source_rgba, bg), composite_rgba(render_rgba, bg))
        for name, bg in backgrounds.items()
    }


def public_metric_dict(metric: dict[str, Any]) -> dict[str, Any]:
    """Journal cache'inde tutulabilecek özel ndarray alanlarını rapordan çıkarır."""
    return {k: v for k, v in metric.items() if not k.startswith("_")}
