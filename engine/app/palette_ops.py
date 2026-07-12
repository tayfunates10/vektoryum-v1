"""Bant bazlı (tiled) palet sınıflandırması: bellek sınırlı, bit-birebir.

3840² girdide tek parça sınıflandırma (H,W,K,3) float32 ara tensörü kurar
(~700 MB) ve argmin çıktısı int64 döner (~118 MB); tepe bellek bunlarla
şişiyordu (ölçüldü: 3346 MB). Piksel bağımsız bir işlem olduğundan satır
bantlarına bölmek çıktıyı BİT-BİREBİR korur (eleman başına aynı float
işlemleri; komşuluk/halo gereksinimi yok, tile dikişi imkânsız).

Bant yüksekliği bellek bütçesinden türetilir; fixture'a özel sabit yoktur.
``VEKTORYUM_TILED_CLASSIFY=off`` tek parça (eski) yola döndürür.
"""

from __future__ import annotations

import os

import numpy as np

# bant başına ara tensör bütçesi (byte) — (b, W, K, 3) float32 bundan küçük
_BAND_BUDGET_BYTES = 48 * 1024 * 1024


def _tiled_enabled() -> bool:
    return os.environ.get("VEKTORYUM_TILED_CLASSIFY", "on").strip().lower() not in {
        "off", "0", "false"
    }


def classify_features(img: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """(H,W,C) özellik görüntüsünü en yakın merkeze sınıflar; uint8 döner.

    ``img`` float32'ye çevrilir; ``centers`` (K,C). Bant bazlıdır ve tek
    parça hesapla bit-birebir aynıdır (piksel bağımsız işlem). K ≤ 255.
    """
    h, w = img.shape[:2]
    k, c = centers.shape
    cen = centers.astype(np.float32, copy=False)
    out = np.empty((h, w), dtype=np.uint8)
    if not _tiled_enabled():
        d = np.linalg.norm(img[:, :, None, :].astype(np.float32) - cen[None, None], axis=3)
        out[:] = np.argmin(d, axis=2).astype(np.uint8)
        return out
    band = max(16, int(_BAND_BUDGET_BYTES / max(1, w * k * c * 4)))
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        blk = img[y0:y1].astype(np.float32, copy=False)
        d = np.linalg.norm(blk[:, :, None, :] - cen[None, None], axis=3)
        out[y0:y1] = np.argmin(d, axis=2).astype(np.uint8)
        del d, blk
    return out


def classify_rgb(img: np.ndarray, fills_rgb: np.ndarray) -> np.ndarray:
    """RGB görüntüyü en yakın dolgu rengine sınıflar (uint8)."""
    return classify_features(img, fills_rgb)


def abs_diff_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a-b| kanal toplamı, (H,W) uint16 — büyük int32 ara tensörsüz.

    uint8 RGB girdiler için cv2.absdiff eşdeğeri; kanal toplamı ≤ 765
    olduğundan uint16 kayıpsızdır.
    """
    import cv2  # noqa: PLC0415

    d = cv2.absdiff(a, b)
    return d.sum(axis=2, dtype=np.uint16)
