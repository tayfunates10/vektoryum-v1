"""Shadow half-edge graph için TEK canonical segmentation kaynağı (SHADOW).

Şartname kuralı: "Aynı görsel için production vectorization labels ve shadow
graph labels farklı algoritmalarla yeniden üretilmemelidir." Bu modül tek bir
palet-farkındalıklı sınıflandırma yolu sağlar ve production'ın sınıflandırma
primitifini (`palette_ops.classify_rgb`) yeniden kullanır — ayrı bir k-means
kolu tutmaz. Palet merkezleri verilmezse kaynaktan determinist türetilir
(cv2.kmeans, sabit seed) ve gerçek renklere snap'lenir.

Üretilen etiketler DEĞİŞMEZ girdi kabul edilir; consolidation ve graph kurulumu
türetilmiş kopyalar üzerinde çalışır (kaynak raster/production label map yerinde
mutate EDİLMEZ).
"""
from __future__ import annotations

import numpy as np

_KMEANS_SEED = 7


def derive_palette(rgb: np.ndarray, k: int = 4) -> np.ndarray:
    """Kaynaktan determinist palet (K,3 uint8) türetir; gerçek piksele snap.

    cv2.kmeans (sabit seed) LAB uzayında merkez bulur; her merkez en yakın
    GERÇEK kaynak rengine snap'lenir (palet exact-snap deseni). Determinist.
    """
    import cv2  # noqa: PLC0415

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    step = max(1, lab.shape[0] // 60000)
    sub = lab[::step]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.4)
    cv2.setRNGSeed(_KMEANS_SEED)
    _c, _l, centers_lab = cv2.kmeans(sub, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    # her LAB merkezini en yakın gerçek kaynak rengine snap (determinist)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float32)
    flat_lab = lab
    fills: list[tuple[int, int, int]] = []
    for c in centers_lab:
        d = np.linalg.norm(flat_lab - c[None, :], axis=1)
        px = flat_rgb[int(np.argmin(d))]
        fills.append((int(round(px[0])), int(round(px[1])), int(round(px[2]))))
    # renk sırasını determinist yap (koyudan açığa) — ID kararlılığı
    fills.sort(key=lambda t: (t[0] + t[1] + t[2], t[0], t[1], t[2]))
    return np.array(fills, dtype=np.uint8)


def canonical_segmentation(rgb: np.ndarray,
                           fills_rgb: np.ndarray | None = None,
                           k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Kaynağı palet sınıflarına ayırır. Döner: (labels HxW uint8, fills_rgb Kx3).

    ``fills_rgb`` verilmezse determinist türetilir. Sınıflandırma production
    primitifi ``palette_ops.classify_rgb`` ile yapılır (bant-bazlı, bit-birebir).
    """
    from app.palette_ops import classify_rgb  # noqa: PLC0415

    if fills_rgb is None:
        fills_rgb = derive_palette(rgb, k)
    labels = classify_rgb(rgb.astype(np.uint8, copy=False),
                          fills_rgb.astype(np.float32))
    return labels.astype(np.uint8, copy=False), fills_rgb.astype(np.uint8)


def fills_to_hex(fills_rgb: np.ndarray) -> list[str]:
    """(K,3) paleti hex listesine çevirir (face dolgu rengi için)."""
    return ["#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            for r, g, b in fills_rgb]
