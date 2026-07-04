from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def resize_for_analysis(image: Image.Image, max_side: int = 700) -> Image.Image:
    img = image.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _rgba_to_rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _color_distance(c1: list[int], c2: list[int]) -> float:
    return float(np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)))


def _merge_near_colors(colors: list[dict[str, Any]], distance_threshold: int = 22) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for color in colors:
        rgb = color["rgb"]
        ratio = color["ratio"]
        matched = False

        for item in merged:
            if _color_distance(rgb, item["rgb"]) <= distance_threshold:
                old_ratio = item["ratio"]
                new_ratio = old_ratio + ratio

                item["rgb"] = [
                    int(round((item["rgb"][i] * old_ratio + rgb[i] * ratio) / new_ratio))
                    for i in range(3)
                ]
                item["ratio"] = round(float(new_ratio), 4)
                matched = True
                break

        if not matched:
            merged.append({"rgb": rgb, "ratio": ratio})

    return sorted(merged, key=lambda item: item["ratio"], reverse=True)


def _point_segment_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """p noktasının [a, b] doğru parçasına dik uzaklığı ve izdüşüm oranı t.

    t=0 -> a ucunda, t=1 -> b ucunda. RGB uzayında anti-alias karışım testi
    için kullanılır (karışım rengi iki gerçek rengin arasında durur).
    """
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / denom)
    t_clamped = min(1.0, max(0.0, t))
    closest = a + t_clamped * ab
    return float(np.linalg.norm(p - closest)), t


def _aa_film_indexes(
    small_rgb: np.ndarray,
    colors: list[dict[str, Any]],
    colinear_tol: float = 26.0,
    max_survival: float = 0.25,
    anchor_factor: float = 0.5,
) -> set[int]:
    """Anti-alias KARIŞIM filmi olan baskın renk indekslerini bulur.

    Bir renk üç koşulu birden sağlıyorsa anti-alias kalıntısıdır, gerçek bir
    tasarım rengi değildir:

    1. Karşılaştırılabilir baskınlıkta iki rengin RGB doğrusu ÜZERİNDE durur
       (karışım rengi).
    2. Görüntüde yalnızca ince bir şerit/film olarak yaşar (1px erozyonda
       kaybolur). Kalın gri bölgeler hayatta kaldığı için gerçek gri tasarım
       renkleri film sayılmaz.
    3. Uç renklerin İKİSİNE de komşudur (film iki bölge ARASINDA yaşar).
       Beyaz zeminde tek başına duran ince gri çizgi uzak uca (siyaha) komşu
       olmadığından film sayılmaz; gerçek bir çizgidir.

    Bu tonlar elenmezse renk sayısı şişer ve sade logolar yanlışlıkla 'çok
    renkli' görünür (ince kenarlıklı görselin logo_color'a kaçması gerçek bir
    hataydı).
    """
    n = len(colors)
    if n < 3:
        return set()
    films: set[int] = set()
    arr = small_rgb.astype(np.float32)
    rgbs = [np.array(c["rgb"], dtype=np.float32) for c in colors]
    ratios = [float(c["ratio"]) for c in colors]
    kernel = np.ones((3, 3), np.uint8)

    # küçükten büyüğe: filmler önce elenir, film olmayanlar çapa (anchor) kalır
    for i in sorted(range(n), key=lambda idx: ratios[idx]):
        far_rgb: np.ndarray | None = None
        for j in range(n):
            if far_rgb is not None:
                break
            if j == i or j in films or ratios[j] < ratios[i] * anchor_factor:
                continue
            for k in range(n):
                if k <= j or k == i or k in films or ratios[k] < ratios[i] * anchor_factor:
                    continue
                dist, t = _point_segment_distance(rgbs[i], rgbs[j], rgbs[k])
                if dist <= colinear_tol and 0.04 <= t <= 0.96:
                    far_rgb = rgbs[k] if t < 0.5 else rgbs[j]
                    break
        if far_rgb is None:
            continue
        mask = (np.linalg.norm(arr - rgbs[i], axis=2) <= 30.0).astype(np.uint8)
        total = int(mask.sum())
        if total == 0:
            films.add(i)
            continue
        survival = float(cv2.erode(mask, kernel).sum()) / float(total)
        if survival > max_survival:
            continue
        ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
        n_ring = int(np.count_nonzero(ring))
        if n_ring > 0:
            far_mask = np.linalg.norm(arr - far_rgb, axis=2) <= 30.0
            far_frac = float(np.count_nonzero(ring & far_mask)) / float(n_ring)
            if far_frac < 0.08:
                continue
        films.add(i)
    return films


def estimate_color_count(image: Image.Image, max_colors: int = 48) -> dict[str, Any]:
    rgb = _rgba_to_rgb_on_white(image)
    small = resize_for_analysis(rgb, 500)

    quantized = small.quantize(
        colors=max_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    histogram = quantized.histogram()
    palette = quantized.getpalette() or []
    total_pixels = small.size[0] * small.size[1]

    raw_colors = []

    for index, count in enumerate(histogram):
        if count <= 0 or index * 3 + 2 >= len(palette):
            continue

        ratio = count / total_pixels

        if ratio < 0.002:
            continue

        raw_colors.append(
            {
                "rgb": [
                    int(palette[index * 3]),
                    int(palette[index * 3 + 1]),
                    int(palette[index * 3 + 2]),
                ],
                "ratio": round(float(ratio), 4),
            }
        )

    raw_colors = sorted(raw_colors, key=lambda item: item["ratio"], reverse=True)
    merged_colors = _merge_near_colors(raw_colors, distance_threshold=22)

    dominant_colors = [
        item for item in merged_colors
        if item["ratio"] >= 0.004
    ]

    # Anti-alias karışım filmleri (iki gerçek renk arasındaki ince kenar
    # tonları) sayımı şişirmesin: ince kenarlıklı sade logolar 'çok renkli'
    # görünüp yanlış moda gitmesin diye film-arındırılmış sayım da üretilir.
    # Ham sayım (photo eşikleri için) korunur.
    film_idx = _aa_film_indexes(np.asarray(small), dominant_colors)
    flat_dominant = [c for i, c in enumerate(dominant_colors) if i not in film_idx]

    return {
        "estimated_color_count": len(dominant_colors),
        "flat_color_count": len(flat_dominant),
        "dominant_colors": dominant_colors[:12],
        "flat_dominant_colors": flat_dominant[:12],
    }


def detect_background(image: Image.Image) -> dict[str, Any]:
    rgba = image.convert("RGBA")
    alpha = np.array(resize_for_analysis(rgba.getchannel("A"), 700))
    rgb = _rgba_to_rgb_on_white(rgba)
    arr = np.array(resize_for_analysis(rgb, 700))

    h, w, _ = arr.shape
    patch_w = max(8, int(w * 0.08))
    patch_h = max(8, int(h * 0.08))

    corners = [
        arr[0:patch_h, 0:patch_w],
        arr[0:patch_h, w - patch_w:w],
        arr[h - patch_h:h, 0:patch_w],
        arr[h - patch_h:h, w - patch_w:w],
    ]

    alpha_corners = [
        alpha[0:patch_h, 0:patch_w],
        alpha[0:patch_h, w - patch_w:w],
        alpha[h - patch_h:h, 0:patch_w],
        alpha[h - patch_h:h, w - patch_w:w],
    ]

    transparent_corner_ratio = float(
        np.mean(np.concatenate([c.reshape(-1) for c in alpha_corners], axis=0) < 245)
    )

    corner_pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_color = np.median(corner_pixels, axis=0)

    distances = np.linalg.norm(
        corner_pixels.astype(np.float32) - bg_color.astype(np.float32),
        axis=1,
    )

    uniform_ratio = float(np.mean(distances < 18))
    brightness = float(np.mean(bg_color))

    if transparent_corner_ratio > 0.45:
        bg_type = "transparent"
    elif brightness > 235:
        bg_type = "white"
    elif brightness < 25:
        bg_type = "black"
    else:
        bg_type = "colored"

    return {
        "background_rgb": [int(x) for x in bg_color],
        "background_type": bg_type,
        "background_uniformity": round(uniform_ratio, 4),
        "is_uniform_background": uniform_ratio > 0.82 or transparent_corner_ratio > 0.45,
        "transparent_corner_ratio": round(transparent_corner_ratio, 4),
    }


def calculate_blur_score(image: Image.Image) -> float:
    rgb = np.array(_rgba_to_rgb_on_white(image))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return round(float(score), 2)


def calculate_edge_density(image: Image.Image) -> float:
    rgb = np.array(resize_for_analysis(_rgba_to_rgb_on_white(image), 700))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 70, 160)
    density = np.count_nonzero(edges) / edges.size

    return round(float(density), 4)


def calculate_semantic_edge_stats(image: Image.Image) -> dict[str, float] | None:
    """Derin (HED) kenar haritasından anlamsal kenar istatistikleri.

    * ``semantic_edge_density`` — güçlü anlamsal kenar (HED >= 0.25) piksel oranı.
      Gerçek nesne/yazı sınırlarında yüksek; fotoğraf dokusu ve gürültüde düşük
      (Canny'nin tam tersi: o dokuda patlar).
    * ``edge_coherence`` — Canny kenarlarının anlamsal kenarlarla örtüşme oranı.
      Temiz tasarımda yüksek, doku/gürültü baskın görselde düşük.

    Korpus ölçümü (ayrım gücü): logolar semantik 0.23-0.30 / Canny <= 0.10;
    gürültülü fotoğraf semantik 0.04 / Canny 0.28. HED modeli yoksa ``None``
    döner ve çağıran taraf bu sinyalleri kullanmaz (davranış değişmez).
    """
    from app.dl_segmentation import compute_edge_map

    rgb = np.array(resize_for_analysis(_rgba_to_rgb_on_white(image), 700))
    edge = compute_edge_map(rgb)
    if edge is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 70, 160) > 0
    strong = edge >= 0.25
    semantic_density = float(strong.mean())
    strong_d = cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    coherence = float((canny & strong_d).sum()) / max(1, int(canny.sum()))
    return {
        "semantic_edge_density": round(semantic_density, 4),
        "edge_coherence": round(coherence, 4),
    }


def calculate_structure_likelihood(image: Image.Image) -> dict[str, float]:
    rgb = np.array(resize_for_analysis(_rgba_to_rgb_on_white(image), 700))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_pixels = max(int(np.count_nonzero(edges)), 1)

    h, w = gray.shape
    min_line_length = max(18, int(min(w, h) * 0.035))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(min(w, h) * 0.025)),
        minLineLength=min_line_length,
        maxLineGap=max(4, int(min(w, h) * 0.01)),
    )

    line_length = 0.0
    aligned_length = 0.0
    long_line_count = 0

    if lines is not None:
        for raw_line in lines[:240]:
            # OpenCV <5 satırı (1,4), OpenCV 5+ (4,) döndürür; ikisini de kaldır
            vals = np.asarray(raw_line).reshape(-1)
            x1, y1, x2, y2 = (int(v) for v in vals[:4])
            length = float(np.hypot(x2 - x1, y2 - y1))

            if length < min_line_length:
                continue

            long_line_count += 1
            line_length += length
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) % 180.0
            nearest = min((0.0, 45.0, 90.0, 135.0, 180.0), key=lambda target: abs(angle - target))

            if abs(angle - nearest) <= 8.0:
                aligned_length += length

    line_density = min(1.0, line_length / max(edge_pixels * 1.8, 1.0))
    alignment_ratio = aligned_length / line_length if line_length > 1e-6 else 0.0
    straight_edge_likelihood = min(1.0, line_density * 0.65 + alignment_ratio * 0.35)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=180,
        qualityLevel=0.025,
        minDistance=max(5, int(min(w, h) * 0.012)),
        blockSize=5,
        useHarrisDetector=False,
    )
    corner_count = 0 if corners is None else len(corners)
    corner_likelihood = min(1.0, (corner_count / 95.0) * 0.55 + min(long_line_count / 28.0, 1.0) * 0.45)

    return {
        "straight_edge_likelihood": round(float(straight_edge_likelihood), 4),
        "corner_likelihood": round(float(corner_likelihood), 4),
    }


def calculate_thin_ink_ratio(image: Image.Image) -> float:
    """Mürekkep piksellerinin ne kadarı İNCE konturlarda (yarı-genişlik <= 1.5px)?

    Kontur çizimlerini (lineart: hemen tüm mürekkep ince çizgi) dolgu
    silüetlerinden (single_color: kalın bloklar) ayırır. AA filmleri elendikçe
    renk sayısı tek başına bu ayrımı yapamaz hale geldi; incelik doğrudan ölçülür.
    """
    rgb = np.array(resize_for_analysis(_rgba_to_rgb_on_white(image), 500))
    h, w = rgb.shape[:2]
    pw, ph = max(8, w // 12), max(8, h // 12)
    corners = np.concatenate([
        rgb[:ph, :pw].reshape(-1, 3), rgb[:ph, -pw:].reshape(-1, 3),
        rgb[-ph:, :pw].reshape(-1, 3), rgb[-ph:, -pw:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
    ink = (dist > 40).astype(np.uint8)
    n_ink = int(ink.sum())
    if n_ink < 30:
        return 0.0
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    return round(float(((dt > 0) & (dt <= 1.5)).sum()) / float(n_ink), 4)


def detect_transparency(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False

    alpha = np.array(image.getchannel("A"))
    return bool(np.any(alpha < 250))


def detect_alpha_foreground(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False

    alpha = np.array(image.getchannel("A"))
    opaque_ratio = float(np.mean(alpha > 12))
    transparent_ratio = float(np.mean(alpha < 245))

    return bool(opaque_ratio > 0.005 and transparent_ratio > 0.005)


def detect_gradient_like_surface(image: Image.Image) -> bool:
    small = resize_for_analysis(_rgba_to_rgb_on_white(image), 350)
    arr = np.array(small).astype(np.float32)

    flattened = arr.reshape(-1, 3)
    rounded = (flattened // 8) * 8
    unique_count = len(np.unique(rounded, axis=0))
    unique_ratio = unique_count / flattened.shape[0]

    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge_mask = cv2.Canny(gray, 70, 160) > 0
    smooth_area_ratio = float(np.mean(~edge_mask))

    return bool(unique_ratio > 0.095 and smooth_area_ratio > 0.68)


def score_image_quality(
    width: int,
    height: int,
    blur_score: float,
    estimated_color_count: int,
    edge_density: float,
    has_gradient: bool,
    is_uniform_background: bool,
) -> int:
    score = 100
    min_side = min(width, height)

    if min_side < 512:
        score -= 25
    elif min_side < 900:
        score -= 10

    if blur_score < 40:
        score -= 30
    elif blur_score < 90:
        score -= 12

    if estimated_color_count > 28:
        score -= 18
    elif estimated_color_count > 16:
        score -= 8

    if edge_density > 0.18:
        score -= 12

    if has_gradient:
        score -= 6

    if not is_uniform_background:
        score -= 8

    return max(0, min(100, score))


def _has_black_white_red_signature(dominant_colors: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    dominant_rgbs = [item["rgb"] for item in dominant_colors]

    has_black = any(max(rgb) <= 45 for rgb in dominant_rgbs)
    has_white = any(min(rgb) >= 218 for rgb in dominant_rgbs)
    # |g-b| <= 45: gerçek kırmızıda yeşil≈mavi (ikisi de düşük); turuncuda yeşil
    # maviden belirgin yüksektir (ör. 245,90,34 -> g-b=56). Bu olmadan turuncu
    # 'kırmızı' sanılıp logo b/w/red geometric'e gidiyor ve kanonik kırmızıya snap.
    has_red = any(
        rgb[0] >= 165 and rgb[1] <= 105 and rgb[2] <= 105
        and rgb[0] > rgb[1] * 1.35 and abs(rgb[1] - rgb[2]) <= 45
        for rgb in dominant_rgbs
    )

    return has_black, has_white, has_red


def _dominant_saturation_stats(dominant_colors: list[dict[str, Any]]) -> dict[str, float]:
    if not dominant_colors:
        return {"saturated_ratio": 0.0, "neutral_ratio": 1.0}

    saturated_ratio = 0.0
    neutral_ratio = 0.0

    for item in dominant_colors:
        rgb = item["rgb"]
        ratio = float(item["ratio"])
        max_c = max(rgb)
        min_c = min(rgb)
        saturation_like = max_c - min_c

        if saturation_like >= 55:
            saturated_ratio += ratio
        if saturation_like <= 30:
            neutral_ratio += ratio

    return {
        "saturated_ratio": round(float(saturated_ratio), 4),
        "neutral_ratio": round(float(neutral_ratio), 4),
    }


def _dominant_color_family_stats(dominant_colors: list[dict[str, Any]]) -> dict[str, float]:
    red_like_ratio = 0.0
    non_red_saturated_ratio = 0.0

    for item in dominant_colors:
        rgb = item["rgb"]
        ratio = float(item["ratio"])
        max_c = max(rgb)
        min_c = min(rgb)
        saturation_like = max_c - min_c
        is_red_like = (
            rgb[0] >= 165
            and rgb[1] <= 115
            and rgb[2] <= 115
            and rgb[0] > rgb[1] * 1.28
            and rgb[0] > rgb[2] * 1.28
            and abs(rgb[1] - rgb[2]) <= 45  # turuncu (g≫b) kırmızı sayılmaz
        )

        if is_red_like:
            red_like_ratio += ratio
        elif saturation_like >= 55:
            non_red_saturated_ratio += ratio

    return {
        "red_like_ratio": round(float(red_like_ratio), 4),
        "non_red_saturated_ratio": round(float(non_red_saturated_ratio), 4),
    }


def _distinct_vivid_hue_count(dominant_colors: list[dict[str, Any]]) -> int:
    """Baskın renkler içinde KAÇ FARKLI canlı ton olduğunu sayar.

    Doygun pikselin ALAN oranına bakan testler, küçük ama canlı çok-renkli bir
    logoyu büyük beyaz zemin yüzünden 'minimal' sanabiliyor (renkler ~%1). Ton
    SAYISI alandan bağımsızdır: 5 farklı canlı ton = renk logosu, alanı küçük olsa
    bile. Tonlar 30°'lik kovalara ayrılır; yakın tonlar tek sayılır.
    """
    buckets: set[int] = set()
    for item in dominant_colors:
        r, g, b = item["rgb"]
        mx, mn = max(r, g, b), min(r, g, b)
        if mx - mn < 55 or mx < 55:
            continue  # canlı değil (gri/siyah/beyaz)
        hue = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[0] * 360.0
        buckets.add(int(hue) // 30)
    return len(buckets)


def _foreground_stats(image: Image.Image) -> dict[str, Any]:
    """ÖN PLAN renk istatistikleri: kromatik renk sayısı + canlı renk oranı.

    ``chromatic_color_count``: ön plandaki farklı KROMATİK (doygun, gri-olmayan)
    renk sayısı (16'lık kova). ``estimate_color_count`` tüm görseli sayar; büyük
    beyaz/şeffaf zeminli bir gradyan logoda her ton ratio eşiğinin altında kalıp
    ELENİR ve görsel '1 renk' görünür. Ön plana bakınca gradyan çok sayıda ton
    gösterir; bu, gradyanı zemine bağımlı olmadan yakalar.

    ÖNEMLİ: yalnızca KROMATİK (max-min >= 35) pikseller sayılır. Gri tonlama
    sayılırsa, anti-alias'lı bir B/W çizim çok 'renk' gösterip yanlışlıkla renk
    logosu sanılır (gerçek bir regresyon kaynağıydı). B/W çizim -> 0 döner.

    ``vivid_ratio``: ön plan piksellerinin ne kadarı GÜÇLÜ kromatik (sat >= 60).
    İnce kırmızı/mavi çizgiler gibi alanı küçük ama canlı renkler, baskın-renk
    listesine giremeden anti-alias'ta kaybolabiliyor; ikili (siyah-beyaz) modlara
    yönlendirmeden önce bu oran kontrol edilir ki RENK YOK EDİLMESİN. Vintage
    kağıt tonu gibi hafif sıcaklıklar (sat < 60) oranı tetiklemez.
    """
    rgba = image.convert("RGBA")
    small = resize_for_analysis(rgba, 400)
    arr = np.asarray(small)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return {"chromatic_color_count": 0, "vivid_ratio": 0.0}
    alpha = arr[..., 3]
    rgb = arr[..., :3].astype(np.float32)
    a = (alpha / 255.0)[..., None]
    comp = (rgb * a + 255.0 * (1.0 - a)).astype(np.int32)
    dist = np.linalg.norm(comp.astype(np.float32) - 255.0, axis=2)
    saturation = comp.max(axis=2) - comp.min(axis=2)
    fg_all = (dist > 40) & (alpha > 40)
    fg_chroma = fg_all & (saturation >= 35)
    vivid = fg_all & (saturation >= 60)
    vivid_ratio = float(np.count_nonzero(vivid)) / max(1, int(np.count_nonzero(fg_all)))
    if int(np.count_nonzero(fg_chroma)) < 20:
        return {"chromatic_color_count": 0, "vivid_ratio": round(vivid_ratio, 4)}
    quantized = comp[fg_chroma] // 16
    return {
        "chromatic_color_count": int(len(np.unique(quantized, axis=0))),
        "vivid_ratio": round(vivid_ratio, 4),
    }


def classify_logo_geometry(
    estimated_color_count: int,
    has_gradient: bool,
    edge_density: float,
    blur_score: float,
    has_black: bool,
    has_white: bool,
    has_red: bool,
    saturated_ratio: float = 0.0,
) -> dict[str, bool]:
    hard_bwr_geometric = (
        estimated_color_count <= 8
        and not has_gradient
        and edge_density <= 0.105
        and has_black
        and has_white
        and has_red
    )

    is_flat_logo = (
        estimated_color_count <= 10
        and not has_gradient
        and edge_density <= 0.13
        and blur_score >= 45
        and (has_black or has_white)
    )

    likely_geometric_logo = hard_bwr_geometric or (
        is_flat_logo
        and estimated_color_count <= 6
        and edge_density <= 0.085
        and saturated_ratio <= 0.35
        and has_black
        and has_white
    )

    return {
        "is_flat_logo": is_flat_logo,
        "likely_geometric_logo": likely_geometric_logo,
    }


def analyze_image_from_mem(image: Image.Image) -> dict[str, Any]:
    """Tek bir PIL görselini analiz eder ve tam sınıflandırma raporu döndürür.

    Bu fonksiyon hem dosya tabanlı ``analyze_image`` hem de API'nin in-memory
    akışı için tek doğruluk kaynağıdır.
    """
    image = image.convert("RGBA")

    width, height = image.size
    has_transparency = detect_transparency(image)
    has_alpha_foreground = detect_alpha_foreground(image)

    color_data = estimate_color_count(image)
    bg_data = detect_background(image)

    blur_score = calculate_blur_score(image)
    edge_density = calculate_edge_density(image)
    structure_data = calculate_structure_likelihood(image)
    straight_edge_likelihood = structure_data["straight_edge_likelihood"]
    corner_likelihood = structure_data["corner_likelihood"]
    has_gradient = detect_gradient_like_surface(image)

    estimated_color_count = color_data["estimated_color_count"]
    # Film-arındırılmış sayım: anti-alias kenar tonları düşülmüş GERÇEK düz renk
    # sayısı. Az-renk eşikli kapılar (geometric/minimal/lineart/bwr) bunu kullanır;
    # ince kenarlıklı sade logolar AA tonları yüzünden 'çok renkli' sanılmaz.
    # Foto eşikleri (>28/>34) ham sayımda kalır (fotoğrafta tonlar gerçektir).
    flat_color_count = int(color_data.get("flat_color_count", estimated_color_count))
    flat_dominant_colors = color_data.get("flat_dominant_colors", color_data["dominant_colors"])
    saturation_stats = _dominant_saturation_stats(flat_dominant_colors)
    color_family_stats = _dominant_color_family_stats(flat_dominant_colors)
    # Küçük ama canlı çok-renkli logolar (büyük beyaz zemin) doygunluk-oranı
    # testlerini geçemiyor; ton SAYISI alandan bağımsız ayırt eder.
    vivid_hue_count = _distinct_vivid_hue_count(flat_dominant_colors)
    vivid_multicolor = vivid_hue_count >= 3
    # Gradyan/tonal-zengin ön plan: büyük zemin yüzünden estimate_color_count'un
    # kaçırdığı tek-ton gradyan logoları (ör. Vektoryum mavi V) yakalar.
    fg_stats = _foreground_stats(image)
    foreground_color_count = fg_stats["chromatic_color_count"]
    vivid_foreground_ratio = fg_stats["vivid_ratio"]
    rich_foreground = foreground_color_count >= 90       # gradyan-zengin renkli logo
    # 28: vintage/distressed siyah-beyaz badge'lerin hafif sıcak tonu (~19-23
    # kromatik kova) 'renk' sayılmasın; gerçek renk aksanları (teal ~33, kahve ~38)
    # üstte kalsın. Düşük eşik monokrom badge'leri lineart'tan dışlayıp minimal_ai'de
    # soldururdu (bilateral ince çizgiyi griye buluyor).
    has_color_foreground = foreground_color_count >= 28   # herhangi bir gerçek renk
    # Alanı küçük ama CANLI renkler (ince kırmızı/mavi çizgi, küçük renkli aksan):
    # ikili modlar bunları siyaha çevirip RENGİ YOK EDER. Ön planın >= %2'si güçlü
    # kromatikse görsel ikili (lineart/single_color) modlara yönlendirilmez.
    chromatic_accents = vivid_foreground_ratio >= 0.02

    quality_score = score_image_quality(
        width=width,
        height=height,
        blur_score=blur_score,
        estimated_color_count=estimated_color_count,
        edge_density=edge_density,
        has_gradient=has_gradient,
        is_uniform_background=bg_data["is_uniform_background"],
    )

    warnings = []

    if min(width, height) < 512:
        warnings.append("Low image resolution. A larger PNG usually gives cleaner vectors.")

    if blur_score < 40:
        warnings.append("Image looks blurry. Vector edges may need manual review.")

    if estimated_color_count > 28:
        warnings.append("Many colors or tones detected. Color logo mode is recommended.")

    if has_gradient:
        warnings.append("Gradient or shadow-like transitions detected. Color logo mode is safer.")

    if not bg_data["is_uniform_background"]:
        warnings.append("Background is not fully uniform. Background cleanup may be needed.")

    has_black, has_white, has_red = _has_black_white_red_signature(flat_dominant_colors)

    bwr_low_color_signature = (
        flat_color_count <= 8
        and edge_density <= 0.105
        and has_black
        and has_white
        and has_red
        and color_family_stats["non_red_saturated_ratio"] <= 0.035
    )

    if bwr_low_color_signature:
        has_gradient = False

    geometry_flags = classify_logo_geometry(
        estimated_color_count=flat_color_count,
        has_gradient=has_gradient,
        edge_density=edge_density,
        blur_score=blur_score,
        has_black=has_black,
        has_white=has_white,
        has_red=has_red,
        saturated_ratio=saturation_stats["saturated_ratio"],
    )

    if (
        not geometry_flags["likely_geometric_logo"]
        and flat_color_count <= 8
        and not has_gradient
        and edge_density <= 0.13
        and has_black
        and has_white
        and (
            (has_red and color_family_stats["non_red_saturated_ratio"] <= 0.035)
            or (
                saturation_stats["saturated_ratio"] <= 0.35
                and color_family_stats["non_red_saturated_ratio"] <= 0.045
            )
        )
        and straight_edge_likelihood >= 0.34
        and corner_likelihood >= 0.28
    ):
        geometry_flags["is_flat_logo"] = True
        geometry_flags["likely_geometric_logo"] = True

    mostly_neutral_art = (
        saturation_stats["neutral_ratio"] >= 0.82
        and saturation_stats["saturated_ratio"] <= 0.10
    )

    likely_line_art = (
        not has_gradient
        and has_black
        and has_white
        and not has_red
        and not has_alpha_foreground
        and edge_density >= 0.018
        and (
            flat_color_count <= 4
            or (
                flat_color_count <= 16
                and mostly_neutral_art
            )
        )
    )

    # incelik ayrımı: mürekkebin çoğu ince konturdaysa bu bir ÇİZİMDİR (lineart),
    # dolgu silüeti (single_color) değil. AA filmleri sayımdan düştüğünden renk
    # sayısı tek başına ikisini ayıramaz; incelik doğrudan ölçülür.
    thin_ink_ratio = calculate_thin_ink_ratio(image)

    likely_single_color = (
        flat_color_count <= 3
        and not has_gradient
        and has_black
        and has_white
        and not has_red
        and edge_density < 0.08
        and thin_ink_ratio <= 0.55
    )

    likely_text_logo = (
        geometry_flags["is_flat_logo"]
        and has_black
        and has_white
        and flat_color_count <= 12
        and edge_density <= 0.14
    )

    likely_color_logo = (
        flat_color_count > 8
        or has_gradient
        or vivid_multicolor
        or rich_foreground
        or saturation_stats["saturated_ratio"] > 0.38
        or color_family_stats["non_red_saturated_ratio"] > 0.045
        or (
            flat_color_count >= 7
            and saturation_stats["saturated_ratio"] > 0.22
            and not bwr_low_color_signature
        )
    )

    color_rich_logo = (
        likely_color_logo
        and not bwr_low_color_signature
        and (
            vivid_multicolor
            or rich_foreground
            or saturation_stats["saturated_ratio"] >= 0.18
            or flat_color_count >= 10
        )
    )

    likely_photo_or_complex = (
        estimated_color_count > 28
        or (has_gradient and edge_density > 0.11)
        or (quality_score < 45 and estimated_color_count > 16)
    )

    likely_natural_photo = (
        estimated_color_count > 34
        and not has_alpha_foreground
        and not geometry_flags["is_flat_logo"]
        and (
            not bg_data["is_uniform_background"]
            or (
                has_gradient
                and edge_density > 0.11
                and straight_edge_likelihood < 0.28
            )
        )
    )

    # DERİN KENAR SİNYALİ (opsiyonel HED): "Canny/anlamsal-kenar oranı yüksek"
    # = fotoğraf/doku imzası. Canny piksel gürültüsü ve dokuda patlar; HED yalnız
    # gerçek nesne/yazı sınırlarında yanıt verir. Oran zemine/kadraja bağımsızdır
    # ve renk sayısı/zemin-düzgünlüğü kriterlerini geçen (ör. düz beyaz zeminli
    # stüdyo fotoğrafı) fotoğrafları da yakalar. Ölçülen marj geniş: logolarda
    # oran <= 0.35, fotoğraflarda >= 2.0 (eşik 1.2 ~ 3.4x güvenlik payı).
    # Model yoksa sinyal None -> kapalı, davranış değişmez.
    semantic_stats = calculate_semantic_edge_stats(image)
    semantic_photo_like = False
    if semantic_stats is not None:
        sem_density = float(semantic_stats["semantic_edge_density"])
        noise_edge_ratio = edge_density / max(sem_density, 0.005)
        semantic_photo_like = bool(
            noise_edge_ratio >= 1.2
            and edge_density >= 0.05
            and sem_density < 0.12
            and not has_alpha_foreground
            and not geometry_flags["is_flat_logo"]
        )
    if semantic_photo_like and not likely_natural_photo:
        likely_natural_photo = True

    # single_color/lineart İKİLİ (siyah-beyaz) modlardır; gerçek renk taşıyan
    # logoda rengi tamamen yok ederler. Ön planda KROMATİK renk varsa (teal ikon,
    # kahve badge) ya da canlı renk aksanları varsa (ince kırmızı/mavi çizgi ->
    # chromatic_accents) bu modlara GİTMEZ; renk korunur. Gerçek B/W çizim
    # (kromatik renk yok -> her iki bayrak False) lineart'ta KALIR.
    if likely_single_color and not has_color_foreground and not chromatic_accents:
        detected_type = "single_color"
        recommended_mode = "single_color"

    elif likely_line_art and not has_color_foreground and not chromatic_accents:
        detected_type = "lineart"
        recommended_mode = "lineart"

    elif likely_natural_photo:
        detected_type = "photo_poster"
        recommended_mode = "photo_poster"
        warnings.append("Photo-like image detected. Vector output is a posterized approximation and should be reviewed.")

    elif color_rich_logo:
        detected_type = "logo_color"
        recommended_mode = "logo_color"

    elif geometry_flags["likely_geometric_logo"]:
        detected_type = "geometric_logo"
        recommended_mode = "geometric_logo"

    elif likely_text_logo and not likely_color_logo:
        detected_type = "minimal_ai"
        recommended_mode = "minimal_ai"

    elif (
        flat_color_count <= 12
        and not has_gradient
        and edge_density < 0.14
        and quality_score >= 55
        and not likely_color_logo
        # minimal_ai b/w/red palet sertleştirmesi uygular; gerçek siyah VEYA beyaz
        # imzası yoksa (ör. navy zemin + coral + ince beyaz yazı) bu yıkıcıdır
        # (coral -> pure red, beyaz yazı -> kırmızı). Renk-koruyan logo_color'a bırak.
        and (has_black or has_white)
    ):
        detected_type = "minimal_ai"
        recommended_mode = "minimal_ai"

    elif likely_photo_or_complex:
        detected_type = "logo_color"
        recommended_mode = "logo_color"
        warnings.append("Complex color image detected. Color logo mode should be reviewed before production.")

    else:
        detected_type = "logo_color"
        recommended_mode = "logo_color"

    if detected_type in ["minimal_ai", "geometric_logo", "single_color", "lineart"]:
        warnings = [
            warning for warning in warnings
            if "Background is not fully uniform" not in warning
        ]

    if detected_type == "geometric_logo":
        has_gradient = False
        likely_color_logo = False

    return {
        "width": width,
        "height": height,
        "has_transparency": has_transparency,
        "estimated_color_count": estimated_color_count,
        "flat_color_count": flat_color_count,
        "vivid_foreground_ratio": vivid_foreground_ratio,
        "thin_ink_ratio": thin_ink_ratio,
        "dominant_colors": color_data["dominant_colors"],
        "background": bg_data,
        "blur_score": blur_score,
        "edge_density": edge_density,
        "has_gradient": has_gradient,
        "quality_score": quality_score,
        "detected_type": detected_type,
        "recommended_mode": recommended_mode,
        "warnings": warnings,
        "is_flat_logo": geometry_flags["is_flat_logo"],
        "likely_geometric_logo": geometry_flags["likely_geometric_logo"],
        "likely_text_logo": likely_text_logo,
        "likely_color_logo": likely_color_logo,
        "likely_line_art": likely_line_art,
        "likely_single_color": likely_single_color,
        "likely_photo_or_complex": likely_photo_or_complex,
        "has_black": has_black,
        "has_white": has_white,
        "has_red": has_red,
        "has_alpha_foreground": has_alpha_foreground,
        "straight_edge_likelihood": straight_edge_likelihood,
        "corner_likelihood": corner_likelihood,
        "semantic_edge_density": (semantic_stats or {}).get("semantic_edge_density"),
        "edge_coherence": (semantic_stats or {}).get("edge_coherence"),
        "semantic_photo_like": semantic_photo_like,
    }


def analyze_image(image_path: Path | str) -> dict[str, Any]:
    """Dosya tabanlı analiz. ``analyze_image_from_mem`` etrafında ince bir sarmalayıcıdır."""
    path = Path(image_path)
    with Image.open(path) as image:
        return analyze_image_from_mem(image)
