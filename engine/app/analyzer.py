from __future__ import annotations

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

    return {
        "estimated_color_count": len(dominant_colors),
        "dominant_colors": dominant_colors[:12],
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
            x1, y1, x2, y2 = [int(v) for v in raw_line[0]]
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
    has_red = any(
        rgb[0] >= 165 and rgb[1] <= 105 and rgb[2] <= 105 and rgb[0] > rgb[1] * 1.35
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
        )

        if is_red_like:
            red_like_ratio += ratio
        elif saturation_like >= 55:
            non_red_saturated_ratio += ratio

    return {
        "red_like_ratio": round(float(red_like_ratio), 4),
        "non_red_saturated_ratio": round(float(non_red_saturated_ratio), 4),
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
    saturation_stats = _dominant_saturation_stats(color_data["dominant_colors"])
    color_family_stats = _dominant_color_family_stats(color_data["dominant_colors"])

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

    has_black, has_white, has_red = _has_black_white_red_signature(color_data["dominant_colors"])

    bwr_low_color_signature = (
        estimated_color_count <= 8
        and edge_density <= 0.105
        and has_black
        and has_white
        and has_red
        and color_family_stats["non_red_saturated_ratio"] <= 0.035
    )

    if bwr_low_color_signature:
        has_gradient = False

    geometry_flags = classify_logo_geometry(
        estimated_color_count=estimated_color_count,
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
        and estimated_color_count <= 8
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
            estimated_color_count <= 4
            or (
                estimated_color_count <= 16
                and mostly_neutral_art
            )
        )
    )

    likely_single_color = (
        estimated_color_count <= 3
        and not has_gradient
        and has_black
        and has_white
        and not has_red
        and edge_density < 0.08
    )

    likely_text_logo = (
        geometry_flags["is_flat_logo"]
        and has_black
        and has_white
        and estimated_color_count <= 12
        and edge_density <= 0.14
    )

    likely_color_logo = (
        estimated_color_count > 8
        or has_gradient
        or saturation_stats["saturated_ratio"] > 0.38
        or color_family_stats["non_red_saturated_ratio"] > 0.045
        or (
            estimated_color_count >= 7
            and saturation_stats["saturated_ratio"] > 0.22
            and not bwr_low_color_signature
        )
    )

    color_rich_logo = (
        likely_color_logo
        and not bwr_low_color_signature
        and (
            saturation_stats["saturated_ratio"] >= 0.18
            or estimated_color_count >= 10
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

    if likely_single_color:
        detected_type = "single_color"
        recommended_mode = "single_color"

    elif likely_line_art:
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
        estimated_color_count <= 12
        and not has_gradient
        and edge_density < 0.14
        and quality_score >= 55
        and not likely_color_logo
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
    }


def analyze_image(image_path: Path | str) -> dict[str, Any]:
    """Dosya tabanlı analiz. ``analyze_image_from_mem`` etrafında ince bir sarmalayıcıdır."""
    path = Path(image_path)
    with Image.open(path) as image:
        return analyze_image_from_mem(image)
