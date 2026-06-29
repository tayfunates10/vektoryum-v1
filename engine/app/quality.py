"""Kalite raporu katmanı.

``basic_svg_quality_check`` sade logolarda düşük path sayısını hata saymaz;
çok renkli logolarda detay/renk dengesini denetler ve gömülü bitmap'i ciddi
sorun olarak işaretler.
"""

from __future__ import annotations

from typing import Any

_FLAT_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color", "lineart", "centerline"}


def basic_svg_quality_check(
    score_details: dict[str, Any],
    mode: str,
    geometry_report: dict[str, float] | None = None,
    total_score: float = 0.0,
) -> dict[str, Any]:
    """En iyi adayın yapısal istatistiklerine göre kalite raporu üretir."""
    path_count = int(score_details.get("path_count", 0))
    unique_colors = int(score_details.get("unique_colors", 0))
    has_bitmap = bool(score_details.get("has_bitmap", False))
    node_count = int(score_details.get("node_count", 0))

    warnings: list[str] = []

    if has_bitmap:
        warnings.append("SVG embeds a bitmap image; output is not fully vector.")

    if path_count == 0:
        warnings.append("No vector paths were produced.")

    flat_mode = mode in _FLAT_MODES
    # Sade logolarda az path uyarısı verme kuralı
    low_path_exempt = flat_mode and unique_colors <= 6 and path_count >= 10

    if mode == "logo_color":
        if 0 < path_count < 250:
            warnings.append("Low path count for a color logo; some detail may be lost.")
        if unique_colors > 24:
            warnings.append("High color count; consider reducing palette for production.")
    elif not low_path_exempt:
        if 0 < path_count < 4:
            warnings.append("Very low path count; the shape may be incomplete.")

    if path_count > 3000:
        warnings.append("Very high path count; the file may be heavy and hard to edit.")

    # genel renk uyarısı (sade modlar için)
    if flat_mode and unique_colors > 8:
        warnings.append("More colors than expected for a flat logo; palette cleanup recommended.")

    geo = geometry_report or {}
    geometry_block = {
        "straight_edge_score": round(float(geo.get("straight_edge_score", 0.0)) * 100, 2),
        "corner_cleanliness_score": round(float(geo.get("corner_cleanliness_score", 0.0)) * 100, 2),
        "axis_alignment_score": round(float(geo.get("axis_alignment_score", 0.0)) * 100, 2),
        "geometry_score": round(float(geo.get("geometry_score", 0.0)) * 100, 2),
    }

    # durum kararı
    serious = has_bitmap or path_count == 0
    if serious:
        status = "failed" if path_count == 0 else "needs_review"
    elif not warnings and total_score >= 80.0:
        status = "production_ready"
    elif total_score >= 70.0 and len(warnings) <= 1:
        status = "production_ready" if not warnings else "needs_review"
    else:
        status = "needs_review"

    return {
        "status": status,
        "warnings": warnings,
        "metrics": {
            "path_count": path_count,
            "node_count": node_count,
            "unique_color_count": unique_colors,
            "has_bitmap": has_bitmap,
        },
        "geometry_report": geometry_block,
    }
