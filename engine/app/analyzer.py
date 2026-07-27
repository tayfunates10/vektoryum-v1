"""Compatibility-preserving analyzer wrapper with neutral multi-tone routing guard.

The original analyzer implementation is retained byte-for-byte in
``app.analyzer_base``. This module re-exports its complete public and private
namespace, then overrides only the two analysis entry points. The guard prevents
real gray design regions from being collapsed by binary ``single_color`` or
``lineart`` modes after anti-alias films have already been removed by the base
analyzer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app import analyzer_base as _base

# Preserve compatibility for callers/tests importing private analyzer helpers.
for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def meaningful_neutral_midtones(image: Image.Image) -> list[dict[str, Any]]:
    """Return substantial neutral mid-tones that survived AA-film removal.

    A true design gray must be neutral, visibly separated from black/white and
    occupy at least one percent of the image. ``estimate_color_count`` has already
    removed thin colinear anti-alias films, so a surviving region is evidence for
    a real third tone rather than edge smoothing.
    """
    color_data = _base.estimate_color_count(image)
    tones: list[dict[str, Any]] = []
    for item in color_data.get("flat_dominant_colors") or []:
        rgb = [int(channel) for channel in item.get("rgb") or []]
        if len(rgb) != 3:
            continue
        ratio = float(item.get("ratio") or 0.0)
        spread = max(rgb) - min(rgb)
        luminance = sum(rgb) / 3.0
        if ratio >= 0.01 and spread <= 30 and 55.0 <= luminance <= 210.0:
            tones.append(
                {
                    "rgb": rgb,
                    "ratio": round(ratio, 4),
                    "luminance": round(luminance, 2),
                }
            )
    return tones


def analyze_image_from_mem(image: Image.Image) -> dict[str, Any]:
    report = dict(_base.analyze_image_from_mem(image))
    neutral_midtones = meaningful_neutral_midtones(image)
    report["meaningful_neutral_midtones"] = neutral_midtones
    report["has_meaningful_neutral_midtone"] = bool(neutral_midtones)

    # Binary modes destroy a real gray band by construction. Route this narrow,
    # evidence-backed class to the color-preserving multi-region pipeline. The
    # geometric profile snaps near-black to canonical black, which preserves shape
    # but still creates a large tone/SSIM error for deliberate dark-gray artwork.
    if neutral_midtones and report.get("recommended_mode") in {"single_color", "lineart"}:
        previous_mode = str(report.get("recommended_mode"))
        report["detected_type"] = "logo_color"
        report["recommended_mode"] = "logo_color"
        report["is_flat_logo"] = True
        report["likely_geometric_logo"] = False
        report["likely_color_logo"] = True
        report["likely_single_color"] = False
        report["likely_line_art"] = False
        warnings = list(report.get("warnings") or [])
        warnings.append(
            "Substantial neutral mid-tone regions detected; "
            f"auto mode changed from {previous_mode} to logo_color to preserve exact gray design layers."
        )
        report["warnings"] = warnings
        report["neutral_multitone_routing_guard"] = {
            "applied": True,
            "previous_mode": previous_mode,
            "selected_mode": "logo_color",
            "tone_count": len(neutral_midtones),
        }
    else:
        report["neutral_multitone_routing_guard"] = {
            "applied": False,
            "previous_mode": report.get("recommended_mode"),
            "selected_mode": report.get("recommended_mode"),
            "tone_count": len(neutral_midtones),
        }
    return report


def analyze_image(image_path: Path | str) -> dict[str, Any]:
    path = Path(image_path)
    with Image.open(path) as image:
        return analyze_image_from_mem(image)
