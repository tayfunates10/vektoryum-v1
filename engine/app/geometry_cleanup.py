"""Palette consolidation guard for intentional broad light-neutral fills.

The established geometry implementation is retained byte-for-byte in
``app.geometry_cleanup_base``. This compatibility layer changes only canonical
white snapping: when an SVG contains both a real white region and a separate,
substantial low-chroma light-neutral region, white is removed from the canonical
snap targets for that consolidation call. Exact white remains white, nearby AA
colors still merge into its cluster, and black/red canonical snapping is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app import geometry_cleanup_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy_consolidate_svg_palette = _base.consolidate_svg_palette
_WHITE = (255, 255, 255)


def _light_neutral_fill_evidence(svg_path: Path) -> list[dict[str, Any]]:
    """Return substantial light-neutral fills distinct from an exact white fill."""
    try:
        root = _base.ET.parse(str(svg_path)).getroot()
    except Exception:
        return []

    weights: dict[tuple[int, int, int], float] = {}
    for element in root.iter():
        if element.tag.split("}")[-1] != "path":
            continue
        fill = element.get("fill")
        rgb = _base._hex_to_rgb(fill) if fill else None
        if rgb is None:
            continue
        weight = _base._path_bbox_weight(element.get("d", ""))
        weights[rgb] = weights.get(rgb, 0.0) + float(weight)

    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0:
        return []
    white_weight = float(weights.get(_WHITE, 0.0))
    if white_weight / total_weight < 0.05:
        return []

    evidence: list[dict[str, Any]] = []
    for rgb, weight in weights.items():
        if rgb == _WHITE:
            continue
        spread = max(rgb) - min(rgb)
        mean = sum(rgb) / 3.0
        white_distance = _base.math.sqrt(float(_base._dist2(rgb, _WHITE)))
        weight_ratio = float(weight) / total_weight
        if (
            spread <= 12
            and 160.0 <= mean <= 247.0
            and 12.0 <= white_distance <= 80.0
            and weight_ratio >= 0.02
        ):
            evidence.append(
                {
                    "rgb": [int(value) for value in rgb],
                    "weight_ratio": round(weight_ratio, 6),
                    "white_distance": round(white_distance, 4),
                }
            )
    return evidence


def consolidate_svg_palette(
    svg_path: Path,
    max_colors: int,
    merge_tol: float = 12.0,
    canonical: list[tuple[int, int, int]] | None = None,
    snap_tol: float = 42.0,
) -> dict[str, Any]:
    evidence = _light_neutral_fill_evidence(Path(svg_path))
    effective_canonical = canonical
    if canonical and _WHITE in canonical and evidence:
        effective_canonical = [color for color in canonical if color != _WHITE]

    result = _legacy_consolidate_svg_palette(
        svg_path,
        max_colors=max_colors,
        merge_tol=merge_tol,
        canonical=effective_canonical,
        snap_tol=snap_tol,
    )
    if evidence:
        result["light_neutral_snap_guard"] = {
            "schema": "broad-light-neutral-canonical-snap-v1",
            "white_snap_removed": bool(canonical and _WHITE in canonical),
            "evidence": evidence,
        }
    return result
