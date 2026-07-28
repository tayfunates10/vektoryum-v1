"""Palette consolidation guard for intentional broad neutral fills.

The established geometry implementation is retained byte-for-byte in
``app.geometry_cleanup_base``. This compatibility layer changes only canonical
black/white snapping: when an SVG contains a separate substantial low-chroma
neutral region near one of those canonical endpoints, that endpoint is removed
from the canonical snap targets for the call. Exact black/white remain unchanged,
nearby AA colors still merge into their traced clusters, and red canonical
snapping is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app import geometry_cleanup_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy_consolidate_svg_palette = _base.consolidate_svg_palette
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


def _neutral_fill_evidence(svg_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return substantial neutral fills distinct from canonical black/white."""
    try:
        root = _base.ET.parse(str(svg_path)).getroot()
    except Exception:
        return {"dark": [], "light": []}

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
        return {"dark": [], "light": []}

    evidence: dict[str, list[dict[str, Any]]] = {"dark": [], "light": []}
    white_weight = float(weights.get(_WHITE, 0.0))
    for rgb, weight in weights.items():
        if rgb in {_BLACK, _WHITE}:
            continue
        spread = max(rgb) - min(rgb)
        mean = sum(rgb) / 3.0
        weight_ratio = float(weight) / total_weight
        if spread > 12 or weight_ratio < 0.02:
            continue

        black_distance = _base.math.sqrt(float(_base._dist2(rgb, _BLACK)))
        white_distance = _base.math.sqrt(float(_base._dist2(rgb, _WHITE)))
        entry = {
            "rgb": [int(value) for value in rgb],
            "weight_ratio": round(weight_ratio, 6),
            "black_distance": round(black_distance, 4),
            "white_distance": round(white_distance, 4),
        }
        if 8.0 <= mean <= 96.0 and 12.0 <= black_distance <= 90.0:
            evidence["dark"].append(entry)
        if (
            white_weight / total_weight >= 0.05
            and 160.0 <= mean <= 247.0
            and 12.0 <= white_distance <= 80.0
        ):
            evidence["light"].append(entry)
    return evidence


def _light_neutral_fill_evidence(svg_path: Path) -> list[dict[str, Any]]:
    """Compatibility alias for the first targeted light-neutral guard."""
    return _neutral_fill_evidence(svg_path)["light"]


def consolidate_svg_palette(
    svg_path: Path,
    max_colors: int,
    merge_tol: float = 12.0,
    canonical: list[tuple[int, int, int]] | None = None,
    snap_tol: float = 42.0,
) -> dict[str, Any]:
    evidence = _neutral_fill_evidence(Path(svg_path))
    effective_canonical = canonical
    removed: list[tuple[int, int, int]] = []
    if canonical:
        removed = [
            endpoint
            for endpoint, tone_class in ((_BLACK, "dark"), (_WHITE, "light"))
            if endpoint in canonical and evidence[tone_class]
        ]
        if removed:
            effective_canonical = [color for color in canonical if color not in removed]

    result = _legacy_consolidate_svg_palette(
        svg_path,
        max_colors=max_colors,
        merge_tol=merge_tol,
        canonical=effective_canonical,
        snap_tol=snap_tol,
    )
    if evidence["dark"] or evidence["light"]:
        result["neutral_snap_guard"] = {
            "schema": "broad-intentional-neutral-canonical-snap-v2",
            "black_snap_removed": _BLACK in removed,
            "white_snap_removed": _WHITE in removed,
            "evidence": evidence,
        }
        # Preserve the original report key for consumers of the v1 light guard.
        if evidence["light"]:
            result["light_neutral_snap_guard"] = {
                "schema": "broad-light-neutral-canonical-snap-v1",
                "white_snap_removed": _WHITE in removed,
                "evidence": evidence["light"],
            }
    return result
