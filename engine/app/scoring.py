"""Aday kalite skorlama katmanı.

Skorlama iki kaynaktan beslenir:

1. **Yapısal metrikler** (her zaman çalışır): path sayısı, düğüm sayısı, renk
   sayısı, gömülü bitmap kontrolü, geometri skorları (düz çizgi / eksen / köşe).
2. **Raster benzerlik** (opsiyonel): CairoSVG ile render edilebilirse orijinalle
   karşılaştırılır. Cairo DLL yoksa bu adım atlanır ve yapısal metrikler kullanılır.

Profil bazlı ağırlıklarla ``total_score`` üretilir.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app.component_quality import gate_candidate_scores, score_svg_component_integrity
from app.fidelity import score_svg_fidelity
from app.geometry_cleanup import compute_geometry_report_for_svg, extract_points_from_path_data
from app.neutral_palette import restore_layered_neutral_svg_palette

logger = logging.getLogger(__name__)

_FILL_RE = re.compile(r'fill\s*[:=]\s*["\']?(#[0-9a-fA-F]{3,6})')

_GEOMETRIC_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color", "lineart", "centerline"}


# ---------------------------------------------------------------------------
# SVG yapısal analiz
# ---------------------------------------------------------------------------
def _parse_svg_stats(svg_path: Path) -> dict[str, Any]:
    stats = {"path_count": 0, "node_count": 0, "unique_colors": 0, "has_bitmap": False, "colors": []}
    try:
        root = ET.parse(str(svg_path)).getroot()
    except Exception:  # noqa: BLE001
        return stats

    colors: set[str] = set()
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "image":
            stats["has_bitmap"] = True
        elif tag == "path":
            d = el.get("d")
            if not d:
                continue
            stats["path_count"] += 1
            fill = el.get("fill")
            if fill and fill.startswith("#"):
                colors.add(fill.lower())
            try:
                for sp in extract_points_from_path_data(d):
                    stats["node_count"] += len(sp.get("points", []))
            except Exception:  # noqa: BLE001
                pass

    # style içindeki fill'ler
    try:
        raw = svg_path.read_text(encoding="utf-8", errors="ignore")
        if "<image" in raw or "data:image" in raw:
            stats["has_bitmap"] = True
        for m in _FILL_RE.finditer(raw):
            colors.add(m.group(1).lower())
    except OSError:
        pass

    stats["colors"] = sorted(colors)
    stats["unique_colors"] = len(colors)
    return stats


# ---------------------------------------------------------------------------
# Alt skorlar
# ---------------------------------------------------------------------------
def _path_efficiency_score(path_count: int, unique_colors: int, mode: str) -> float:
    """Path sayısının moda göre verimliliğini 0-100 arası puanlar.

    Sade modlarda az path makbuldür; çok renkli modlarda yeterli path beklenir.
    """
    if path_count <= 0:
        return 0.0

    if mode == "logo_color":
        lo, hi = 120, 2600
    elif mode == "photo_poster":
        lo, hi = 60, 5000
    else:  # geometric_logo, minimal_ai, single_color, lineart, centerline, flat_logo
        lo, hi = 3, 600

    if lo <= path_count <= hi:
        return 100.0
    if path_count < lo:
        deficit = (lo - path_count) / max(lo, 1)
        return round(max(45.0, 100.0 - deficit * 55.0), 2)
    over = (path_count - hi) / max(hi, 1)
    return round(max(0.0, 100.0 - over * 60.0), 2)


def _color_fidelity_score(unique_colors: int, analysis: dict[str, Any], mode: str) -> float:
    expected = max(1, int(analysis.get("estimated_color_count", 4)))
    if unique_colors <= 0:
        return 0.0

    if mode in ("geometric_logo", "minimal_ai", "flat_logo"):
        # az ve net renk beklenir
        if unique_colors <= max(6, expected + 1):
            return 100.0
        excess = unique_colors - (expected + 1)
        return round(max(40.0, 100.0 - excess * 8.0), 2)

    if mode in ("single_color", "lineart", "centerline"):
        if unique_colors <= 3:
            return 100.0
        return round(max(50.0, 100.0 - (unique_colors - 3) * 12.0), 2)

    if mode == "logo_color":
        # 12-64 renk ideal (adaptif cap renk/ton/foto-zengini logolarda ΔE'yi
        # düşürür); çok az = detay kaybı, çok fazla = gürültü
        if 12 <= unique_colors <= 64:
            return 100.0
        if unique_colors < 12:
            return round(max(50.0, 100.0 - (12 - unique_colors) * 5.0), 2)
        return round(max(45.0, 100.0 - (unique_colors - 64) * 4.0), 2)

    return 80.0


def _detail_score(node_count: int, path_count: int, analysis: dict[str, Any], mode: str) -> float:
    """Düğüm yoğunluğunu moda göre değerlendirir (denge: detay vs gürültü)."""
    if path_count <= 0 or node_count <= 0:
        return 0.0
    nodes_per_path = node_count / path_count

    if mode in _GEOMETRIC_MODES:
        # 4-40 düğüm/path makul; çok yüksek = dalgalı/gürültülü
        if nodes_per_path <= 40:
            return 100.0
        return round(max(40.0, 100.0 - (nodes_per_path - 40) * 1.5), 2)

    # color/photo: yeterli düğüm detay demektir
    if node_count >= 400:
        return 100.0
    return round(max(45.0, 50.0 + node_count / 8.0), 2)


# ---------------------------------------------------------------------------
# Ana skorlama
# ---------------------------------------------------------------------------
def _weights(mode: str) -> dict[str, float]:
    if mode == "geometric_logo":
        return {
            "color": 0.16, "edge": 0.16, "detail": 0.10, "path": 0.08, "warning": 0.08,
            "straight_edge": 0.17, "corner_cleanliness": 0.15, "axis_alignment": 0.10,
        }
    if mode in ("minimal_ai", "flat_logo"):
        return {
            "color": 0.25, "edge": 0.25, "detail": 0.10, "path": 0.15, "warning": 0.10,
            "geometry": 0.15,
        }
    if mode in ("single_color", "lineart", "centerline"):
        return {
            "color": 0.18, "edge": 0.22, "detail": 0.10, "path": 0.15, "warning": 0.10,
            "geometry": 0.25,
        }
    # logo_color, photo_poster ve diğerleri: geometri ağırlığı yok
    return {"color": 0.25, "edge": 0.30, "detail": 0.25, "path": 0.12, "warning": 0.08}


def score_vector_candidate(
    original_path: Path,
    svg_path: Path,
    analysis_report: dict[str, Any],
    mode: str,
    geometry_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bir vektör adayını puanlar ve detaylı skor sözlüğü döndürür."""
    # P0-B2a source-neutral correction happens before any candidate metric is
    # computed, so global fidelity and the P1-A component gate both judge the
    # exact artifact that can win.  The helper is a no-op outside layered-neutral
    # geometric_logo and never touches chromatic fills.
    neutral_palette_restore = restore_layered_neutral_svg_palette(
        Path(svg_path), Path(original_path), mode=mode
    )
    stats = _parse_svg_stats(Path(svg_path))

    # geometri skorları: cleanup raporu varsa onu kullan, yoksa SVG'den hesapla
    geo = (geometry_report or {}).get("report") if geometry_report else None
    if not geo:
        geo = compute_geometry_report_for_svg(Path(svg_path))

    straight_edge_score = float(geo.get("straight_edge_score", 0.0)) * 100.0
    corner_cleanliness_score = float(geo.get("corner_cleanliness_score", 0.0)) * 100.0
    axis_alignment_score = float(geo.get("axis_alignment_score", 0.0)) * 100.0
    geometry_score = float(geo.get("geometry_score", 0.0)) * 100.0

    color_score = _color_fidelity_score(stats["unique_colors"], analysis_report, mode)
    path_score = _path_efficiency_score(stats["path_count"], stats["unique_colors"], mode)
    detail_score = _detail_score(stats["node_count"], stats["path_count"], analysis_report, mode)

    # edge_score: render edilebilirse ALGISAL sadakat (SSIM + LAB ΔE + kenar-F1);
    # render mümkün değilse yapısal tahmine düşülür (CairoSVG yoksa çökme yok).
    rendered_ok = False
    fidelity = score_svg_fidelity(Path(svg_path), Path(original_path))
    if fidelity is not None:
        rendered_ok = True
        edge_score = float(fidelity["fidelity_score"])
    else:
        fidelity = {}
        # yapısal tahmin: geometri + detay dengesinden türet
        if mode in _GEOMETRIC_MODES:
            edge_score = round(0.5 * straight_edge_score + 0.5 * detail_score, 2)
        else:
            edge_score = round(min(100.0, 0.6 * detail_score + 0.4 * color_score), 2)

    # warning_score: yapısal cezalar
    warning_score = 100.0
    warnings: list[str] = []
    if stats["has_bitmap"]:
        warning_score -= 60.0
        warnings.append("embedded_bitmap")
    if stats["path_count"] == 0:
        warning_score -= 100.0
        warnings.append("empty_svg")
    if stats["path_count"] > 3000:
        warning_score -= 20.0
        warnings.append("too_many_paths")
    if mode in ("geometric_logo", "minimal_ai", "flat_logo") and stats["unique_colors"] > 8:
        warning_score -= 15.0
        warnings.append("too_many_colors_for_flat")
    warning_score = max(0.0, warning_score)

    w = _weights(mode)
    total = (
        color_score * w.get("color", 0.0)
        + edge_score * w.get("edge", 0.0)
        + detail_score * w.get("detail", 0.0)
        + path_score * w.get("path", 0.0)
        + warning_score * w.get("warning", 0.0)
        + straight_edge_score * w.get("straight_edge", 0.0)
        + corner_cleanliness_score * w.get("corner_cleanliness", 0.0)
        + axis_alignment_score * w.get("axis_alignment", 0.0)
        + geometry_score * w.get("geometry", 0.0)
    )

    raw_total_score = round(total, 2)
    raw_fidelity_score = (
        float(fidelity.get("fidelity_score", edge_score)) if rendered_ok else None
    )

    # AI-2 P1-A: connected-component integrity is an independent winner gate.
    # Existing weighted scores are calculated first and retained verbatim.  A
    # palette-like candidate that loses/splits/invents components is then made
    # ineligible to beat a measured-pass candidate, regardless of global SSIM.
    # Missing applicable measurement is fail-closed as needs_review.  Photo/
    # gradient inputs are explicitly not-applicable and preserve legacy ranking.
    component_quality = score_svg_component_integrity(
        Path(svg_path), Path(original_path), mode=mode, analysis=analysis_report
    )
    gated = gate_candidate_scores(raw_total_score, raw_fidelity_score, component_quality)
    if gated["selection_disqualified"]:
        warnings.append(f"component_quality_{gated['component_quality_status']}")

    return {
        "total_score": round(float(gated["total_score"]), 2),
        "raw_total_score": raw_total_score,
        "color_score": round(color_score, 2),
        "edge_score": round(edge_score, 2),
        "detail_score": round(detail_score, 2),
        "path_score": round(path_score, 2),
        "warning_score": round(warning_score, 2),
        "straight_edge_score": round(straight_edge_score, 2),
        "corner_cleanliness_score": round(corner_cleanliness_score, 2),
        "axis_alignment_score": round(axis_alignment_score, 2),
        "geometry_score": round(geometry_score, 2),
        "rendered_ok": rendered_ok,
        "fidelity_score": gated["fidelity_score"],
        "raw_fidelity_score": raw_fidelity_score,
        "fidelity": fidelity or None,
        "selection_disqualified": bool(gated["selection_disqualified"]),
        "component_quality_status": gated["component_quality_status"],
        "component_quality": component_quality,
        "neutral_palette_restore": neutral_palette_restore,
        "score_details": {
            "path_count": stats["path_count"],
            "node_count": stats["node_count"],
            "unique_colors": stats["unique_colors"],
            "has_bitmap": stats["has_bitmap"],
            "warnings": warnings,
            "ssim": fidelity.get("ssim"),
            "mean_delta_e": fidelity.get("mean_delta_e"),
            "edge_f1": fidelity.get("edge_f1"),
            "source_cc_recall": component_quality.get("source_cc_recall"),
            "render_cc_precision": component_quality.get("render_cc_precision"),
            "min_true_cc_iou": component_quality.get("min_true_cc_iou"),
            "component_quality_status": gated["component_quality_status"],
            "component_quality_applicable": component_quality.get("applicable"),
            "neutral_palette_restore_applied": neutral_palette_restore.get("applied"),
        },
    }
