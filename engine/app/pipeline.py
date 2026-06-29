"""Çekirdek vektörleştirme pipeline'ı (export öncesi).

Adımlar 1-7'yi (analiz → ön işleme → aday üretimi → geometri temizleme →
skorlama → seçim) tek bir yeniden kullanılabilir fonksiyonda toplar. Hem FastAPI
endpoint'i (``app.main``) hem de ölçüm CLI'si (``regression/fidelity_report.py``)
bu fonksiyonu çağırır; böylece iki yol arasında davranış sapması olmaz.

Export (SVG/PDF/EPS/DXF) ve HTTP'ye özgü her şey ``app.main``'de kalır.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.analyzer import analyze_image_from_mem
from app.geometry_cleanup import cleanup_svg_geometry, consolidate_svg_palette
from app.preprocess import preprocess_for_mode
from app.scoring import score_vector_candidate
from app.shape_fitting import regularize_svg_geometry
from app.vector_engines import build_vector_candidates, get_autotrace_path, run_candidate

logger = logging.getLogger(__name__)

# Mod bazlı son palet üst sınırı (VTracer'ın kenar ara-tonlarını temizlemek için)
PALETTE_CAP = {
    "geometric_logo": 5,
    "minimal_ai": 6,
    "flat_logo": 6,
    "single_color": 2,
    "lineart": 2,
    "centerline": 2,
    "logo_color": 22,
    "photo_poster": 18,
}

# Sade modlarda renkler saf siyah/beyaz/kırmızıya yaslanır (yakınsa)
FLAT_PALETTE_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color", "lineart", "centerline"}
CANONICAL_BWR = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]

# Geometrik şekil oturtma (line+arc fitting) uygulanan profiller
REGULARIZE_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color"}


# ---------------------------------------------------------------------------
# Mod uyarıları
# ---------------------------------------------------------------------------
def compute_mode_warning(trace_mode: str, mode_used: str, analysis: dict[str, Any]) -> str | None:
    if trace_mode == "auto":
        return None
    colors = int(analysis.get("estimated_color_count", 0))
    if mode_used == "minimal_ai" and colors > 12:
        return "Seçilen mod 'minimal_ai' ancak görselde çok renk var. 'logo_color' daha iyi olabilir."
    if mode_used == "logo_color" and analysis.get("likely_geometric_logo"):
        return "Seçilen mod 'logo_color' ancak görsel geometrik bir logoya benziyor. 'geometric_logo' daha iyi olabilir."
    if mode_used == "geometric_logo" and analysis.get("likely_color_logo") and colors > 12:
        return "Seçilen mod 'geometric_logo' ancak görsel çok renkli. 'logo_color' daha iyi olabilir."
    if mode_used == "centerline" and not get_autotrace_path():
        return "AutoTrace bulunamadı; centerline için skeleton fallback kullanılıyor."
    return None


# ---------------------------------------------------------------------------
# Aday seçim mantığı
# ---------------------------------------------------------------------------
def select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
    """En iyi adayı profil kurallarına göre seçer. (best, raw_best, reason)."""
    raw_best = max(scored, key=lambda c: c["total_score"])
    by_name = {c["name"]: c for c in scored}

    if mode != "geometric_logo":
        # color/diğer: en yüksek skor; eşitlikte 'standard' tercih
        near = [c for c in scored if raw_best["total_score"] - c["total_score"] <= 2.0]
        for suffix in ("logo_standard", "minimal_standard", "lineart_clean", "single_clean", "photo_standard"):
            if suffix in {c["name"] for c in near}:
                chosen = by_name[suffix]
                reason = "highest_total_score" if chosen is raw_best else "near_score_preference"
                return chosen, raw_best, reason
        return raw_best, raw_best, "highest_total_score"

    # geometric_logo seçimi
    def g(c: dict, key: str) -> float:
        return float(c.get(key, 0.0))

    def colors(c: dict) -> int:
        return int((c.get("score_details") or {}).get("unique_colors", 0))

    def paths(c: dict) -> int:
        return int((c.get("score_details") or {}).get("path_count", 0))

    # Dejenerasyon koruması: en güçlü adaya göre belirgin renk/path kaybı yaşayan
    # adayları (ör. kırmızıyı yitiren contour) ele — yoksa hepsini kullan.
    max_colors = max((colors(c) for c in scored), default=0)
    max_paths = max((paths(c) for c in scored), default=0)
    viable = [
        c for c in scored
        if colors(c) >= max_colors - 0 and paths(c) >= max(1, int(0.4 * max_paths))
    ] or scored

    raw_best = max(viable, key=lambda c: c["total_score"])
    near_names = {c["name"] for c in viable if raw_best["total_score"] - c["total_score"] <= 4.0}

    # spline adaylar (yuvarlak formlar pürüzsüz) önce; polygon/contour yalnız
    # belirgin üstünlükte (sert kenarlı logolar) seçilir
    preference = ["geo_standard", "geo_detail", "geo_mixed", "geo_clean", "geo_contour"]
    chosen = raw_best
    reason = "highest_total_score"
    for name in preference:
        if name in near_names:
            chosen = by_name[name]
            reason = "highest_total_score" if chosen is raw_best else "near_score_geometric_preference"
            break

    # geo_clean (polygon) yalnız köşe/eksende ÇOK belirgin üstünse (saf sert kenar)
    clean = by_name.get("geo_clean")
    if clean and clean["name"] in near_names and clean is not chosen:
        if (
            g(clean, "corner_cleanliness_score") >= g(chosen, "corner_cleanliness_score") + 8
            and g(clean, "axis_alignment_score") >= g(chosen, "axis_alignment_score") + 8
            and g(clean, "straight_edge_score") >= g(chosen, "straight_edge_score")
        ):
            chosen, reason = clean, "corner_cleanliness_preference"

    return chosen, raw_best, reason


# ---------------------------------------------------------------------------
# Çekirdek pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    image: Image.Image,
    original_path: Path,
    trace_mode: str,
    job_dir: Path,
) -> dict[str, Any]:
    """Analiz → ön işleme → aday → temizleme → skor → seçim akışını yürütür.

    Export yapılmaz. Dönen sözlük:
    ``analysis, mode_used, mode_warning, preprocess_report, results, scored,
    best, raw_best, selection_reason``. ``scored`` boşsa ``best`` ``None`` olur.
    """
    # 1. Analiz
    analysis = analyze_image_from_mem(image)
    mode_used = analysis["recommended_mode"] if trace_mode == "auto" else trace_mode
    mode_warning = compute_mode_warning(trace_mode, mode_used, analysis)

    # 2. Ön işleme
    preprocessed_path, preprocess_report = preprocess_for_mode(
        original_path, mode_used, job_dir, analysis=analysis
    )

    # 3. + 4. Aday üretimi + geometri temizleme
    candidates = build_vector_candidates(mode_used)
    results: list[dict[str, Any]] = []

    for name, spec in candidates.items():
        svg_path = job_dir / f"{name}.svg"
        try:
            run_candidate(spec["engine"], preprocessed_path, svg_path, spec)
            cleanup_report: dict[str, Any] = {}
            if spec.get("cleanup"):
                cleanup_report = cleanup_svg_geometry(svg_path, mode=mode_used, aggressiveness=spec["cleanup"])
            # palet konsolidasyonu: kenar ara-ton renklerini en baskın renklere indir
            cap = PALETTE_CAP.get(mode_used)
            if cap:
                canonical = CANONICAL_BWR if mode_used in FLAT_PALETTE_MODES else None
                consolidate_svg_palette(svg_path, max_colors=cap, canonical=canonical)
            # geometrik idealleştirme: düz çizgi + tam dairesel yay oturtma
            if mode_used in REGULARIZE_MODES:
                try:
                    regularize_svg_geometry(svg_path)
                except Exception as reg_err:  # noqa: BLE001
                    logger.debug("regularize atlandı (%s): %s", name, reg_err)
            results.append({
                "name": name,
                "svg_path": svg_path,
                "engine": spec["engine"],
                "cleanup_report": cleanup_report,
                "success": True,
                "error": None,
            })
        except FileNotFoundError as e:
            # opsiyonel CLI yok (potrace/autotrace)
            results.append({"name": name, "success": False, "error": str(e), "engine": spec["engine"]})
        except Exception as e:  # noqa: BLE001
            logger.warning("Aday '%s' üretilemedi: %s", name, e)
            results.append({"name": name, "success": False, "error": str(e), "engine": spec["engine"]})

    # 5. Skorlama
    scored: list[dict[str, Any]] = []
    for res in results:
        if not res.get("success"):
            continue
        score = score_vector_candidate(
            original_path=original_path,
            svg_path=res["svg_path"],
            analysis_report=analysis,
            mode=mode_used,
            geometry_report=res.get("cleanup_report", {}),
        )
        scored.append({**res, **score})

    best: dict[str, Any] | None = None
    raw_best: dict[str, Any] | None = None
    selection_reason = "no_candidate"
    if scored:
        best, raw_best, selection_reason = select_best(scored, mode_used)

    return {
        "analysis": analysis,
        "mode_used": mode_used,
        "mode_warning": mode_warning,
        "preprocess_report": preprocess_report,
        "results": results,
        "scored": scored,
        "best": best,
        "raw_best": raw_best,
        "selection_reason": selection_reason,
    }
