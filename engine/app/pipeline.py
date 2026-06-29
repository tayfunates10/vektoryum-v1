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
def _fidelity_rank_key(c: dict[str, Any]) -> tuple[float, int, float]:
    """Fidelity-öncelikli sıralama anahtarı (hem seçim hem refinement kullanır).

    Birincil: algısal sadakat (0.1'e yuvarlanır, gürültü payı). Eşitlikte daha az
    path (daha düzenlenebilir), sonra yapısal total_score. Böylece neredeyse eşit
    sadakatte sade/düzenlenebilir çıktı kazanır.
    """
    paths = int((c.get("score_details") or {}).get("path_count", 0))
    return (round(float(c.get("fidelity_score") or 0.0), 1), -paths, float(c.get("total_score", 0.0)))


def select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
    """En iyi adayı profil kurallarına göre seçer. (best, raw_best, reason)."""
    raw_best = max(scored, key=lambda c: c["total_score"])
    by_name = {c["name"]: c for c in scored}

    # Renkli modlarda (logo_color/photo_poster): render edilebildiyse seçimi
    # GERÇEK algısal sadakate yasla. Yapısal total_score detay/karmaşıklığı
    # ödüllendirip daha karmaşık ama daha sadık-olmayan adayı seçebiliyor;
    # fidelity bunu düzeltir (eşitlikte daha az path = daha düzenlenebilir tercih).
    if mode in FIDELITY_LED_MODES:
        rendered = [
            c for c in scored
            if c.get("rendered_ok") and c.get("fidelity_score") is not None
            and not (c.get("score_details") or {}).get("has_bitmap")
        ]
        if rendered:
            chosen = max(rendered, key=_fidelity_rank_key)
            return chosen, raw_best, "highest_fidelity"

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


# Refinement uygulanan modlar: piksel sadakatinin asıl hedef olduğu renkli
# modlar. Geometrik/sade modlarda "temiz çizgi" önceliklidir, ham sadakat
# uğruna geometriyi bozmayız; o yüzden onlar refinement dışıdır.
FIDELITY_LED_MODES = {"logo_color", "photo_poster"}


# ---------------------------------------------------------------------------
# Tek aday üretimi + skorlama (ana döngü ve refinement ortak kullanır)
# ---------------------------------------------------------------------------
def produce_candidate(
    name: str,
    spec: dict[str, Any],
    preprocessed_path: Path,
    mode: str,
    job_dir: Path,
) -> dict[str, Any]:
    """Tek bir adayı üretir: trace → cleanup → palet konsolidasyonu → regularize."""
    svg_path = job_dir / f"{name}.svg"
    try:
        run_candidate(spec["engine"], preprocessed_path, svg_path, spec)
        cleanup_report: dict[str, Any] = {}
        if spec.get("cleanup"):
            cleanup_report = cleanup_svg_geometry(svg_path, mode=mode, aggressiveness=spec["cleanup"])
        # palet konsolidasyonu: kenar ara-ton renklerini en baskın renklere indir
        cap = PALETTE_CAP.get(mode)
        if cap:
            canonical = CANONICAL_BWR if mode in FLAT_PALETTE_MODES else None
            consolidate_svg_palette(svg_path, max_colors=cap, canonical=canonical)
        # geometrik idealleştirme: düz çizgi + tam dairesel yay oturtma
        if mode in REGULARIZE_MODES:
            try:
                regularize_svg_geometry(svg_path)
            except Exception as reg_err:  # noqa: BLE001
                logger.debug("regularize atlandı (%s): %s", name, reg_err)
        return {
            "name": name,
            "svg_path": svg_path,
            "engine": spec["engine"],
            "cleanup_report": cleanup_report,
            "success": True,
            "error": None,
        }
    except FileNotFoundError as e:
        # opsiyonel CLI yok (potrace/autotrace)
        return {"name": name, "success": False, "error": str(e), "engine": spec["engine"]}
    except Exception as e:  # noqa: BLE001
        logger.warning("Aday '%s' üretilemedi: %s", name, e)
        return {"name": name, "success": False, "error": str(e), "engine": spec["engine"]}


def score_candidate(
    res: dict[str, Any],
    original_path: Path,
    analysis: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    """Başarılı bir aday sonucunu skorlar; başarısızsa None döner."""
    if not res.get("success"):
        return None
    score = score_vector_candidate(
        original_path=original_path,
        svg_path=res["svg_path"],
        analysis_report=analysis,
        mode=mode,
        geometry_report=res.get("cleanup_report", {}),
    )
    return {**res, **score}


# ---------------------------------------------------------------------------
# Refinement: en iyi adayın hata-güdümlü iyileştirilmesi (kapalı döngü)
# ---------------------------------------------------------------------------
def _refine_variants(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """En iyi adayın spec'ine göre güdümlü VTracer parametre varyantları.

    Yön: daha çok detay/renk katmanı (color_precision↑, filter_speckle↓,
    layer_difference↓) — renkli logoda kayıp detay/bantlaşmayı geri kazandırır.
    """
    base = dict(spec.get("vtracer_params") or {})
    if not base:
        return []
    cp = int(base.get("color_precision", 6))
    fs = int(base.get("filter_speckle", 4))
    ld = base.get("layer_difference")

    v1 = dict(base)
    v1["color_precision"] = min(8, cp + 1)
    v1["filter_speckle"] = max(2, fs - 2)
    if ld is not None:
        v1["layer_difference"] = max(8, int(ld) - 8)

    v2 = dict(base)
    v2["color_precision"] = min(8, cp + 2)
    v2["filter_speckle"] = max(1, fs - 1)
    if ld is not None:
        v2["layer_difference"] = max(6, int(ld) - 12)

    return [("refine_detail", v1), ("refine_detail2", v2)]


def refine_best(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    preprocessed_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """En iyi adayın komşuluğunda güdümlü arama yapıp daha sadık varyant arar.

    İki kaldıraç: (1) VTracer parametre varyantları (mevcut ön işlenmiş görselde),
    (2) renk-sayısı bump'ı — ΔE'yi düşürmenin asıl yolu; daha yüksek k ile yeniden
    ön işleyip yeniden trace eder. Yalnızca sadakati ``_REFINE_MIN_GAIN`` kadar
    artıran varyant benimsenir; aksi halde orijinal en iyi korunur.

    Döner: (yeni_best, refine_info). ``scored`` listesi yeni adaylarla genişler.
    """
    info: dict[str, Any] = {"applied": False}
    if mode not in FIDELITY_LED_MODES or best is None or not best.get("rendered_ok"):
        return best, info

    base_fid = float(best.get("fidelity_score") or 0.0)
    if base_fid >= 99.0:  # zaten kusursuza yakın
        return best, info

    spec = build_vector_candidates(mode).get(best["name"])
    if not spec or spec.get("engine") != "vtracer":
        return best, info

    pool: list[dict[str, Any]] = [best]
    tried: list[dict[str, Any]] = []

    def _consider(sc: dict[str, Any] | None) -> None:
        if sc is None:
            return
        scored.append(sc)
        tried.append({"name": sc["name"], "fidelity": sc.get("fidelity_score"),
                      "path_count": (sc.get("score_details") or {}).get("path_count")})
        if sc.get("rendered_ok") and sc.get("fidelity_score") is not None:
            pool.append(sc)

    # 1) VTracer parametre varyantları (mevcut ön işlenmiş görsel)
    for vname, vparams in _refine_variants(spec):
        vspec = {"engine": "vtracer", "vtracer_params": vparams, "cleanup": spec.get("cleanup")}
        _consider(score_candidate(produce_candidate(vname, vspec, preprocessed_path, mode, job_dir),
                                  original_path, analysis, mode))

    # 2) Renk-sayısı bump'ı (ΔE odaklı): daha yüksek k ile yeniden ön işle + trace
    cur_k = int(analysis.get("estimated_color_count", 14))
    for bump in (8, 16):
        k = min(48, max(16, cur_k + bump))
        try:
            pp_path, _ = preprocess_for_mode(
                original_path, mode, job_dir, analysis=analysis,
                color_override=k, output_suffix=f"_k{k}",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("refine ön işleme atlandı (k=%s): %s", k, e)
            continue
        vname = f"refine_k{k}"
        vspec = {"engine": spec["engine"],
                 "vtracer_params": dict(spec.get("vtracer_params") or {}),
                 "cleanup": spec.get("cleanup")}
        _consider(score_candidate(produce_candidate(vname, vspec, pp_path, mode, job_dir),
                                  original_path, analysis, mode))

    # Benimseme: seçimle AYNI sıralama anahtarı (fidelity → az path → total).
    # Orijinal best havuzda; kazanan oysa hiçbir şey değişmez.
    improved = max(pool, key=_fidelity_rank_key)
    applied = improved is not best
    info = {
        "applied": applied,
        "base_fidelity": round(base_fid, 2),
        "refined_fidelity": round(float(improved.get("fidelity_score") or base_fid), 2),
        "winner": improved["name"],
        "tried": tried,
    }
    return improved, info


# ---------------------------------------------------------------------------
# Çekirdek pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    image: Image.Image,
    original_path: Path,
    trace_mode: str,
    job_dir: Path,
    refine: bool = True,
) -> dict[str, Any]:
    """Analiz → ön işleme → aday → temizleme → skor → seçim → refinement akışı.

    Export yapılmaz. Dönen sözlük:
    ``analysis, mode_used, mode_warning, preprocess_report, results, scored,
    best, raw_best, selection_reason, refine_info``. ``scored`` boşsa ``best``
    ``None`` olur. ``refine=False`` ile refinement kapatılabilir (ölçüm/karşılaştırma).
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
    results = [
        produce_candidate(name, spec, preprocessed_path, mode_used, job_dir)
        for name, spec in candidates.items()
    ]

    # 5. Skorlama
    scored: list[dict[str, Any]] = []
    for res in results:
        sc = score_candidate(res, original_path, analysis, mode_used)
        if sc is not None:
            scored.append(sc)

    best: dict[str, Any] | None = None
    raw_best: dict[str, Any] | None = None
    selection_reason = "no_candidate"
    refine_info: dict[str, Any] = {"applied": False}
    if scored:
        # 6. + 7. Seçim
        best, raw_best, selection_reason = select_best(scored, mode_used)
        # 8. Refinement (kapalı döngü): en iyi adayı hata-güdümlü iyileştir
        if refine:
            best, refine_info = refine_best(
                best, mode_used, analysis, original_path, preprocessed_path, job_dir, scored
            )
            if refine_info.get("applied"):
                selection_reason = "refined"

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
        "refine_info": refine_info,
    }
