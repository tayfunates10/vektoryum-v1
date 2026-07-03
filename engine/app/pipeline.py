"""Çekirdek vektörleştirme pipeline'ı (export öncesi).

Adımlar 1-7'yi (analiz → ön işleme → aday üretimi → geometri temizleme →
skorlama → seçim) tek bir yeniden kullanılabilir fonksiyonda toplar. Hem FastAPI
endpoint'i (``app.main``) hem de ölçüm CLI'si (``regression/fidelity_report.py``)
bu fonksiyonu çağırır; böylece iki yol arasında davranış sapması olmaz.

Export (SVG/PDF/EPS/DXF) ve HTTP'ye özgü her şey ``app.main``'de kalır.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.analyzer import analyze_image_from_mem
from app.curve_fairing import fair_svg_curves
from app.fidelity import score_structure_integrity
from app.geometry_cleanup import cleanup_svg_geometry, consolidate_svg_palette
from app.preprocess import preprocess_for_mode
from app.scoring import score_vector_candidate
from app.shape_fitting import fit_whole_shapes_svg, regularize_svg_geometry
from app.vector_engines import build_vector_candidates, get_autotrace_path, run_candidate

logger = logging.getLogger(__name__)

# Mod bazlı son palet üst sınırı (VTracer'ın kenar ara-tonlarını temizlemek için).
# geometric/minimal: taban palet + korunan canlı aksanlar (bkz. _reduce_to_dominant
# protect_chromatic) kırpılmasın diye taban k + 2 pay bırakılır.
PALETTE_CAP = {
    "geometric_logo": 6,
    "minimal_ai": 7,
    "flat_logo": 7,
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
# Düzenlenebilirlik-duyarlı seçim eşikleri: sadakat bu kadar puan içindeyken
# (marj), path sayısı bu oranın altına inen aday "çok daha düzenlenebilir" sayılıp
# tercih edilir (az path = düzenlenebilir + gradyanda sonsuz pürüzsüz çıktı).
_EDIT_MARGIN = 2.5
_EDIT_LEAN_RATIO = 0.5
# Düzenlenebilirlik için path azaltırken kenar bütünlüğünden bu kadardan fazla
# ödün verilmez. Az-path ama kenarı bozuk (ince çizgi parçalanmış) aday seçilmez.
_EDIT_EDGE_TOL = 0.03
# Az-path tercihi kalite eşiğinin (quality._LOW_FIDELITY_THRESHOLD) altına
# İNEMEZ: en sadık aday eşiğin üstündeyken marj içindeki daha düşük aday eşiğin
# altına düşüyorsa seçilirse çıktı gereksiz yere needs_review'a döner
# (gerçek bir seçim hatasıydı).
_EDIT_QUALITY_FLOOR = 78.0


def _path_count(c: dict[str, Any]) -> int:
    return int((c.get("score_details") or {}).get("path_count", 0))


def _edge_f1(c: dict[str, Any]) -> float:
    return float((c.get("score_details") or {}).get("edge_f1") or 0.0)


def _apply_editability_preference(
    scored: list[dict[str, Any]], current_best: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Sadakat marjı içinde belirgin şekilde daha az path'li adayı tercih eder.

    En yüksek sadakatli adayı bulur; sadakati ondan en çok ``_EDIT_MARGIN`` düşük
    OLAN ve kenar bütünlüğü (edge_f1) belirgin düşük OLMAYAN adaylar arasında en az
    path'liyi seçer — ancak yalnızca path sayısı en iyinin ``_EDIT_LEAN_RATIO``
    katından azsa. Kenar koruması: az-path uğruna ince çizgileri parçalayan
    (edge_f1 düşük) adaya geçilmez. Aksi halde en yüksek sadakatli aday kalır.
    """
    rendered = [c for c in scored if c.get("rendered_ok") and c.get("fidelity_score") is not None]
    if not rendered:
        return current_best, "highest_total_score"

    top = max(rendered, key=_fidelity_rank_key)
    top_fid = float(top["fidelity_score"])
    top_paths = max(1, _path_count(top))
    top_edge = _edge_f1(top)

    fid_floor = top_fid - _EDIT_MARGIN
    if top_fid >= _EDIT_QUALITY_FLOOR:
        fid_floor = max(fid_floor, _EDIT_QUALITY_FLOOR)
    eligible = [
        c for c in rendered
        if float(c["fidelity_score"]) >= fid_floor
        and _edge_f1(c) >= top_edge - _EDIT_EDGE_TOL
    ]
    leanest = min(eligible, key=_path_count)
    if leanest is not top and _path_count(leanest) <= _EDIT_LEAN_RATIO * top_paths:
        return leanest, "editability_preference"
    return top, "highest_fidelity"


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

    # SADAKAT GÜVENLİĞİ: yapısal tercih sıralaması, ölçülen algısal sadakati
    # belirgin (>1.5 puan) daha yüksek bir adayı gölgede bırakmasın. Geometri
    # temizliği anlamlı ölçüde düşük olmayan en sadık aday varsa ona geçilir
    # (temiz çizgi önceliği korunur, ama gözle görülür bozulma pahasına değil).
    rendered = [c for c in viable if c.get("rendered_ok") and c.get("fidelity_score") is not None]
    if rendered and chosen.get("rendered_ok") and chosen.get("fidelity_score") is not None:
        top_fid = max(rendered, key=lambda c: float(c["fidelity_score"]))
        if (
            top_fid is not chosen
            and float(top_fid["fidelity_score"]) > float(chosen["fidelity_score"]) + 1.5
            and g(top_fid, "geometry_score") >= g(chosen, "geometry_score") - 5.0
            and g(top_fid, "corner_cleanliness_score") >= g(chosen, "corner_cleanliness_score") - 5.0
        ):
            chosen, reason = top_fid, "fidelity_guard"

    return chosen, raw_best, reason


# Refinement uygulanan modlar: piksel sadakatinin asıl hedef olduğu renkli
# modlar. Geometrik/sade modlarda "temiz çizgi" önceliklidir, ham sadakat
# uğruna geometriyi bozmayız; o yüzden onlar refinement dışıdır.
FIDELITY_LED_MODES = {"logo_color", "photo_poster"}


# ---------------------------------------------------------------------------
# İnce kontur koruması
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _has_thin_strokes(preprocessed_path_str: str) -> bool:
    """Ön işlenmiş görselde ince (<=~3px) kontur oranı belirgin mi?

    VTracer'ın ``filter_speckle`` filtresi yalnız gürültü beneklerini değil,
    genişliği eşiğin altındaki ÇİZGİLERİ de tümüyle siler (2px kırmızı çizginin
    çıktıdan kaybolması gerçek bir hataydı). Mürekkep piksellerinin anlamlı bir
    bölümü ince konturdaysa (distance-transform yarı-genişliği <= 1.5px), speckle
    filtresi güvenle kapatılır; gürültüyü zaten ön işleme despeckle'ı temizler.
    """
    try:
        img = cv2.imread(preprocessed_path_str, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        # zemin: köşe medyanı
        pw, ph = max(8, w // 12), max(8, h // 12)
        corners = np.concatenate([
            img[:ph, :pw].reshape(-1, 3), img[:ph, -pw:].reshape(-1, 3),
            img[-ph:, :pw].reshape(-1, 3), img[-ph:, -pw:].reshape(-1, 3),
        ]).astype(np.float32)
        bg = np.median(corners, axis=0)
        dist = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
        ink = (dist > 40).astype(np.uint8)
        n_ink = int(ink.sum())
        if n_ink < 50:
            return False
        dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        thin_ratio = float(((dt > 0) & (dt <= 1.5)).sum()) / float(n_ink)
        return thin_ratio >= 0.20
    except Exception:  # noqa: BLE001
        return False


def _adapt_spec_for_thin_strokes(spec: dict[str, Any], preprocessed_path: Path) -> dict[str, Any]:
    """İnce konturlu görselde VTracer speckle filtresini kapatır (çizgi silinmesin)."""
    params = spec.get("vtracer_params") or {}
    if spec.get("engine") != "vtracer" or int(params.get("filter_speckle", 0) or 0) <= 0:
        return spec
    if not _has_thin_strokes(str(preprocessed_path)):
        return spec
    return {**spec, "vtracer_params": {**params, "filter_speckle": 0}}


# ---------------------------------------------------------------------------
# Tek aday üretimi + skorlama (ana döngü ve refinement ortak kullanır)
# ---------------------------------------------------------------------------
def produce_candidate(
    name: str,
    spec: dict[str, Any],
    preprocessed_path: Path,
    mode: str,
    job_dir: Path,
    original_path: Path | None = None,
    palette_cap: int | None = None,
) -> dict[str, Any]:
    """Tek bir adayı üretir: trace → cleanup → palet konsolidasyonu → regularize.

    ``original_path`` gradyan motoruna iletilir (ham pikseller gerekir).
    ``palette_cap`` verilirse sabit PALETTE_CAP yerine kullanılır (logo_color'da
    preprocess renk bütçesine bağlanan adaptif cap için).
    """
    svg_path = job_dir / f"{name}.svg"
    engine = spec["engine"]
    try:
        spec = _adapt_spec_for_thin_strokes(spec, preprocessed_path)
        run_candidate(engine, preprocessed_path, svg_path, spec, original_path=original_path)
        cleanup_report: dict[str, Any] = {}
        if spec.get("cleanup"):
            cleanup_report = cleanup_svg_geometry(svg_path, mode=mode, aggressiveness=spec["cleanup"])
        # palet konsolidasyonu: kenar ara-ton renklerini en baskın renklere indir.
        # Gradyan adayında ATLA — gradyan stop'larını/url() fill'lerini bozar.
        cap = palette_cap if palette_cap is not None else PALETTE_CAP.get(mode)
        if cap and engine != "gradient":
            canonical = CANONICAL_BWR if mode in FLAT_PALETTE_MODES else None
            # geniş paletli (foto-zengin) çıktıda ince ton merdivenleri 8-15
            # RGB adımlıdır; varsayılan merge_tol=12 bunları birbirine
            # yapıştırıp detayı düzleştirir -> daha sıkı tolerans
            merge_tol = 6.0 if (mode == "logo_color" and cap >= 40) else 12.0
            consolidate_svg_palette(svg_path, max_colors=cap, canonical=canonical, merge_tol=merge_tol)
        # geometrik idealleştirme: düz çizgi + tam dairesel yay oturtma
        # (bütünsel şekil oturtma dahil — regularize önce tam şekli dener)
        if mode in REGULARIZE_MODES:
            try:
                regularize_svg_geometry(svg_path)
            except Exception as reg_err:  # noqa: BLE001
                logger.debug("regularize atlandı (%s): %s", name, reg_err)
        elif engine == "vtracer":
            # renkli modlarda yalnız BÜTÜNSEL şekil oturtma: organik path'lere
            # dokunulmaz, gerçekten daire/elips/dikdörtgen olan alt yollar
            # ideal parametrik şekle döner (çift yönlü sapma toleransı sıkı)
            try:
                fit_whole_shapes_svg(svg_path)
            except Exception as ws_err:  # noqa: BLE001
                logger.debug("whole-shape fitting atlandı (%s): %s", name, ws_err)
        # eğri pürüzsüzleştirme (tangent matching): spline eklemlerindeki küçük
        # açılı kinkler G1 sürekliliğe çekilir; köşeler ve düz çizgiler korunur.
        # Gradyan adayı atlanır (tek path, el-yapımı geometri).
        if engine == "vtracer":
            try:
                fair_svg_curves(svg_path)
            except Exception as fair_err:  # noqa: BLE001
                logger.debug("curve fairing atlandı (%s): %s", name, fair_err)
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
        _consider(score_candidate(produce_candidate(vname, vspec, preprocessed_path, mode, job_dir,
                                                     original_path=original_path),
                                  original_path, analysis, mode))

    # 2) Renk-sayısı bump'ı (ΔE odaklı): daha yüksek k ile yeniden ön işle + trace
    cur_k = int(analysis.get("estimated_color_count", 14))
    for bump in (8, 16):
        k = min(64, max(16, cur_k + bump))
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
        _consider(score_candidate(produce_candidate(vname, vspec, pp_path, mode, job_dir,
                                                     original_path=original_path),
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
# Renk refit (kapalı-form renk optimizasyonu; bkz. app/color_refit.py)
# ---------------------------------------------------------------------------
# Refit uygulanan modlar: renk sadakati hedef olan renkli modlar + geometrik
# logo (kanonik palet orijinal mürekkep tonundan sapabilir; refit ölçülen
# sadakat artarsa gerçek tona çeker). Düz/binary modlarda kanonik siyah-beyaz
# stilizasyon bilinçli tercih olduğundan dokunulmaz.
_COLOR_REFIT_MODES = {"logo_color", "photo_poster", "geometric_logo"}


def _apply_color_refit(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Kazanan adayın dolgu renklerini orijinale yeniden oturtur (ölçüm korumalı).

    Yeni SVG ayrı dosyaya yazılır, aynı skorlayıcıyla puanlanır ve YALNIZCA
    ölçülen fidelity artarsa benimsenir; aksi halde eski kazanan aynen kalır.
    Gradyan uzanımı yalnız fidelity-led modlarda denenir (geometrik modda düz
    dolgu idealdir).
    """
    info: dict[str, Any] = {"applied": False}
    if (
        best is None
        or not best.get("rendered_ok")
        or best.get("fidelity_score") is None
        or best.get("engine") == "gradient"  # url() dolguları zaten optimize
        or mode not in _COLOR_REFIT_MODES
    ):
        return best, info

    # foto-yoğun çıktı geçidi: çok yüksek path sayılı sonuçlarda aynı kuantize
    # rengi uzak, ilgisiz bölgelere dağılır; global-renk havuzlama yerel ΔE'yi
    # ARTIRIR (ölçüm sonucu refit reddedilir). Bu durumda pahalı ID-render'ı
    # boşuna yapmamak için baştan atlanır — güvenlik geçidi yine de korur ama
    # gereksiz maliyeti önleriz. Düz çok-renkli logolar (~yüzlerce path) geçer.
    if int((best.get("score_details") or {}).get("path_count", 0)) > 700:
        return best, {"applied": False, "skipped": "high_path_count"}

    from app.color_refit import refit_svg_colors  # noqa: PLC0415

    src = Path(best["svg_path"])
    dst = job_dir / f"{src.stem}_refit.svg"
    try:
        rep = refit_svg_colors(
            src, original_path, dst, gradients=(mode in FIDELITY_LED_MODES)
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("color refit atlandı: %s", e)
        return best, info
    if not rep.get("changed"):
        return best, {"applied": False, **rep}

    res = {
        "name": f"{best['name']}_refit",
        "svg_path": dst,
        "engine": best.get("engine"),
        "cleanup_report": best.get("cleanup_report", {}),
        "success": True,
        "error": None,
    }
    sc = score_candidate(res, original_path, analysis, mode)
    if sc is None or not sc.get("rendered_ok") or sc.get("fidelity_score") is None:
        return best, {"applied": False, **rep}
    base_fid = float(best["fidelity_score"])
    new_fid = float(sc["fidelity_score"])
    info = {
        "applied": new_fid > base_fid,
        "base_fidelity": round(base_fid, 2),
        "refit_fidelity": round(new_fid, 2),
        **{k: v for k, v in rep.items() if k != "changed"},
        "fills_changed": rep.get("changed"),
    }
    if new_fid <= base_fid:
        return best, info
    scored.append(sc)
    return sc, info


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
    # Gradyan-farkındalıklı aday yalnızca gradyan olasılığı olan renkli logolarda
    # eklenir (çok-renkli düz logolarda boşuna çalışmasın). Gradyanlar çoğu kez az
    # sayıda baskın renge çöküp 'color logo' olarak okunur; bu yüzden has_gradient
    # YA DA (color logo + az renk) tetikler. Ham görselde çalışır, <linearGradient>
    # üretir; fidelity yargılar, uygun değilse elenir.
    if mode_used == "logo_color" and (
        analysis.get("has_gradient")
        or (analysis.get("likely_color_logo") and int(analysis.get("estimated_color_count", 99)) <= 8)
    ):
        candidates = {
            **candidates,
            "logo_gradient": {"engine": "gradient", "params": {"epsilon": 0.3}, "cleanup": None},
        }
    # logo_color'da palet cap'i preprocess renk bütçesine bağlanır (sabit 22 yerine):
    # üretilen renkler kırpılmaz, renk-zengini logolarda ΔE düşer. Hata-güdümlü
    # eklenen aksan kümeleri k'yı aşabildiğinden GERÇEK renk sayısı esas alınır
    # (aksi halde konsolidasyon kırmızı/turuncu aksanları geri kırpar).
    lc_cap = None
    if mode_used == "logo_color":
        lc_cap = max(
            int(preprocess_report.get("auto_color_count") or 0),
            int(preprocess_report.get("actual_color_count") or 0),
        ) or None
        if lc_cap:
            lc_cap = min(64, lc_cap)  # üretim paleti üst sınırı (quality eşiği 64)
    results = [
        produce_candidate(name, spec, preprocessed_path, mode_used, job_dir,
                          original_path=original_path, palette_cap=lc_cap)
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
    refit_info: dict[str, Any] = {"applied": False}
    if scored:
        # 6. + 7. Seçim (fidelity-led; refinement'ı tohumlar)
        best, raw_best, selection_reason = select_best(scored, mode_used)
        # 8. Refinement (kapalı döngü): en iyi adayı hata-güdümlü iyileştir
        if refine:
            best, refine_info = refine_best(
                best, mode_used, analysis, original_path, preprocessed_path, job_dir, scored
            )
            if refine_info.get("applied"):
                selection_reason = "refined"
        # 9. Düzenlenebilirlik-duyarlı son seçim (renkli modlar): sadakat marjı
        # içinde çok daha az path'li/gradyanlı adayı tercih et. FOTO-ZENGİN
        # görsellerde (est >= 22: çok tonlu fotoğrafik içerik) uygulanmaz —
        # orada detay ürünün kendisidir; az-path uğruna sadakatten ödün vermek
        # çıktıyı gözle görülür düzleştirir.
        if mode_used in FIDELITY_LED_MODES and int(analysis.get("estimated_color_count", 0)) < 22:
            edit_best, edit_reason = _apply_editability_preference(scored, best)
            if edit_best is not best:
                best, selection_reason = edit_best, edit_reason
        # 9.5 Renk refit (kapalı-form renk optimizasyonu): kazananın dolguları
        # orijinal görüntünün bölge medyanlarına oturtulur; ölçülen sadakat
        # artmazsa benimsenmez. İzleme sonrası kaybın ana bileşeni renktir —
        # tavan analizi: kayıpların ~%60-85'i sabit dolgu renk sapmasından.
        best, refit_info = _apply_color_refit(
            best, mode_used, analysis, original_path, job_dir, scored
        )
        if refit_info.get("applied"):
            selection_reason = f"{selection_reason}+color_refit"

    # 10. Yapı bütünlüğü denetimi (kırık/eksik çizgi, hayalet çizik): nihai
    # çıktıda orijinaldeki her kontur karşılanıyor mu? Foto benzeri sürekli-tonlu
    # girdilerde ve düz olmayan zeminlerde mürekkep eşiği güvenilir olmadığından
    # atlanır. Render backend'i yoksa None kalır (çökme yok).
    structure_report = None
    if (
        best is not None
        and mode_used != "photo_poster"
        and (analysis.get("background") or {}).get("is_uniform_background")
    ):
        structure_report = score_structure_integrity(best["svg_path"], original_path)

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
        "refit_info": refit_info,
        "structure_report": structure_report,
    }
