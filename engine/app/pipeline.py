"""Çekirdek vektörleştirme pipeline'ı (export öncesi).

Adımlar 1-7'yi (analiz → ön işleme → aday üretimi → geometri temizleme →
skorlama → seçim) tek bir yeniden kullanılabilir fonksiyonda toplar. Hem FastAPI
endpoint'i (``app.main``) hem de ölçüm CLI'si (``regression/fidelity_report.py``)
bu fonksiyonu çağırır; böylece iki yol arasında davranış sapması olmaz.

Export (SVG/PDF/EPS/DXF) ve HTTP'ye özgü her şey ``app.main``'de kalır.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
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
from app.neutral_palette import detect_neutral_luminance_bands, geometric_preserves_neutral_palette
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

_JOURNAL_IMAGE_CLASS = {
    "logo_color": "clean_logo", "flat_logo": "clean_logo",
    "geometric_logo": "geometric", "minimal_ai": "clean_logo",
    "single_color": "geometric", "lineart": "lineart", "centerline": "lineart",
    "photo_poster": "photo",
}


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


def _journal_source_rgb(image_path: Path) -> np.ndarray:
    """Stage gate referansını güvenli biçimde beyaz zemine kompoze eder."""
    with Image.open(image_path) as source:
        if source.mode in ("RGBA", "LA", "PA") or (
            source.mode == "P" and "transparency" in source.info
        ):
            rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            return np.clip(
                rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
                0, 255,
            ).astype(np.uint8)
        return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()


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
        from app.transform_journal import TransformJournal  # noqa: PLC0415

        candidate_source_rgb = _journal_source_rgb(preprocessed_path)
        candidate_journal = TransformJournal(
            svg_path, candidate_source_rgb,
            image_class=_JOURNAL_IMAGE_CLASS.get(mode, "clean_logo"),
            max_side=256,
        )
        # VTracer bazı çıktılarda yalnız width/height + translate üretir;
        # viewBox yokken sonraki mutator'ları yapısal olarak ölçmek mümkün
        # değildir. İlk transaction bu engine artifact'ını trace raster'ının
        # koordinat uzayına normalize eder; başarısızsa raw parent korunur.
        candidate_journal.run_in_place(
            "establish_coordinate_contract", svg_path,
            lambda path: _restore_source_dimensions(
                path,
                {"width": candidate_source_rgb.shape[1],
                 "height": candidate_source_rgb.shape[0]},
            ),
        )
        cleanup_report: dict[str, Any] = {}
        if spec.get("cleanup"):
            _ok, cleanup_value, cleanup_stage = candidate_journal.run_in_place(
                "geometry_cleanup", svg_path,
                lambda path: cleanup_svg_geometry(
                    path, mode=mode, aggressiveness=spec["cleanup"],
                ),
            )
            cleanup_report = cleanup_value or {
                "status": cleanup_stage["status"],
                "journal_reasons": cleanup_stage["reason_codes"],
            }
        # palet konsolidasyonu: kenar ara-ton renklerini en baskın renklere indir.
        # Gradyan adayında ATLA — gradyan stop'larını/url() fill'lerini bozar.
        # FOTO-YOĞUN izleme çıktısında da ATLA (fidelity-led modlar, >700 path):
        # izole ölçüm (aslan bölge ΔE): kaynak 8.6 -> vtracer 15.8 -> konsolidasyon
        # 19.1 — ton merdivenini ezmek foto içerikte saf kayıptır; per-path renk
        # refit'i zaten her bölgeyi orijinaline oturtur (ham+refit 85.7 vs
        # konsolide+refit 84.8). Logo/marka çıktısında (az path) hijyen için kalır.
        cap = palette_cap if palette_cap is not None else PALETTE_CAP.get(mode)
        if cap and engine != "gradient":
            try:
                n_traced_paths = svg_path.read_text(encoding="utf-8", errors="ignore").count("<path")
            except OSError:
                n_traced_paths = 0
            if not (mode in FIDELITY_LED_MODES and n_traced_paths > 700):
                canonical = CANONICAL_BWR if mode in FLAT_PALETTE_MODES else None
                # P0-B2a: a geometric logo with three or more *real source*
                # neutral luminance plateaus must keep those greys.  Only the
                # canonical snap is disabled; palette cap, trace family and
                # winner heuristics remain byte-for-byte policy-equivalent.
                if canonical is not None and original_path is not None:
                    neutral_report = detect_neutral_luminance_bands(original_path)
                    if geometric_preserves_neutral_palette(mode, neutral_report):
                        canonical = None
                # geniş paletli (foto-zengin) çıktıda ince ton merdivenleri 8-15
                # RGB adımlıdır; varsayılan merge_tol=12 bunları birbirine
                # yapıştırıp detayı düzleştirir -> daha sıkı tolerans
                merge_tol = 6.0 if (mode == "logo_color" and cap >= 40) else 12.0
                candidate_journal.run_in_place(
                    "palette_consolidation", svg_path,
                    lambda path: consolidate_svg_palette(
                        path, max_colors=cap, canonical=canonical, merge_tol=merge_tol,
                    ),
                )
        # geometrik idealleştirme: düz çizgi + tam dairesel yay oturtma
        # (bütünsel şekil oturtma dahil — regularize önce tam şekli dener)
        if mode in REGULARIZE_MODES:
            try:
                candidate_journal.run_in_place(
                    "geometry_regularization", svg_path, regularize_svg_geometry,
                )
            except Exception as reg_err:  # noqa: BLE001
                logger.debug("regularize atlandı (%s): %s", name, reg_err)
        elif engine in {"vtracer", "gradient"}:
            try:
                candidate_journal.run_in_place(
                    "whole_shape_fitting", svg_path, fit_whole_shapes_svg,
                )
            except Exception as ws_err:  # noqa: BLE001
                logger.debug("whole-shape fitting atlandı (%s): %s", name, ws_err)
        if engine == "vtracer":
            try:
                candidate_journal.run_in_place(
                    "curve_fairing", svg_path, fair_svg_curves,
                )
            except Exception as fair_err:  # noqa: BLE001
                logger.debug("curve fairing atlandı (%s): %s", name, fair_err)
        return {
            "name": name,
            "svg_path": svg_path,
            "engine": spec["engine"],
            "cleanup_report": cleanup_report,
            "candidate_transform_journal": candidate_journal.to_dict(),
            "success": True,
            "error": None,
        }
    except FileNotFoundError as e:
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
# Paralel aday üretimi
# ---------------------------------------------------------------------------
def _pool_size() -> int:
    env = os.environ.get("VEKTORYUM_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 4))


def _pool_init() -> None:
    try:
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass


_POOL: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor | None:
    global _POOL
    if _POOL is not None:
        return _POOL
    workers = _pool_size()
    if workers <= 1:
        return None
    try:
        import multiprocessing as mp  # noqa: PLC0415

        thread_env = {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        }
        saved = {k: os.environ.get(k) for k in thread_env}
        os.environ.update(thread_env)
        try:
            method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            _POOL = ProcessPoolExecutor(
                max_workers=workers, initializer=_pool_init,
                mp_context=mp.get_context(method),
            )
            _POOL.submit(_pool_init).result(timeout=180)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    except Exception as e:  # noqa: BLE001
        logger.warning("Süreç havuzu kurulamadı, sıralı mod: %s", e)
        _POOL = None
    return _POOL


def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:
    (name, spec, preprocessed_path, mode, job_dir,
     original_path, palette_cap, analysis, pp_override) = args
    if pp_override is not None:
        try:
            preprocessed_path, _ = preprocess_for_mode(
                original_path, mode, job_dir, analysis=analysis,
                color_override=pp_override, output_suffix=f"_k{pp_override}",
            )
        except Exception as e:  # noqa: BLE001
            return {"name": name, "success": False, "error": str(e),
                    "engine": spec.get("engine")}, None
    res = produce_candidate(name, spec, preprocessed_path, mode, job_dir,
                            original_path=original_path, palette_cap=palette_cap)
    sc = score_candidate(res, original_path, analysis, mode)
    return res, sc


class WorkerFailure(RuntimeError):
    """Paralel aday havuzu başarısız/zaman aşımı — KONTROLLÜ hata."""


def _job_timeout() -> float:
    env = os.environ.get("VEKTORYUM_JOB_TIMEOUT", "").strip()
    if env:
        try:
            return max(30.0, float(env))
        except ValueError:
            pass
    return 600.0


def _reset_pool(kill: bool = True) -> None:
    global _POOL
    pool = _POOL
    _POOL = None
    if pool is None:
        return
    if kill:
        try:
            for p in list(getattr(pool, "_processes", {}).values()):
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        try:
            pool.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def _run_jobs(jobs: list[tuple]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    if len(jobs) <= 1 or _pool_size() <= 1:
        return [_produce_and_score_job(j) for j in jobs]
    pool = _get_pool()
    if pool is None:
        return [_produce_and_score_job(j) for j in jobs]
    try:
        return list(pool.map(_produce_and_score_job, jobs, timeout=_job_timeout()))
    except Exception as e:  # noqa: BLE001
        logger.warning("Paralel aday üretimi başarısız (%s) — havuz sıfırlanıyor; inline sıralı yeniden çalıştırma YAPILMIYOR", e)
        _reset_pool(kill=True)
        raise WorkerFailure(str(e)) from e


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
def _refine_variants(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
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
    best: dict[str, Any], mode: str, analysis: dict[str, Any], original_path: Path,
    preprocessed_path: Path, job_dir: Path, scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {"applied": False}
    if mode not in FIDELITY_LED_MODES or best is None or not best.get("rendered_ok"):
        return best, info

    base_fid = float(best.get("fidelity_score") or 0.0)
    if base_fid >= 99.0:
        return best, info

    spec = build_vector_candidates(mode).get(best["name"])
    if not spec or spec.get("engine") != "vtracer":
        return best, info

    pool: list[dict[str, Any]] = [best]
    tried: list[dict[str, Any]] = []
    jobs: list[tuple] = []
    for vname, vparams in _refine_variants(spec):
        vspec = {"engine": "vtracer", "vtracer_params": vparams, "cleanup": spec.get("cleanup")}
        jobs.append((vname, vspec, preprocessed_path, mode, job_dir,
                     original_path, None, analysis, None))
    cur_k = int(analysis.get("estimated_color_count", 14))
    for bump in (8, 16):
        k = min(64, max(16, cur_k + bump))
        vname = f"refine_k{k}"
        vspec = {"engine": spec["engine"],
                 "vtracer_params": dict(spec.get("vtracer_params") or {}),
                 "cleanup": spec.get("cleanup")}
        jobs.append((vname, vspec, None, mode, job_dir,
                     original_path, None, analysis, k))

    for _res, sc in _run_jobs(jobs):
        if sc is None:
            continue
        scored.append(sc)
        tried.append({"name": sc["name"], "fidelity": sc.get("fidelity_score"),
                      "path_count": (sc.get("score_details") or {}).get("path_count")})
        if sc.get("rendered_ok") and sc.get("fidelity_score") is not None:
            pool.append(sc)

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


_COLOR_REFIT_MODES = {"logo_color", "photo_poster", "geometric_logo"}
_REFIT_TOP_N = 4


def _refit_one(
    cand: dict[str, Any], mode: str, analysis: dict[str, Any],
    original_path: Path, job_dir: Path,
) -> dict[str, Any] | None:
    if (
        not cand.get("rendered_ok")
        or cand.get("fidelity_score") is None
        or cand.get("engine") == "gradient"
        or cand["name"].endswith("_refit")
        or mode not in _COLOR_REFIT_MODES
    ):
        return None
    from app.color_refit import refit_svg_colors, refit_svg_colors_per_path  # noqa: PLC0415

    src = Path(cand["svg_path"])
    dst = job_dir / f"{src.stem}_refit.svg"
    path_count = int((cand.get("score_details") or {}).get("path_count", 0))
    photo_rich = path_count > 700
    try:
        if photo_rich:
            if path_count > 12000:
                return None
            rep = refit_svg_colors_per_path(src, original_path, dst)
        else:
            if path_count > 700:
                return None
            rep = refit_svg_colors(src, original_path, dst, gradients=(mode in FIDELITY_LED_MODES))
    except Exception as e:  # noqa: BLE001
        logger.debug("color refit atlandı (%s): %s", cand["name"], e)
        return None
    if not rep.get("changed"):
        return None

    from app.transform_journal import TransformJournal, merge_journal_reports  # noqa: PLC0415

    required_metrics: set[str] = set()
    try:
        with Image.open(original_path) as original_image:
            if original_image.mode in ("RGBA", "LA", "PA") or (
                original_image.mode == "P" and "transparency" in original_image.info
            ):
                required_metrics.add("alpha_fidelity")
    except Exception:  # noqa: BLE001
        pass
    if analysis.get("has_gradient"):
        required_metrics.add("gradient_fidelity")
    try:
        refit_journal = TransformJournal(
            src, _journal_source_rgb(original_path),
            image_class=_JOURNAL_IMAGE_CLASS.get(mode, "clean_logo"),
            required_metrics=required_metrics,
        )
        accepted_path, refit_stage = refit_journal.consider_candidate(
            "color_refit", src, dst, transform_report=rep,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("color refit journal kurulamadı (%s): %s", cand["name"], e)
        return None
    if accepted_path != dst:
        logger.debug("color refit journal tarafından geri alındı (%s): %s", cand["name"], refit_stage["reason_codes"])
        return None

    res = {
        "name": f"{cand['name']}_refit",
        "svg_path": dst,
        "engine": cand.get("engine"),
        "cleanup_report": cand.get("cleanup_report", {}),
        "candidate_transform_journal": merge_journal_reports(
            cand.get("candidate_transform_journal"), refit_journal.to_dict(),
        ),
        "success": True,
        "error": None,
    }
    sc = score_candidate(res, original_path, analysis, mode)
    if sc is None or not sc.get("rendered_ok") or sc.get("fidelity_score") is None:
        return None
    if float(sc["fidelity_score"]) <= float(cand["fidelity_score"]):
        return None
    sc["_refit_rep"] = {
        "base_fidelity": round(float(cand["fidelity_score"]), 2),
        "refit_fidelity": round(float(sc["fidelity_score"]), 2),
        **{k: v for k, v in rep.items() if k != "changed"},
        "fills_changed": rep.get("changed"),
    }
    return sc


def _refit_candidates_preselect(
    scored: list[dict[str, Any]], mode: str, analysis: dict[str, Any],
    original_path: Path, job_dir: Path,
) -> dict[str, Any]:
    if mode not in _COLOR_REFIT_MODES:
        return {"applied": False}
    rendered = [
        c for c in scored
        if c.get("rendered_ok") and c.get("fidelity_score") is not None
        and not c["name"].endswith("_refit")
    ]
    rendered.sort(key=lambda c: -float(c["fidelity_score"]))
    best_info: dict[str, Any] = {"applied": False}
    for cand in rendered[:_REFIT_TOP_N]:
        sc = _refit_one(cand, mode, analysis, original_path, job_dir)
        if sc is None:
            continue
        rep = sc.pop("_refit_rep", {})
        scored.append(sc)
        if not best_info.get("applied") or rep.get("refit_fidelity", 0) > best_info.get("refit_fidelity", 0):
            best_info = {"applied": True, **rep, "candidate": sc["name"]}
    return best_info


def _apply_boundary_refit(
    best: dict[str, Any], mode: str, analysis: dict[str, Any],
    original_path: Path, job_dir: Path, scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {"applied": False}
    if (
        best is None
        or not best.get("rendered_ok")
        or best.get("fidelity_score") is None
        or mode not in _COLOR_REFIT_MODES
    ):
        return best, info
    if int((best.get("score_details") or {}).get("path_count", 0)) > 2000:
        return best, {"applied": False, "skipped": "high_path_count"}

    from app.boundary_refit import refit_svg_boundaries  # noqa: PLC0415

    src = Path(best["svg_path"])
    dst = job_dir / f"{src.stem}_bnd.svg"
    try:
        rep = refit_svg_boundaries(src, original_path, dst)
    except Exception as e:  # noqa: BLE001
        logger.debug("boundary refit atlandı: %s", e)
        return best, info
    if not rep.get("moved"):
        return best, {"applied": False, **rep}

    res = {
        "name": f"{best['name']}_bnd",
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
        "bnd_fidelity": round(new_fid, 2),
        "anchors_moved": rep.get("moved"),
    }
    if new_fid <= base_fid:
        return best, info
    scored.append(sc)
    return sc, info


def _restore_source_dimensions(svg_path: Path, analysis: dict[str, Any]) -> None:
    ow, oh = int(analysis.get("width", 0) or 0), int(analysis.get("height", 0) or 0)
    if ow <= 0 or oh <= 0:
        return
    import re as _re  # noqa: PLC0415
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
        if not root.get("viewBox"):
            w_attr, h_attr = root.get("width"), root.get("height")
            if not w_attr or not h_attr:
                return
            try:
                vw = float(str(w_attr).rstrip("px"))
                vh = float(str(h_attr).rstrip("px"))
            except ValueError:
                return
            root.set("viewBox", f"0 0 {vw:g} {vh:g}")

        def _finish() -> None:
            root.set("width", str(ow))
            root.set("height", str(oh))
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)

        vb = root.get("viewBox", "").replace(",", " ").split()
        try:
            vbw, vbh = float(vb[2]), float(vb[3])
        except (IndexError, ValueError):
            _finish()
            return
        has_xf = any(el.get("transform") for el in root.iter() if el.tag.split("}")[-1] == "path")
        if abs(vbw - ow) < 1e-6 and abs(vbh - oh) < 1e-6 and not has_xf:
            _finish()
            return
        try:
            import numpy as _np  # noqa: PLC0415
            from svgpathtools import parse_path  # noqa: PLC0415
            from svgpathtools.path import transform as _svgt_transform  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            _finish()
            return
        geo_tags = {"rect", "circle", "ellipse", "line", "polygon", "polyline"}
        if any(el.tag.split("}")[-1] in geo_tags for el in root.iter()):
            _finish()
            return
        sx, sy = ow / vbw, oh / vbh

        def _round_d(d: str) -> str:
            return _re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.2f}".rstrip("0").rstrip("."), d)

        plan: list[tuple[Any, float, float, str]] = []
        for el in root.iter():
            if el.tag.split("}")[-1] != "path" or not el.get("d"):
                continue
            tx = ty = 0.0
            tr = (el.get("transform") or "").strip()
            if tr:
                m = _re.fullmatch(r"translate\(\s*([-\d.eE]+)\s*[, ]?\s*([-\d.eE]+)?\s*\)", tr)
                if not m:
                    _finish()
                    return
                tx, ty = float(m.group(1)), float(m.group(2) or 0.0)
            plan.append((el, tx, ty, tr))
        for el, tx, ty, tr in plan:
            mat = _np.array([[sx, 0, sx * tx], [0, sy, sy * ty], [0, 0, 1]], dtype=float)
            p2 = _svgt_transform(parse_path(el.get("d")), mat)
            el.set("d", _round_d(p2.d()))
            if tr:
                del el.attrib["transform"]
        root.set("viewBox", f"0 0 {ow:g} {oh:g}")
        _finish()
    except Exception as e:  # noqa: BLE001
        logger.debug("Kaynak boyut geri yüklemesi atlandı: %s", e)


def run_pipeline(
    image: Image.Image,
    original_path: Path,
    trace_mode: str,
    job_dir: Path,
    refine: bool = True,
    edge_cleanup: bool = True,
) -> dict[str, Any]:
    analysis = analyze_image_from_mem(image)
    mode_used = analysis["recommended_mode"] if trace_mode == "auto" else trace_mode
    mode_warning = compute_mode_warning(trace_mode, mode_used, analysis)

    preprocessed_path, preprocess_report = preprocess_for_mode(
        original_path, mode_used, job_dir, analysis=analysis
    )

    candidates = build_vector_candidates(mode_used)
    if mode_used == "logo_color" and (
        analysis.get("has_gradient")
        or (analysis.get("likely_color_logo") and int(analysis.get("estimated_color_count", 99)) <= 8)
    ):
        candidates = {
            **candidates,
            "logo_gradient": {"engine": "gradient", "params": {"epsilon": 0.3}, "cleanup": None},
        }
    lc_cap = None
    if mode_used == "logo_color":
        lc_cap = max(
            int(preprocess_report.get("auto_color_count") or 0),
            int(preprocess_report.get("actual_color_count") or 0),
        ) or None
        if lc_cap:
            lc_cap = min(64, lc_cap)
    jobs = [
        (name, spec, preprocessed_path, mode_used, job_dir,
         original_path, lc_cap, analysis, None)
        for name, spec in candidates.items()
    ]
    pairs = _run_jobs(jobs)
    results = [r for r, _ in pairs]
    scored: list[dict[str, Any]] = [s for _, s in pairs if s is not None]

    best: dict[str, Any] | None = None
    raw_best: dict[str, Any] | None = None
    selection_reason = "no_candidate"
    refine_info: dict[str, Any] = {"applied": False}
    refit_info: dict[str, Any] = {"applied": False}
    transform_journal = None
    if scored:
        best, raw_best, selection_reason = select_best(scored, mode_used)
        if refine:
            best, refine_info = refine_best(
                best, mode_used, analysis, original_path, preprocessed_path, job_dir, scored
            )
            if refine_info.get("applied"):
                selection_reason = "refined"
        refit_info = _refit_candidates_preselect(scored, mode_used, analysis, original_path, job_dir)
        if refit_info.get("applied"):
            best, raw_best, selection_reason = select_best(scored, mode_used)
            selection_reason = f"{selection_reason}+color_refit"
        if mode_used in FIDELITY_LED_MODES and int(analysis.get("estimated_color_count", 0)) < 22:
            edit_best, edit_reason = _apply_editability_preference(scored, best)
            if edit_best is not best:
                best = edit_best
                selection_reason = f"{edit_reason}+color_refit" if best["name"].endswith("_refit") else edit_reason
        if best is not None and not best["name"].endswith("_refit"):
            sc = _refit_one(best, mode_used, analysis, original_path, job_dir)
            if sc is not None:
                rep = sc.pop("_refit_rep", {})
                scored.append(sc)
                best = sc
                refit_info = {"applied": True, **rep, "candidate": sc["name"]}
                if not selection_reason.endswith("+color_refit"):
                    selection_reason = f"{selection_reason}+color_refit"
        from app.transform_journal import TransformJournal  # noqa: PLC0415

        if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[:, :, 3].astype(np.float32)[:, :, None] / 255.0
            journal_source_rgb = np.clip(
                rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255,
            ).astype(np.uint8)
            journal_required = {"alpha_fidelity"}
        else:
            journal_source_rgb = np.asarray(image.convert("RGB"))
            journal_required = set()
        if analysis.get("has_gradient"):
            journal_required.add("gradient_fidelity")
        journal = TransformJournal(
            Path(best["svg_path"]), journal_source_rgb,
            image_class=_JOURNAL_IMAGE_CLASS.get(mode_used, "clean_logo"),
            required_metrics=journal_required,
        )
        selected_candidate_history = best.get("candidate_transform_journal")

        _accepted, restore_rep, restore_stage = journal.run_in_place(
            "restore_source_dimensions", Path(best["svg_path"]),
            lambda path: (_restore_source_dimensions(path, analysis) or {"attempted": True}),
        )
        refit_info = {**refit_info, "restore_source_dimensions": {
            **(restore_rep or {}), "journal_status": restore_stage["status"],
            "journal_reasons": restore_stage["reason_codes"],
        }}

        pre_bnd_best = best
        bnd_candidate, bnd_info = _apply_boundary_refit(
            pre_bnd_best, mode_used, analysis, original_path, job_dir, scored
        )
        if bnd_candidate is not pre_bnd_best and Path(bnd_candidate["svg_path"]) != Path(pre_bnd_best["svg_path"]):
            accepted_path, bnd_stage = journal.consider_candidate(
                "boundary_refit", Path(pre_bnd_best["svg_path"]),
                Path(bnd_candidate["svg_path"]), transform_report=bnd_info,
            )
            if accepted_path == Path(bnd_candidate["svg_path"]):
                best = bnd_candidate
                selection_reason = f"{selection_reason}+boundary_refit"
            else:
                best = pre_bnd_best
                bnd_info = {**bnd_info, "applied": False,
                            "journal_status": bnd_stage["status"],
                            "journal_reasons": bnd_stage["reason_codes"]}
        else:
            best = pre_bnd_best
            journal.record_noop(
                "boundary_refit", Path(pre_bnd_best["svg_path"]),
                reason_codes=["boundary_refit_not_applied"], transform_report=bnd_info,
            )
        refit_info = {**refit_info, "boundary": bnd_info}
        if edge_cleanup and best is not None:
            from app.edge_cleanup import apply_edge_cleanup  # noqa: PLC0415

            src_svg = Path(best["svg_path"])
            ec_dst = job_dir / f"{src_svg.stem}_edgeclean.svg"
            try:
                ec_rep = apply_edge_cleanup(src_svg, original_path, ec_dst)
            except Exception as e:  # noqa: BLE001
                logger.debug("edge_cleanup atlandı: %s", e)
                ec_rep = {"applied": False, "error": str(e)}
            if ec_rep.get("applied"):
                cleaned = {**best, "svg_path": ec_dst}
                rescored = score_candidate(cleaned, original_path, analysis, mode_used)
                edge_candidate = rescored if rescored is not None else cleaned
                accepted_path, ec_stage = journal.consider_candidate(
                    "edge_cleanup", src_svg, ec_dst, transform_report=ec_rep,
                )
                if accepted_path == ec_dst:
                    best = edge_candidate
                    selection_reason = f"{selection_reason}+edge_cleanup"
                else:
                    ec_rep = {**ec_rep, "applied": False,
                              "journal_status": ec_stage["status"],
                              "journal_reasons": ec_stage["reason_codes"]}
            else:
                journal.record_noop(
                    "edge_cleanup", src_svg, reason_codes=["edge_cleanup_not_applied"],
                    transform_report=ec_rep,
                )
            refit_info = {**refit_info, "edge_cleanup": ec_rep}
        if best is not None:
            from app.component_align import apply_component_align  # noqa: PLC0415

            ca_src = Path(best["svg_path"])
            ca_dst = job_dir / f"{ca_src.stem}_align.svg"
            try:
                ca_rep = apply_component_align(ca_src, original_path, ca_dst)
            except Exception as e:  # noqa: BLE001
                logger.debug("component_align atlandı: %s", e)
                ca_rep = {"applied": False, "error": str(e)}
            if ca_rep.get("applied"):
                aligned = {**best, "svg_path": ca_dst}
                rescored = score_candidate(aligned, original_path, analysis, mode_used)
                align_candidate = rescored if rescored is not None else aligned
                accepted_path, ca_stage = journal.consider_candidate(
                    "component_align", ca_src, ca_dst, transform_report=ca_rep,
                )
                if accepted_path == ca_dst:
                    best = align_candidate
                    selection_reason = f"{selection_reason}+component_align"
                else:
                    ca_rep = {**ca_rep, "applied": False,
                              "journal_status": ca_stage["status"],
                              "journal_reasons": ca_stage["reason_codes"]}
            else:
                skip_code = "viewbox_missing_precondition" if ca_rep.get("reason") == "viewBox_yok" else "component_align_not_applied"
                journal.record_noop(
                    "component_align", ca_src, reason_codes=[skip_code], transform_report=ca_rep,
                )
            refit_info = {**refit_info, "component_align": ca_rep}
        refine_cache = None
        src_rgb_arr = journal_source_rgb if best is not None else None
        if best is not None and os.environ.get("VEKTORYUM_REFINE_CACHE", "on").strip().lower() not in {"off", "0", "false"}:
            try:
                from app.refine_cache import RefinementCache  # noqa: PLC0415

                refine_cache = RefinementCache(src_rgb_arr)
            except Exception as e:  # noqa: BLE001
                logger.debug("refine_cache kurulamadı: %s", e)
                refine_cache = None
        _render_fn = refine_cache.render if refine_cache is not None else None

        if best is not None and os.environ.get("VEKTORYUM_COUNTER_MERGE", "on").strip().lower() not in {"off", "0", "false"}:
            try:
                from app.counter_merge import merge_counters  # noqa: PLC0415
                from app.fidelity import render_svg_to_rgb  # noqa: PLC0415

                cm_w = int(analysis.get("width", 0) or 0)
                cm_h = int(analysis.get("height", 0) or 0)
                if cm_w > 0 and cm_h > 0:
                    _cm_ok, cm_rep, cm_stage = journal.run_in_place(
                        "counter_merge", Path(best["svg_path"]),
                        lambda path: merge_counters(
                            path, cm_w, cm_h, _render_fn or render_svg_to_rgb,
                            source_rgb=src_rgb_arr, cache=refine_cache,
                        ),
                    )
                    refit_info = {**refit_info, "counter_merge": {
                        **(cm_rep or {}), "journal_status": cm_stage["status"],
                        "journal_reasons": cm_stage["reason_codes"],
                    }}
            except Exception as e:  # noqa: BLE001
                logger.debug("counter_merge atlandı: %s", e)
                refit_info = {**refit_info, "counter_merge": {"status": "failed", "error": str(e)}}
        if best is not None and os.environ.get("VEKTORYUM_LOCAL_REFINE", "on").strip().lower() not in {"off", "0", "false"}:
            try:
                from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
                from app.local_refine import refine_critical_components  # noqa: PLC0415

                lr_w = int(analysis.get("width", 0) or 0)
                lr_h = int(analysis.get("height", 0) or 0)
                if lr_w > 0 and lr_h > 0:
                    _lr_ok, lr_rep, lr_stage = journal.run_in_place(
                        "local_refine", Path(best["svg_path"]),
                        lambda path: refine_critical_components(
                            path, src_rgb_arr, lr_w, lr_h,
                            _render_fn or render_svg_to_rgb, cache=refine_cache,
                        ),
                    )
                    refit_info = {**refit_info, "local_refine": {
                        **(lr_rep or {}), "journal_status": lr_stage["status"],
                        "journal_reasons": lr_stage["reason_codes"],
                    }}
            except Exception as e:  # noqa: BLE001
                logger.debug("local_refine atlandı: %s", e)
                refit_info = {**refit_info, "local_refine": {"status": "failed", "error": str(e)}}
        if best is not None and os.environ.get("VEKTORYUM_RENDER_REFINE", "on").strip().lower() not in {"off", "0", "false"}:
            try:
                from app.fidelity import render_svg_to_rgb  # noqa: PLC0415
                from app.local_refine import refine_error_regions  # noqa: PLC0415

                er_w = int(analysis.get("width", 0) or 0)
                er_h = int(analysis.get("height", 0) or 0)
                if er_w > 0 and er_h > 0:
                    _er_ok, er_rep, er_stage = journal.run_in_place(
                        "error_refine", Path(best["svg_path"]),
                        lambda path: refine_error_regions(
                            path, src_rgb_arr, er_w, er_h,
                            _render_fn or render_svg_to_rgb, cache=refine_cache,
                        ),
                    )
                    refit_info = {**refit_info, "error_refine": {
                        **(er_rep or {}), "journal_status": er_stage["status"],
                        "journal_reasons": er_stage["reason_codes"],
                    }}
            except Exception as e:  # noqa: BLE001
                logger.debug("error_refine atlandı: %s", e)
                refit_info = {**refit_info, "error_refine": {"status": "failed", "error": str(e)}}
        if best is not None and os.environ.get("VEKTORYUM_CUSP_REFINE", "on").strip().lower() not in {"off", "0", "false"}:
            try:
                from app.cusp_refine import refine_cusp_regions  # noqa: PLC0415
                from app.fidelity import render_svg_to_rgb  # noqa: PLC0415

                cr_w = int(analysis.get("width", 0) or 0)
                cr_h = int(analysis.get("height", 0) or 0)
                if cr_w > 0 and cr_h > 0:
                    def _cusp_transform(path: Path) -> dict[str, Any]:
                        report = refine_cusp_regions(
                            path, src_rgb_arr, cr_w, cr_h,
                            _render_fn or render_svg_to_rgb, cache=refine_cache,
                        )
                        return {k: v for k, v in report.items() if k != "candidates"} | {"candidate_count": len(report.get("candidates", []))}

                    _cu_ok, cu_rep, cu_stage = journal.run_in_place(
                        "cusp_refine", Path(best["svg_path"]), _cusp_transform,
                    )
                    refit_info = {**refit_info, "cusp_refine": {
                        **(cu_rep or {}), "journal_status": cu_stage["status"],
                        "journal_reasons": cu_stage["reason_codes"],
                    }}
            except Exception as e:  # noqa: BLE001
                logger.debug("cusp_refine atlandı: %s", e)
                refit_info = {**refit_info, "cusp_refine": {"status": "failed", "error": str(e)}}
        if refine_cache is not None:
            refit_info = {**refit_info, "refine_cache": refine_cache.stats()}
            refine_cache.close()
        from app.transform_journal import merge_journal_reports  # noqa: PLC0415

        transform_journal = merge_journal_reports(selected_candidate_history, journal.to_dict())

        try:
            final_scored = score_candidate(best, original_path, analysis, mode_used)
        except Exception as e:  # noqa: BLE001
            logger.warning("Kesin final SVG yeniden skorlanamadı: %s", e)
            final_scored = None
            final_rescore_error = type(e).__name__
        else:
            final_rescore_error = None
        if final_scored is not None and final_scored.get("rendered_ok"):
            best = final_scored
            refit_info = {**refit_info, "final_rescore": {
                "status": "measured", "fidelity_score": best.get("fidelity_score"),
                "svg_sha256": journal.final_accepted_sha256,
            }}
        else:
            best = {
                **best, "rendered_ok": False, "fidelity_score": None,
                "final_rescore_error": final_rescore_error or "measurement_unavailable",
            }
            refit_info = {**refit_info, "final_rescore": {
                "status": "unmeasured", "error": best["final_rescore_error"],
                "svg_sha256": journal.final_accepted_sha256,
            }}

    structure_report = None
    if (
        best is not None
        and mode_used != "photo_poster"
        and (analysis.get("background") or {}).get("is_uniform_background")
    ):
        structure_report = score_structure_integrity(best["svg_path"], original_path)

    for transaction_tmp in Path(job_dir).glob(".*.candidate.svg"):
        try:
            transaction_tmp.unlink(missing_ok=True)
        except OSError:
            logger.warning("Transform geçici dosyası temizlenemedi: %s", transaction_tmp)

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
        "transform_journal": transform_journal,
        "structure_report": structure_report,
    }
