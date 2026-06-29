"""Vektoryum API - FastAPI giriş noktası.

Akış:
1. Analiz (analyzer)
2. Mod seçimi + uyarılar
3. Profil bazlı ön işleme (preprocess)
4. Çoklu aday üretimi (vector_engines)
5. Geometri temizleme (geometry_cleanup)
6. Skorlama (scoring)
7. Profil bazlı en iyi aday seçimi
8. Export: SVG / PDF / EPS / DXF (exporters)
9. Kalite raporu (quality)

Dayanıklılık: CairoSVG/Inkscape/Potrace/AutoTrace yoksa sistem çökmez; ilgili
adım atlanır ve hata raporlanır.
"""

from __future__ import annotations

import io
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from app.analyzer import analyze_image_from_mem
from app.exporters import export_all
from app.geometry_cleanup import cleanup_svg_geometry, consolidate_svg_palette
from app.preprocess import preprocess_for_mode
from app.quality import basic_svg_quality_check
from app.scoring import score_vector_candidate
from app.shape_fitting import regularize_svg_geometry
from app.vector_engines import build_vector_candidates, get_autotrace_path, run_candidate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vektoryum API", version="2.0.0")

ALLOWED_MODES = [
    "auto", "geometric_logo", "minimal_ai", "logo_color",
    "flat_logo", "single_color", "lineart", "centerline", "photo_poster",
]

# Geriye dönük uyumluluk için ikinci ad (README'de geçer)
ALLOWED_TRACE_MODES = ALLOWED_MODES

JOBS_ROOT = Path(tempfile.gettempdir()) / "vector_jobs"

# Mod bazlı son palet üst sınırı (VTracer'ın kenar ara-tonlarını temizlemek için)
_PALETTE_CAP = {
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
_FLAT_PALETTE_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color", "lineart", "centerline"}
_CANONICAL_BWR = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]

# Geometrik şekil oturtma (line+arc fitting) uygulanan profiller
_REGULARIZE_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color"}

_MEDIA_TYPES = {
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "eps": "application/postscript",
    "dxf": "image/vnd.dxf",
}


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


# ---------------------------------------------------------------------------
# Mod uyarıları
# ---------------------------------------------------------------------------
def _compute_mode_warning(trace_mode: str, mode_used: str, analysis: dict[str, Any]) -> str | None:
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
def _select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
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
# /api/vectorize
# ---------------------------------------------------------------------------
@app.post("/api/vectorize", summary="Raster görseli vektöre dönüştürür")
async def vectorize_image(
    file: UploadFile = File(...),
    trace_mode: str = Form("auto"),
):
    if trace_mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Geçersiz trace_mode. İzin verilenler: {ALLOWED_MODES}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya türü.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:  # noqa: BLE001
        logger.error("Görsel okuma hatası: %s", e)
        raise HTTPException(status_code=400, detail="Görsel dosyası bozuk veya okunamıyor.")

    # 1. Analiz
    analysis = analyze_image_from_mem(image)
    mode_used = analysis["recommended_mode"] if trace_mode == "auto" else trace_mode
    mode_warning = _compute_mode_warning(trace_mode, mode_used, analysis)

    # iş klasörü
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    original_path = job_dir / f"original{suffix}"
    original_path.write_bytes(contents)

    # 2. Ön işleme
    try:
        preprocessed_path, preprocess_report = preprocess_for_mode(original_path, mode_used, job_dir, analysis=analysis)
    except Exception as e:  # noqa: BLE001
        logger.error("Ön işleme hatası: %s", e)
        raise HTTPException(status_code=500, detail=f"Ön işleme başarısız: {e}")

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
            cap = _PALETTE_CAP.get(mode_used)
            if cap:
                canonical = _CANONICAL_BWR if mode_used in _FLAT_PALETTE_MODES else None
                consolidate_svg_palette(svg_path, max_colors=cap, canonical=canonical)
            # geometrik idealleştirme: düz çizgi + tam dairesel yay oturtma
            if mode_used in _REGULARIZE_MODES:
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

    if not scored:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Hiçbir vektör adayı üretilemedi.",
                "job_id": job_id,
                "mode_used": mode_used,
                "candidate_report": {
                    "candidates": [
                        {"name": r["name"], "success": False, "error": r.get("error")}
                        for r in results
                    ],
                },
            },
        )

    # 6. + 7. Seçim
    best, raw_best, selection_reason = _select_best(scored, mode_used)

    # 8. Export
    best_geo = best.get("cleanup_report", {}).get("report", {})
    outputs, output_errors = export_all(
        best_svg=best["svg_path"],
        job_dir=job_dir,
        job_id=job_id,
        candidate_id=f"{mode_used}:{best['name']}",
    )

    # 9. Kalite raporu
    quality_report = basic_svg_quality_check(
        score_details=best.get("score_details", {}),
        mode=mode_used,
        geometry_report=best_geo,
        total_score=best["total_score"],
    )

    download_links = {fmt: f"/api/download/{job_id}/{fmt}" for fmt in ("svg", "pdf", "eps", "dxf")}

    final_report = {
        "job_id": job_id,
        "mode_used": mode_used,
        "mode_warning": mode_warning,
        "analysis": analysis,
        "preprocess": {"steps": preprocess_report.get("steps", []), "palette": preprocess_report.get("palette", [])},
        "candidate_report": {
            "best_candidate": best["name"],
            "best_score": best["total_score"],
            "raw_best_candidate": raw_best["name"],
            "raw_best_score": raw_best["total_score"],
            "selection_reason": selection_reason,
            "candidates": [
                {
                    "name": (c.get("name")),
                    "success": c.get("success", False),
                    "error": c.get("error"),
                    "engine": c.get("engine"),
                    "total_score": c.get("total_score"),
                    "color_score": c.get("color_score"),
                    "edge_score": c.get("edge_score"),
                    "detail_score": c.get("detail_score"),
                    "path_score": c.get("path_score"),
                    "warning_score": c.get("warning_score"),
                    "straight_edge_score": c.get("straight_edge_score"),
                    "corner_cleanliness_score": c.get("corner_cleanliness_score"),
                    "axis_alignment_score": c.get("axis_alignment_score"),
                    "geometry_score": c.get("geometry_score"),
                    "rendered_ok": c.get("rendered_ok"),
                    "details": c.get("score_details"),
                }
                # başarısız adaylar da raporlanır
                for c in _merge_for_report(scored, results)
            ],
        },
        "quality_report": quality_report,
        "outputs": {fmt: Path(p).name for fmt, p in outputs.items()},
        "output_errors": output_errors,
        "download_links": download_links,
    }

    return JSONResponse(content=final_report)


def _merge_for_report(scored: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skorlanan adaylar + başarısız adayları tek listede birleştirir."""
    scored_by_name = {c["name"]: c for c in scored}
    merged = []
    for r in results:
        if r["name"] in scored_by_name:
            merged.append(scored_by_name[r["name"]])
        else:
            merged.append(r)
    return merged


# ---------------------------------------------------------------------------
# /api/download/{job_id}/{file_type}
# ---------------------------------------------------------------------------
@app.get("/api/download/{job_id}/{file_type}", summary="Üretilen vektör dosyasını indir")
async def download_file(job_id: str, file_type: str):
    if file_type not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı.")

    # job_id güvenlik: sadece hex
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Geçersiz job_id.")

    file_path = _job_dir(job_id) / f"{job_id}.{file_type}"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{file_type}' dosyası bu iş için üretilmedi (export başarısız olmuş olabilir).",
        )

    return FileResponse(
        file_path,
        media_type=_MEDIA_TYPES[file_type],
        filename=f"{job_id}.{file_type}",
    )


@app.get("/", summary="Sağlık kontrolü")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vektoryum-api", "modes": ALLOWED_MODES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
