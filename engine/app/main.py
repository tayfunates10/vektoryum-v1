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

from app.exporters import export_all
from app.pipeline import run_pipeline
from app.quality import basic_svg_quality_check

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

_MEDIA_TYPES = {
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "eps": "application/postscript",
    "dxf": "image/vnd.dxf",
}


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


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

    # iş klasörü
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    original_path = job_dir / f"original{suffix}"
    original_path.write_bytes(contents)

    # 1-7. Çekirdek pipeline (analiz → ön işleme → aday → temizleme → skor → seçim)
    try:
        pipe = run_pipeline(image, original_path, trace_mode, job_dir)
    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline hatası: %s", e)
        raise HTTPException(status_code=500, detail=f"İşlem başarısız: {e}")

    analysis = pipe["analysis"]
    mode_used = pipe["mode_used"]
    mode_warning = pipe["mode_warning"]
    preprocess_report = pipe["preprocess_report"]
    results = pipe["results"]
    scored = pipe["scored"]

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

    best = pipe["best"]
    raw_best = pipe["raw_best"]
    selection_reason = pipe["selection_reason"]

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
        fidelity_score=best.get("fidelity_score"),
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
                    "fidelity_score": c.get("fidelity_score"),
                    "details": c.get("score_details"),
                }
                # başarısız adaylar da raporlanır
                for c in _merge_for_report(scored, results)
            ],
        },
        "quality_report": quality_report,
        "refine_info": pipe.get("refine_info"),
        "outputs": {fmt: Path(p).name for fmt, p in outputs.items()},
        "output_errors": output_errors,
        "download_links": download_links,
    }

    return JSONResponse(content=final_report)


def _merge_for_report(scored: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skorlanan adaylar + başarısız adayları tek listede birleştirir.

    Refinement'ta üretilen adaylar ``results`` içinde olmayabilir; onları da
    sona ekleriz ki rapor (ve seçilen kazanan) eksik kalmasın.
    """
    scored_by_name = {c["name"]: c for c in scored}
    merged = []
    seen: set[str] = set()
    for r in results:
        seen.add(r["name"])
        merged.append(scored_by_name.get(r["name"], r))
    for c in scored:
        if c["name"] not in seen:
            merged.append(c)
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
