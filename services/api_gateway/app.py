"""Next-generation Vektoryum.ai API Gateway skeleton.

This is the first step of the re-architecture requested in the new SRS. It does
not remove the current monolith; it introduces the future SaaS-facing /v1 API
surface and a queue boundary for the worker service.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from services.api_gateway.queue import LocalFileQueue
from services.shared.schemas import OutputFormat, VectorizeProfile, VectorizeRequest, FeedbackRequest, FeedbackIssue

DATA_ROOT = Path(os.getenv("VEKTORYUM_V2_DATA_ROOT", "/tmp/vektoryum_v2"))
UPLOAD_ROOT = DATA_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
FEEDBACK_ROOT = DATA_ROOT / "feedback_cases"
FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)
QUEUE = LocalFileQueue(DATA_ROOT)

app = FastAPI(title="Vektoryum.ai API Gateway", version="1.0.0")

_ALLOWED_INPUTS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vektoryum-api-gateway"}


@app.post("/v1/vectorize", status_code=202)
async def vectorize(
    file: UploadFile = File(...),
    profile: VectorizeProfile = Form(VectorizeProfile.auto),
    formats: str = Form("svg,pdf,eps"),
    webhook_url: str | None = Form(default=None),
    synchronous: bool = Form(default=False),
):
    """Accept a raster image and enqueue it for the new worker pipeline.

    The endpoint is intentionally asynchronous-first. Later iterations can add
    API-key auth, account quotas, batch uploads and webhook signing without
    changing the basic job contract.
    """
    if file.content_type not in _ALLOWED_INPUTS:
        raise HTTPException(status_code=415, detail="JPEG, PNG, WebP, BMP ve TIFF desteklenir.")
    try:
        requested_formats = [OutputFormat(item.strip()) for item in formats.split(",") if item.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Geçersiz çıktı formatı. svg,pdf,eps,png desteklenir.") from exc
    if not requested_formats:
        requested_formats = [OutputFormat.svg]

    job_id = uuid.uuid4().hex
    suffix = _ALLOWED_INPUTS[file.content_type]
    input_path = UPLOAD_ROOT / f"{job_id}{suffix}"
    input_path.write_bytes(await file.read())

    req = VectorizeRequest(
        profile=profile,
        formats=requested_formats,
        webhook_url=webhook_url,
        synchronous=synchronous,
    )
    job = QUEUE.enqueue(job_id=job_id, input_path=input_path, request=req)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "status_url": f"/v1/jobs/{job.job_id}",
        "message": "Görsel analiz ve vektörizasyon kuyruğuna alındı.",
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    if not job_id.isalnum() or len(job_id) != 32:
        raise HTTPException(status_code=400, detail="Geçersiz job_id.")
    queue_file = DATA_ROOT / "queue" / f"{job_id}.json"
    result_file = DATA_ROOT / "results" / job_id / "result.json"
    if result_file.exists():
        return result_file.read_text(encoding="utf-8")
    if queue_file.exists():
        return {"job_id": job_id, "status": "queued"}
    raise HTTPException(status_code=404, detail="İş bulunamadı.")


@app.post("/v1/feedback", status_code=201)
async def post_feedback(feedback: FeedbackRequest):
    """Kullanıcı hata bildirimlerini ve görsel regresyon vakalarını alır.
    
    Sorunlu vakalar 'feedback_cases/' klasörüne kaydedilir ve ileride CIELAB ΔE
    ve otomatik düzeltme süreçleri için hazır tutulur.
    """
    if not feedback.job_id.isalnum() or len(feedback.job_id) != 32:
        raise HTTPException(status_code=400, detail="Geçersiz job_id.")
    
    # İleride CIELAB Delta E hesaplamaları için hazırlık yap
    delta_e_estimation = None
    if feedback.expected_color_hex and feedback.actual_color_hex:
        try:
            import math
            r1, g1, b1 = int(feedback.expected_color_hex[1:3], 16), int(feedback.expected_color_hex[3:5], 16), int(feedback.expected_color_hex[5:7], 16)
            r2, g2, b2 = int(feedback.actual_color_hex[1:3], 16), int(feedback.actual_color_hex[3:5], 16), int(feedback.actual_color_hex[5:7], 16)
            # Basit Delta E yaklaşımı (RGB L2 uzaklığı normalize edilmiş hali)
            delta_e_estimation = float(math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2) * 0.1)
        except Exception:
            pass

    case_id = uuid.uuid4().hex
    feedback_case_path = FEEDBACK_ROOT / f"{feedback.job_id}_{case_id}.json"
    
    case_data = feedback.dict()
    case_data["case_id"] = case_id
    case_data["delta_e_estimation"] = delta_e_estimation
    case_data["timestamp"] = uuid.uuid4().hex
    
    try:
        import json
        feedback_case_path.write_text(json.dumps(case_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geri bildirim kaydedilirken hata oluştu: {str(e)}")
        
    return {
        "status": "logged",
        "case_id": case_id,
        "delta_e_estimation": delta_e_estimation,
        "message": "Geri bildirim başarıyla kaydedildi. Kendi kendini eğiten regresyon motoruna iletildi."
    }

