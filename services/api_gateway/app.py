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
from services.shared.schemas import OutputFormat, VectorizeProfile, VectorizeRequest

DATA_ROOT = Path(os.getenv("VEKTORYUM_V2_DATA_ROOT", "/tmp/vektoryum_v2"))
UPLOAD_ROOT = DATA_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
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
