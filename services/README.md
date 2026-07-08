# Vektoryum.ai v2 Microservice Skeleton

This folder starts the SRS-driven re-architecture without deleting the current
production app. The new design separates public API, queue boundary and worker
pipeline so we can replace the legacy vectorization path step by step.

## Services

```text
services/
  api_gateway/          FastAPI SaaS-facing /v1 API surface
  vectorizer_worker/    Python image-processing/vectorization worker
  shared/               Shared Pydantic contracts between services
```

## Target flow

1. `POST /v1/vectorize` receives JPG/PNG/WebP/BMP/TIFF and creates a queued job.
2. Queue adapter stores the job locally now; RabbitMQ/Celery can replace it later.
3. Worker runs the staged pipeline:
   - Bilateral filtering
   - Super-resolution hook
   - K-Means/SLIC planning hook
   - Canny baseline + HED hook
   - Marching Squares / Potrace / Bézier fitting hook
   - Clean SVG contract validation
4. Result artifacts are written under `VEKTORYUM_V2_DATA_ROOT/results/{job_id}`.

## Important boundary

The current monolith remains active while this v2 system is built. New code must
not call legacy internals directly; shared behavior should move behind contracts
in `services/shared`.

## Controlled rollout

The production `/api/vectorize` endpoint is intentionally unchanged. The
monolith exposes the v2 worker only through an authenticated canary endpoint:

```text
POST /api/vectorize-v2
```

This endpoint writes a normal job folder and `report.json`, so admin review,
download links and Hugging Face persistence can inspect v2 output without
sending regular users through the new engine. Set `VEKTORYUM_V2_CANARY=0` to
disable the canary gate completely.
