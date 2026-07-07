"""Queue abstraction for the new API Gateway.

Production can swap this adapter with RabbitMQ/Celery without changing the
/v1/vectorize contract. The local adapter is intentionally file-backed so it is
safe in development and test environments without a broker.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from services.shared.schemas import VectorizeJob, VectorizeRequest, JobStatus


class JobQueue(Protocol):
    def enqueue(self, *, job_id: str, input_path: Path, request: VectorizeRequest) -> VectorizeJob:
        ...


class LocalFileQueue:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.queue_dir = root / "queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, *, job_id: str, input_path: Path, request: VectorizeRequest) -> VectorizeJob:
        job = VectorizeJob(
            job_id=job_id,
            status=JobStatus.queued,
            input_path=input_path,
            request=request,
            created_at=datetime.now(UTC).isoformat(),
        )
        payload = job.model_dump(mode="json")
        (self.queue_dir / f"{job_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return job
