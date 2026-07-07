"""Shared contracts for the next-generation Vektoryum.ai microservices.

These schemas deliberately model the public SaaS API separately from the legacy
monolith so the new architecture can evolve without breaking the existing app.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


class VectorizeProfile(str, Enum):
    auto = "auto"
    logo = "logo"
    icon = "icon"
    typography = "typography"
    line_art = "line_art"
    photo_poster = "photo_poster"


class OutputFormat(str, Enum):
    svg = "svg"
    pdf = "pdf"
    eps = "eps"
    png = "png"


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    needs_review = "needs_review"
    failed = "failed"


class QualityPolicy(BaseModel):
    """Hard targets used by the new worker before an output is accepted."""

    min_fidelity: float = Field(default=0.98, ge=0, le=1)
    min_edge_f1: float = Field(default=0.96, ge=0, le=1)
    max_delta_e: float = Field(default=2.8, ge=0)
    max_banding_ratio: float = Field(default=0.008, ge=0)
    max_anchor_points: int | None = Field(default=None, ge=1)
    require_clean_svg: bool = True


class VectorizeRequest(BaseModel):
    profile: VectorizeProfile = VectorizeProfile.auto
    formats: list[OutputFormat] = Field(default_factory=lambda: [OutputFormat.svg, OutputFormat.pdf, OutputFormat.eps])
    webhook_url: HttpUrl | None = None
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    synchronous: bool = False


class VectorizeJob(BaseModel):
    job_id: str
    status: JobStatus
    input_path: Path
    request: VectorizeRequest
    created_at: str


class QualityReport(BaseModel):
    status: Literal["production_ready", "needs_review", "failed"]
    fidelity: float | None = None
    edge_f1: float | None = None
    mean_delta_e: float | None = None
    banding_ratio: float | None = None
    anchor_points: int | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class VectorizeResult(BaseModel):
    job_id: str
    status: JobStatus
    outputs: dict[OutputFormat, Path] = Field(default_factory=dict)
    quality_report: QualityReport | None = None
    error: str | None = None
