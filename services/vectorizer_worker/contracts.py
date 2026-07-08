"""Internal worker contracts for the new vectorization engine."""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from services.shared.schemas import QualityPolicy


class RasterAnalysis(BaseModel):
    width: int
    height: int
    estimated_colors: int
    has_gradient: bool
    has_text_like_regions: bool = False
    profile: str


class VectorizerConfig(BaseModel):
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    enable_hed: bool = True
    enable_super_resolution: bool = True
    max_processing_side: int = 2048


class VectorizerArtifacts(BaseModel):
    preprocessed: Path
    edge_map: Path | None = None
    segment_map: Path | None = None
    svg_path: Path
