"""Fail-isolated canonical SVG candidate attachment for the public pipeline.

The feature is opt-in and never changes the production winner or SVG path. When
it is disabled, the exact pipeline result object is returned. When enabled, a
canonical candidate report is attached under ``canonical_svg_candidate`` only
if configuration parsing and the HG-2..HG-7 builder complete. Every failure is
converted into an explicit, non-promotable report instead of escaping into the
legacy production path.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from PIL import Image

from app.canonical_svg_candidate import (
    CanonicalSvgCandidateReport,
    build_canonical_svg_candidate,
)

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_SCHEMA_VERSION = "canonical-pipeline-report-v1"


@dataclass(frozen=True)
class PipelineCanonicalSvgReport:
    schema_version: str
    enabled: bool
    attempted: bool
    status: str
    candidate: CanonicalSvgCandidateReport | None
    document_sha256: str
    face_count: int
    palette_size: int
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.status == "ready"
            and self.candidate is not None
            and self.candidate.valid
            and self.candidate.document is not None
            and self.candidate.promotion is not None
            and self.candidate.promotion.ready
            and not self.errors
        )


def canonical_candidate_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get("VEKTORYUM_CANONICAL_CANDIDATE_ENABLED", "off")).strip().lower() in _TRUE_VALUES


def _bounded_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = str(env.get(name, str(default))).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _failure(status: str, *errors: str, enabled: bool = True, attempted: bool = False) -> PipelineCanonicalSvgReport:
    return PipelineCanonicalSvgReport(
        schema_version=_SCHEMA_VERSION,
        enabled=enabled,
        attempted=attempted,
        status=status,
        candidate=None,
        document_sha256="",
        face_count=0,
        palette_size=0,
        errors=tuple(errors),
    )


def build_pipeline_canonical_svg_report(
    image: Image.Image,
    *,
    env: Mapping[str, str] | None = None,
) -> PipelineCanonicalSvgReport:
    """Build an auditable candidate report without changing production output."""
    source = os.environ if env is None else env
    if not canonical_candidate_enabled(source):
        return _failure("disabled", enabled=False, attempted=False)

    try:
        max_colors = _bounded_int(
            source, "VEKTORYUM_CANONICAL_CANDIDATE_MAX_COLORS", 32, 2, 64
        )
        repeat_runs = _bounded_int(
            source, "VEKTORYUM_CANONICAL_CANDIDATE_REPEAT_RUNS", 3, 3, 5
        )
        max_pixels = _bounded_int(
            source, "VEKTORYUM_CANONICAL_CANDIDATE_MAX_PIXELS", 16_000_000, 1, 64_000_000
        )
    except ValueError as exc:
        return _failure("configuration_error", str(exc), attempted=False)

    try:
        candidate = build_canonical_svg_candidate(
            image,
            max_colors=max_colors,
            repeat_runs=repeat_runs,
            max_pixels=max_pixels,
        )
    except Exception as exc:  # noqa: BLE001 - canonical path must not fail production
        return _failure(
            "builder_error",
            f"canonical candidate builder failed: {type(exc).__name__}",
            attempted=True,
        )

    if (
        not candidate.valid
        or candidate.document is None
        or candidate.promotion is None
        or not candidate.promotion.ready
        or candidate.errors
    ):
        return PipelineCanonicalSvgReport(
            schema_version=_SCHEMA_VERSION,
            enabled=True,
            attempted=True,
            status="invalid",
            candidate=candidate,
            document_sha256="",
            face_count=0,
            palette_size=int(candidate.palette_size),
            errors=tuple(candidate.errors or ("canonical candidate is not promotion-ready",)),
        )

    return PipelineCanonicalSvgReport(
        schema_version=_SCHEMA_VERSION,
        enabled=True,
        attempted=True,
        status="ready",
        candidate=candidate,
        document_sha256=candidate.document.document_sha256,
        face_count=candidate.document.face_count,
        palette_size=int(candidate.palette_size),
        errors=(),
    )


def maybe_attach_canonical_svg_candidate(
    pipeline_result: dict[str, Any],
    image: Image.Image,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Attach an opt-in report while preserving all production values exactly."""
    source = os.environ if env is None else env
    if not canonical_candidate_enabled(source):
        return pipeline_result

    result = dict(pipeline_result)
    try:
        report = build_pipeline_canonical_svg_report(image, env=source)
    except Exception as exc:  # noqa: BLE001 - defense-in-depth isolation
        report = _failure(
            "attachment_error",
            f"canonical report attachment failed: {type(exc).__name__}",
            attempted=True,
        )
    result["canonical_svg_candidate"] = report
    return result


__all__ = [
    "PipelineCanonicalSvgReport",
    "build_pipeline_canonical_svg_report",
    "canonical_candidate_enabled",
    "maybe_attach_canonical_svg_candidate",
]
