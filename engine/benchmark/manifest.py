"""Benchmark v1 manifest and report contracts.

The first benchmark layer is intentionally deterministic and dependency-light so it
can run in pull-request CI. Large image execution is handled by a separate scheduled
workflow in later phases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ALLOWED_CATEGORIES = {
    "logos",
    "seals",
    "technical",
    "signatures",
    "gradients",
    "low_resolution",
    "transparent",
    "multilingual",
}
REQUIRED_METRICS = {
    "fidelity",
    "ssim",
    "edge_f1",
    "alpha_iou",
    "delta_e00",
    "path_count",
    "svg_bytes",
    "render_ms",
    "peak_rss_mb",
}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    source_path: str
    license_id: str
    source_sha256: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if self.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported benchmark category: {self.category}")
        if not self.source_path.strip():
            raise ValueError("source_path is required")
        if not self.license_id.strip():
            raise ValueError("license_id is required")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a 64-character SHA-256 hex digest")
        int(self.source_sha256, 16)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    engine_version: str
    metrics: dict[str, float | int | None]
    artifact_sha256: str | None = None
    failure: str | None = None

    def validate(self) -> None:
        if not self.case_id.strip() or not self.engine_version.strip():
            raise ValueError("case_id and engine_version are required")
        missing = REQUIRED_METRICS.difference(self.metrics)
        if missing:
            raise ValueError(f"missing benchmark metrics: {sorted(missing)}")
        if self.artifact_sha256 is not None:
            if len(self.artifact_sha256) != 64:
                raise ValueError("artifact_sha256 must be a 64-character SHA-256 hex digest")
            int(self.artifact_sha256, 16)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_manifest(cases: Iterable[BenchmarkCase], *, root: Path | None = None) -> list[BenchmarkCase]:
    validated = list(cases)
    ids: set[str] = set()
    for case in validated:
        case.validate()
        if case.case_id in ids:
            raise ValueError(f"duplicate benchmark case_id: {case.case_id}")
        ids.add(case.case_id)
        if root is not None:
            path = (root / case.source_path).resolve()
            if root.resolve() not in path.parents and path != root.resolve():
                raise ValueError(f"source_path escapes benchmark root: {case.source_path}")
    return sorted(validated, key=lambda item: item.case_id)
