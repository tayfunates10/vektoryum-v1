"""Public pipeline entry point with optional fail-isolated shadow reports.

The heavy production implementation remains in :mod:`app.pipeline`. This facade keeps
its call signature and exception contract, then applies the telemetry and canonical
candidate gates. Both feature flags default to off, so callers receive the original
result object and production winner unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from app.pipeline import WorkerFailure
from app.pipeline import run_pipeline as _run_pipeline_core
from app.pipeline_canonical_report import maybe_attach_canonical_svg_candidate
from app.production_export_integration import register_pipeline_canonical_report
from app.shadow_runtime import maybe_attach_shadow_telemetry


def _audit_path(job_dir: Path) -> Path | None:
    """Resolve optional per-job audit output without enabling persistence by default."""
    raw = os.environ.get("VEKTORYUM_SHADOW_AUDIT", "").strip()
    if not raw:
        return None
    if raw.lower() in {"1", "true", "yes", "on", "job"}:
        return Path(job_dir) / "shadow_telemetry.jsonl"
    return Path(raw)


def run_pipeline(
    image: Image.Image,
    original_path: Path,
    trace_mode: str,
    job_dir: Path,
    refine: bool = True,
    edge_cleanup: bool = True,
) -> dict[str, Any]:
    """Run production and optionally attach non-authoritative shadow reports.

    The production winner, SVG path and all existing result fields are produced by the
    unchanged core implementation. Shadow and canonical candidate failures are isolated
    and never replace the winner or escape into the vectorization request. A ready
    canonical report is registered only for the matching job's one-shot export call.
    """
    result = _run_pipeline_core(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=refine,
        edge_cleanup=edge_cleanup,
    )
    result = maybe_attach_shadow_telemetry(
        result,
        audit_path=_audit_path(Path(job_dir)),
    )
    result = maybe_attach_canonical_svg_candidate(result, image)
    register_pipeline_canonical_report(Path(job_dir), result)
    return result


__all__ = ["WorkerFailure", "run_pipeline"]
