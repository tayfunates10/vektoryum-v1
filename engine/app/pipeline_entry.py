"""Public pipeline entry point with optional FAZ 4 shadow telemetry.

The heavy production implementation remains in :mod:`app.pipeline`. This facade keeps
its call signature and exception contract, then applies the fail-isolated runtime gate.
The feature flag defaults to off, so callers receive the original result object.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from app.pipeline import WorkerFailure
from app.pipeline import run_pipeline as _run_pipeline_core
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
    """Run the production pipeline and optionally attach shadow-only telemetry.

    The production winner, SVG path and all existing result fields are produced by the
    unchanged core implementation. Shadow failures are isolated by
    ``maybe_attach_shadow_telemetry`` and never replace the winner.
    """
    result = _run_pipeline_core(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=refine,
        edge_cleanup=edge_cleanup,
    )
    return maybe_attach_shadow_telemetry(
        result,
        audit_path=_audit_path(Path(job_dir)),
    )


__all__ = ["WorkerFailure", "run_pipeline"]
