"""Production ASGI entrypoint with fail-closed canonical SVG cutover.

The existing FastAPI application and routes remain unchanged. The module-level
``run_pipeline`` and ``export_all`` references used by ``app.main.vectorize_image``
are replaced with feature-flag-aware facades. With canonical flags off (default),
legacy pipeline and export behavior is preserved.
"""
from __future__ import annotations

from pathlib import Path

from app import main as _main
from app.exporters import export_all as _legacy_export_all
from app.pipeline_entry import run_pipeline as _shadow_aware_run_pipeline
from app.production_export_integration import export_all_with_canonical


def _runtime_export_all(
    best_svg: Path,
    job_dir: Path,
    job_id: str,
    candidate_id: str | None = None,
    formats: tuple[str, ...] = ("svg", "pdf", "eps", "dxf", "png"),
    png_size: tuple[int, int] | None = None,
):
    return export_all_with_canonical(
        _legacy_export_all,
        best_svg=best_svg,
        job_dir=job_dir,
        job_id=job_id,
        candidate_id=candidate_id,
        formats=formats,
        png_size=png_size,
    )


# ``vectorize_image`` resolves these globals from app.main at request time, so
# assignment is sufficient and avoids copying or redefining any route.
_main.run_pipeline = _shadow_aware_run_pipeline
_main.export_all = _runtime_export_all

app = _main.app
