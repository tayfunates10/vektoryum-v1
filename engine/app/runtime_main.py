"""Production ASGI entrypoint with fail-closed canonical SVG cutover."""
from __future__ import annotations

from pathlib import Path

from app import main as _main
from app.exporters import export_all as _legacy_export_all
from app.pipeline_entry import run_pipeline as _shadow_aware_run_pipeline
from app.platform_frontend import install_platform_frontend
from app.platform_identity import install_platform_identity
from app.platform_operations import install_platform_operations
from app.platform_request_compat import install_request_compat
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


_main.run_pipeline = _shadow_aware_run_pipeline
_main.export_all = _runtime_export_all
platform_identity = install_platform_identity(_main)
install_platform_frontend(_main)
install_request_compat(_main.app)
platform_operations = install_platform_operations(_main.app)
app = _main.app
