"""Production ASGI entrypoint with optional FAZ 4 shadow telemetry.

The existing FastAPI application and routes remain unchanged. Only the module-level
``run_pipeline`` reference used by ``app.main.vectorize_image`` is replaced with the
feature-flag-aware façade. With ``VEKTORYUM_SHADOW_SELECTOR=off`` (default), the
façade returns the core pipeline result unchanged.
"""
from __future__ import annotations

from app import main as _main
from app.pipeline_entry import run_pipeline as _shadow_aware_run_pipeline

# ``vectorize_image`` resolves this global from app.main at request time, so the
# assignment is sufficient and avoids copying/redefining any route.
_main.run_pipeline = _shadow_aware_run_pipeline

app = _main.app
