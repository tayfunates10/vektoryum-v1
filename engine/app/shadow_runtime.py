"""FAZ 4.3 runtime gate for shadow selector telemetry.

This module is intentionally side-effect free unless the feature flag is enabled.
It never replaces the production winner. Failures are converted into an explicit
telemetry status so the main vectorization pipeline remains available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.shadow_telemetry import append_shadow_telemetry, build_shadow_telemetry

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


def shadow_selector_enabled(env: dict[str, str] | None = None) -> bool:
    """Return whether shadow telemetry is enabled; default is fail-safe ``False``."""
    source = os.environ if env is None else env
    return str(source.get("VEKTORYUM_SHADOW_SELECTOR", "off")).strip().lower() in _TRUE_VALUES


def maybe_attach_shadow_telemetry(
    pipeline_result: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Return a shallow result copy with optional shadow telemetry.

    Production keys, winner objects and SVG paths are preserved exactly. When the
    flag is disabled, the original object is returned unchanged. When enabled,
    any telemetry error is isolated and reported instead of escaping into the
    vectorization request.
    """
    if not shadow_selector_enabled(env):
        return pipeline_result

    result = dict(pipeline_result)
    try:
        telemetry = build_shadow_telemetry(pipeline_result)
        result["shadow_telemetry"] = telemetry
        if audit_path is not None:
            append_shadow_telemetry(Path(audit_path), telemetry)
    except Exception as exc:  # noqa: BLE001 - shadow path must never fail production
        result["shadow_telemetry"] = {
            "schema_version": "faz4.3-shadow-runtime-v1",
            "status": "telemetry_error",
            "error_type": type(exc).__name__,
        }
    return result
