"""PPC-4 operations contract: modes, health, correlation and shutdown."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ALLOWED_SERVICE_MODES = frozenset({"beta", "live", "maintenance"})
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HEALTH_PATHS = frozenset({"/livez", "/readyz"})
logger = logging.getLogger("vektoryum.operations")


def service_mode() -> str:
    mode = os.environ.get("VEKTORYUM_SERVICE_MODE", "beta").strip().lower()
    if mode not in ALLOWED_SERVICE_MODES:
        raise RuntimeError(f"invalid VEKTORYUM_SERVICE_MODE: {mode}")
    return mode


@dataclass
class RuntimeState:
    started_at: float
    accepting_requests: bool = True
    active_requests: int = 0

    def readiness(self) -> tuple[bool, list[str]]:
        checks: list[str] = []
        if not self.accepting_requests:
            checks.append("shutdown_in_progress")
        if service_mode() == "maintenance":
            checks.append("maintenance_mode")
        return not checks, checks


def _structured(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":")))


def _remove_health_routes(app: FastAPI) -> None:
    app.router.routes[:] = [
        route for route in app.router.routes
        if getattr(route, "path", None) not in _HEALTH_PATHS
    ]


def install_platform_operations(app: FastAPI, *, revision: str | None = None) -> RuntimeState:
    if getattr(app.state, "platform_operations_installed", False):
        return app.state.platform_runtime_state
    service_mode()
    state = RuntimeState(started_at=time.time())
    app.state.platform_runtime_state = state
    app.state.platform_operations_installed = True
    source_revision = revision or os.environ.get("VEKTORYUM_SOURCE_REVISION", "unknown")
    _remove_health_routes(app)

    async def operations_boundary(request: Request, call_next: Callable):
        correlation_id = request.headers.get("X-Correlation-ID", "").strip() or uuid.uuid4().hex
        mode = service_mode()
        if not state.accepting_requests and request.url.path not in _HEALTH_PATHS:
            return JSONResponse({"status": "unavailable", "reason": "shutdown_in_progress", "correlation_id": correlation_id}, status_code=503, headers={"X-Correlation-ID": correlation_id})
        if mode == "maintenance" and request.method in _WRITE_METHODS and request.url.path not in _HEALTH_PATHS:
            return JSONResponse({"status": "maintenance", "correlation_id": correlation_id}, status_code=503, headers={"X-Correlation-ID": correlation_id})
        state.active_requests += 1
        started = time.monotonic()
        _structured("request_started", correlation_id=correlation_id, method=request.method, path=request.url.path, mode=mode)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            state.active_requests -= 1
            _structured("request_finished", correlation_id=correlation_id, method=request.method, path=request.url.path, duration_ms=round((time.monotonic() - started) * 1000, 3))

    if app.middleware_stack is None:
        app.middleware("http")(operations_boundary)
    else:
        _structured("middleware_registration_skipped", reason="application_already_started")

    async def livez():
        return {"status": "ok", "check": "liveness", "mode": service_mode(), "revision": source_revision}

    async def readyz():
        ready, checks = state.readiness()
        return JSONResponse({"status": "ready" if ready else "not_ready", "check": "readiness", "mode": service_mode(), "revision": source_revision, "reasons": checks, "active_requests": state.active_requests}, status_code=200 if ready else 503)

    async def shutdown() -> None:
        state.accepting_requests = False
        _structured("shutdown_started", active_requests=state.active_requests)

    app.add_api_route("/livez", livez, methods=["GET"], include_in_schema=False)
    app.add_api_route("/readyz", readyz, methods=["GET"], include_in_schema=False)
    app.add_event_handler("shutdown", shutdown)
    return state


__all__ = ["ALLOWED_SERVICE_MODES", "RuntimeState", "install_platform_operations", "service_mode"]
