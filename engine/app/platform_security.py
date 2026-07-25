"""Production HTTP error sanitization and response security headers."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from starlette.responses import Response

logger = logging.getLogger("vektoryum.security")

_GENERIC_ERROR = {
    "detail": "İşlem sunucuda tamamlanamadı.",
    "code": "internal_error",
}
_NO_STORE_PREFIXES = ("/api/auth", "/api/admin", "/admin")
_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)


def _set_default_header(response: Response, name: str, value: str) -> None:
    if name not in response.headers:
        response.headers[name] = value


def _apply_security_headers(response: Response, path: str) -> Response:
    _set_default_header(response, "X-Content-Type-Options", "nosniff")
    _set_default_header(response, "X-Frame-Options", "DENY")
    _set_default_header(response, "Referrer-Policy", "no-referrer")
    _set_default_header(
        response,
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    _set_default_header(response, "Content-Security-Policy", _CONTENT_SECURITY_POLICY)
    if path.startswith(_NO_STORE_PREFIXES):
        _set_default_header(response, "Cache-Control", "no-store")
    return response


def install_platform_security(app: FastAPI) -> None:
    """Install fail-closed HTTP error handling and response hardening once."""
    if getattr(app.state, "platform_security_installed", False):
        return

    @app.exception_handler(HTTPException)
    async def sanitized_http_exception(request: Request, exc: HTTPException):
        if exc.status_code < 500:
            return await http_exception_handler(request, exc)
        logger.error(
            "sanitized HTTP server error status=%s path=%s",
            exc.status_code,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_GENERIC_ERROR,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def sanitized_unhandled_exception(request: Request, exc: Exception):
        logger.error(
            "sanitized unhandled server error path=%s",
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(status_code=500, content=_GENERIC_ERROR)

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        response = await call_next(request)
        return _apply_security_headers(response, request.url.path)

    app.state.platform_security_installed = True


__all__ = ["install_platform_security"]
