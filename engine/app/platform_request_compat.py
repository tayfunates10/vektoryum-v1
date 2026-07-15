"""Same-origin browser compatibility for the PPC-2 request boundary.

Some production browsers retain the valid request cookie but fail to execute the
frontend fetch adapter before the first protected upload. This middleware does
not bypass CSRF validation: it only mirrors the request cookie into the expected
header when the immutable browser Origin exactly matches the effective request
scheme and Host. The identity middleware still verifies the token against the
active SQLite session.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from starlette.datastructures import MutableHeaders

from app.platform_identity import REQUEST_COOKIE, REQUEST_HEADER, SESSION_COOKIE

_PROTECTED_POST_PATHS = frozenset({"/api/vectorize", "/api/auth/logout"})


def _effective_scheme(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    if forwarded in {"http", "https"}:
        return forwarded
    scheme = request.url.scheme.strip().lower()
    return scheme if scheme in {"http", "https"} else ""


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin", "").strip()
    host = request.headers.get("host", "").strip().lower()
    if not origin or not host:
        return False
    parsed = urlsplit(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.scheme == _effective_scheme(request)
        and parsed.netloc.lower() == host
    )


def install_request_compat(app: FastAPI) -> None:
    if getattr(app.state, "request_compat_installed", False):
        return

    @app.middleware("http")
    async def same_origin_request_cookie_adapter(request: Request, call_next):
        if (
            request.method.upper() == "POST"
            and request.url.path in _PROTECTED_POST_PATHS
            and request.cookies.get(SESSION_COOKIE)
            and not request.headers.get(REQUEST_HEADER)
            and _same_origin(request)
        ):
            request_token = request.cookies.get(REQUEST_COOKIE)
            if request_token:
                MutableHeaders(scope=request.scope)[REQUEST_HEADER] = request_token
        return await call_next(request)

    app.state.request_compat_installed = True


__all__ = ["install_request_compat"]
