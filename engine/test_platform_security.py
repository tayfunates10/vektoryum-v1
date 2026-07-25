from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.platform_security import install_platform_security


def _app() -> FastAPI:
    app = FastAPI()
    install_platform_security(app)
    install_platform_security(app)  # idempotency contract

    @app.get("/")
    async def index():
        return {"ok": True}

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/bad-request")
    async def bad_request():
        raise HTTPException(status_code=400, detail={"code": "invalid_input"})

    @app.get("/http-server-error")
    async def http_server_error():
        raise HTTPException(status_code=500, detail="secret backend detail")

    @app.get("/unhandled")
    async def unhandled():
        raise RuntimeError("secret runtime detail")

    @app.get("/api/auth/me")
    async def me():
        return {"user": None}

    return app


def test_client_errors_preserve_public_detail() -> None:
    response = TestClient(_app()).get("/bad-request")
    assert response.status_code == 400
    assert response.json() == {"detail": {"code": "invalid_input"}}


def test_http_server_errors_are_sanitized() -> None:
    response = TestClient(_app(), raise_server_exceptions=False).get("/http-server-error")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "İşlem sunucuda tamamlanamadı.",
        "code": "internal_error",
    }
    assert "secret backend detail" not in response.text


def test_unhandled_errors_are_sanitized() -> None:
    response = TestClient(_app(), raise_server_exceptions=False).get("/unhandled")
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "secret runtime detail" not in response.text


def test_security_headers_and_sensitive_cache_policy() -> None:
    client = TestClient(_app())
    normal = client.get("/ok")
    assert normal.headers["x-content-type-options"] == "nosniff"
    assert normal.headers["x-frame-options"] == "DENY"
    assert normal.headers["referrer-policy"] == "no-referrer"
    assert normal.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert "content-security-policy" not in normal.headers
    assert "cache-control" not in normal.headers

    platform_page = client.get("/")
    assert "frame-ancestors 'none'" in platform_page.headers["content-security-policy"]

    sensitive = client.get("/api/auth/me")
    assert sensitive.headers["cache-control"] == "no-store"
