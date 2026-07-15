from __future__ import annotations

from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.platform_identity import PlatformIdentity, REQUEST_COOKIE, REQUEST_HEADER, SESSION_COOKIE
from app.platform_request_compat import install_request_compat


def _fake_main(tmp_path: Path) -> ModuleType:
    module = ModuleType("request_compat_main")
    module.app = FastAPI()
    module.DATA_ROOT = tmp_path
    module.USERS_FILE = tmp_path / "users.json"
    module.SESSIONS = {}
    module.ALLOWED_MODES = ["auto"]
    module._hash_password = lambda value: "hashed:" + value
    module._verify_password = lambda value, encoded: encoded == "hashed:" + value
    module._safe_user = lambda user: {
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "role": user.get("role", "user"),
    }
    module.__file__ = str(tmp_path / "app" / "main.py")
    return module


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("VEKTORYUM_LOGIN_DB", str(tmp_path / "login.sqlite3"))
    monkeypatch.setenv("VEKTORYUM_COOKIE_SECURE", "0")
    module = _fake_main(tmp_path)
    identity = PlatformIdentity(module).install()

    async def protected():
        return {"ok": True}

    module.app.add_api_route("/api/vectorize", protected, methods=["POST"])
    install_request_compat(module.app)
    client = TestClient(module.app, base_url="https://testserver")
    response = client.post(
        "/api/auth/register",
        json={"name": "Test", "email": "user@example.com", "password": "password8"},
    )
    assert response.status_code == 200
    assert identity.state.resolve(client.cookies.get(SESSION_COOKIE)) is not None
    return client


def test_same_origin_browser_fallback_preserves_verified_request_cookie(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/vectorize", headers={"Origin": "https://testserver"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_cross_origin_request_stays_fail_closed(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/vectorize", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_scheme_mismatch_stays_fail_closed(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/vectorize", headers={"Origin": "http://testserver"})
    assert response.status_code == 403


def test_forwarded_https_same_origin_is_allowed(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post(
        "/api/vectorize",
        headers={"Origin": "https://testserver", "X-Forwarded-Proto": "https"},
    )
    assert response.status_code == 200


def test_missing_origin_stays_fail_closed(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post("/api/vectorize")
    assert response.status_code == 403


def test_explicit_bad_header_cannot_be_replaced(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    response = client.post(
        "/api/vectorize",
        headers={"Origin": "https://testserver", REQUEST_HEADER: "invalid"},
    )
    assert response.status_code == 403
    assert client.cookies.get(REQUEST_COOKIE)
