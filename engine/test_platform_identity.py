from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.platform_frontend import SCRIPT_PATH, install_platform_frontend
from app.platform_identity import (
    LoginStateStore,
    PlatformIdentity,
    REQUEST_COOKIE,
    REQUEST_HEADER,
    SESSION_COOKIE,
)


def _hash_password(password: str) -> str:
    return hashlib.sha256(("test:" + password).encode("utf-8")).hexdigest()


def _verify_password(password: str, encoded: str) -> bool:
    return _hash_password(password) == encoded


def _fake_main(tmp_path: Path, name: str = "fake_main") -> ModuleType:
    module = ModuleType(name)
    module.app = FastAPI()
    module.DATA_ROOT = tmp_path
    module.USERS_FILE = tmp_path / "users.json"
    module.SESSIONS = {}
    module.ALLOWED_MODES = ["auto"]
    module._hash_password = _hash_password
    module._verify_password = _verify_password
    module._safe_user = lambda user: {
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "role": user.get("role", "user"),
    }
    module.__file__ = str(tmp_path / "app" / "main.py")
    return module


def test_login_state_survives_reopen_and_supports_expiry_and_revoke(tmp_path) -> None:
    path = tmp_path / "login.sqlite3"
    first = LoginStateStore(path)
    session, request_token, expires_at = first.create_session(
        "user@example.com",
        ttl_seconds=100,
        now=1_000,
    )
    assert expires_at == 1_100
    assert first.resolve(session, now=1_050)["email"] == "user@example.com"
    assert first.verify_request_token(session, request_token, now=1_050)

    reopened = LoginStateStore(path)
    assert reopened.resolve(session, now=1_099)["email"] == "user@example.com"
    assert reopened.resolve(session, now=1_100) is None

    second_session, _request, _expires = reopened.create_session(
        "user@example.com",
        ttl_seconds=100,
        now=2_000,
    )
    assert reopened.revoke(second_session, now=2_010)
    assert reopened.resolve(second_session, now=2_011) is None


def test_request_token_rotation_invalidates_previous_value(tmp_path) -> None:
    store = LoginStateStore(tmp_path / "login.sqlite3")
    session, initial, _expires = store.create_session("u@example.com", ttl_seconds=100, now=10)
    rotated = store.rotate_request_token(session, now=11)
    assert rotated
    assert not store.verify_request_token(session, initial, now=12)
    assert store.verify_request_token(session, rotated, now=12)


def test_attempt_window_is_shared_and_resets(tmp_path) -> None:
    path = tmp_path / "login.sqlite3"
    first = LoginStateStore(path)
    assert first.consume_attempt("login", "client", limit=2, window_seconds=60, now=120).allowed
    assert first.consume_attempt("login", "client", limit=2, window_seconds=60, now=121).allowed
    third = LoginStateStore(path).consume_attempt(
        "login",
        "client",
        limit=2,
        window_seconds=60,
        now=122,
    )
    assert not third.allowed
    assert third.retry_after == 58
    first.reset_attempt("login", "client")
    assert first.consume_attempt("login", "client", limit=2, window_seconds=60, now=123).allowed


def test_administrator_bootstrap_has_no_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VEKTORYUM_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("VEKTORYUM_ADMIN_PASSWORD", raising=False)
    runtime = PlatformIdentity(_fake_main(tmp_path / "empty"))
    assert runtime.load_users() == {}

    monkeypatch.setenv("VEKTORYUM_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.delenv("VEKTORYUM_ADMIN_PASSWORD", raising=False)
    partial = PlatformIdentity(_fake_main(tmp_path / "partial"))
    with pytest.raises(RuntimeError):
        partial.load_users()

    monkeypatch.setenv("VEKTORYUM_ADMIN_PASSWORD", "long-production-password")
    configured = PlatformIdentity(_fake_main(tmp_path / "configured"))
    users = configured.load_users()
    assert users["admin@example.com"]["role"] == "admin"
    assert users["admin@example.com"]["password"] != "long-production-password"


def test_account_api_cookie_request_boundary_and_logout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VEKTORYUM_LOGIN_DB", str(tmp_path / "login.sqlite3"))
    monkeypatch.setenv("VEKTORYUM_COOKIE_SECURE", "0")
    module = _fake_main(tmp_path)
    runtime = PlatformIdentity(module).install()

    async def protected():
        return {"ok": True}

    module.app.add_api_route("/api/vectorize", protected, methods=["POST"])
    client = TestClient(module.app)

    response = client.post(
        "/api/auth/register",
        json={"name": "Test", "email": "user@example.com", "password": "password8"},
    )
    assert response.status_code == 200
    assert client.cookies.get(SESSION_COOKIE)
    request_token = client.cookies.get(REQUEST_COOKIE)
    assert request_token
    set_cookies = response.headers.get_list("set-cookie")
    session_header = next(item for item in set_cookies if item.startswith(SESSION_COOKIE + "="))
    request_header = next(item for item in set_cookies if item.startswith(REQUEST_COOKIE + "="))
    assert "HttpOnly" in session_header
    assert "SameSite=lax" in session_header
    assert "HttpOnly" not in request_header

    blocked = client.post("/api/vectorize")
    assert blocked.status_code == 403
    allowed = client.post("/api/vectorize", headers={REQUEST_HEADER: request_token})
    assert allowed.status_code == 200

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "user@example.com"
    rotated = client.cookies.get(REQUEST_COOKIE)
    assert rotated and rotated != request_token

    logout = client.post("/api/auth/logout", headers={REQUEST_HEADER: rotated})
    assert logout.status_code == 200
    assert client.get("/api/auth/me").json() == {"user": None}
    assert runtime.state.resolve(client.cookies.get(SESSION_COOKIE)) is None


def test_login_limit_is_generic_and_restart_shared(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VEKTORYUM_LOGIN_DB", str(tmp_path / "login.sqlite3"))
    monkeypatch.setenv("VEKTORYUM_LOGIN_LIMIT", "2")
    monkeypatch.setenv("VEKTORYUM_LOGIN_WINDOW_SECONDS", "600")
    module = _fake_main(tmp_path)
    PlatformIdentity(module).install()
    module._save_users(
        {
            "user@example.com": {
                "email": "user@example.com",
                "name": "User",
                "role": "user",
                "password": _hash_password("correct-password"),
            }
        }
    )
    client = TestClient(module.app)
    for _index in range(2):
        response = client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "wrong-password"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "E-posta veya şifre hatalı."
    limited = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "correct-password"},
    )
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) > 0


def test_browser_adapter_injects_request_script(tmp_path) -> None:
    module = _fake_main(tmp_path)
    static_dir = Path(module.__file__).parent / "static"
    static_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("<html><head></head><body>ok</body></html>", encoding="utf-8")
    install_platform_frontend(module)
    client = TestClient(module.app)
    index = client.get("/")
    assert index.status_code == 200
    assert f'<script src="{SCRIPT_PATH}"></script>' in index.text
    script = client.get(SCRIPT_PATH)
    assert script.status_code == 200
    assert REQUEST_HEADER in script.text
    assert REQUEST_COOKIE in script.text
