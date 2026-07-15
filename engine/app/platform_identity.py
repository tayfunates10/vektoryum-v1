"""Production account and login-state contract for the FastAPI runtime."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

SESSION_COOKIE = "session"
REQUEST_COOKIE = "vektoryum_request"
REQUEST_HEADER = "X-Vektoryum-Request"
PROTECTED_POST_PATHS = frozenset({"/api/vectorize", "/api/auth/logout"})


def _integer_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_key(*parts: str) -> str:
    return "\x00".join(str(part).strip().lower() for part in parts)


@dataclass(frozen=True)
class AttemptDecision:
    allowed: bool
    remaining: int
    retry_after: int


class LoginStateStore:
    """SQLite-backed expiring login state and deterministic attempt windows."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS login_sessions (
                    session_hash TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_login_sessions_email
                    ON login_sessions(email);
                CREATE INDEX IF NOT EXISTS idx_login_sessions_expires
                    ON login_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS attempt_windows (
                    namespace TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key_hash, window_start)
                );
                """
            )

    def create_session(
        self,
        email: str,
        *,
        ttl_seconds: int,
        now: int | None = None,
    ) -> tuple[str, str, int]:
        issued_at = int(time.time() if now is None else now)
        expires_at = issued_at + int(ttl_seconds)
        session_token = secrets.token_urlsafe(32)
        request_token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO login_sessions(
                        session_hash, email, request_hash, created_at, expires_at, revoked_at
                    ) VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        _sha256(session_token),
                        email.strip().lower(),
                        _sha256(request_token),
                        issued_at,
                        expires_at,
                    ),
                )
                connection.execute(
                    "DELETE FROM login_sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL",
                    (issued_at,),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return session_token, request_token, expires_at

    def resolve(self, session_token: str | None, *, now: int | None = None) -> dict[str, Any] | None:
        if not session_token:
            return None
        current = int(time.time() if now is None else now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT email, created_at, expires_at
                FROM login_sessions
                WHERE session_hash = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (_sha256(session_token), current),
            ).fetchone()
        return dict(row) if row is not None else None

    def rotate_request_token(
        self,
        session_token: str,
        *,
        now: int | None = None,
    ) -> str | None:
        current = int(time.time() if now is None else now)
        request_token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE login_sessions
                SET request_hash = ?
                WHERE session_hash = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (_sha256(request_token), _sha256(session_token), current),
            )
        return request_token if cursor.rowcount == 1 else None

    def verify_request_token(
        self,
        session_token: str | None,
        request_token: str | None,
        *,
        now: int | None = None,
    ) -> bool:
        if not session_token or not request_token:
            return False
        current = int(time.time() if now is None else now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_hash
                FROM login_sessions
                WHERE session_hash = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (_sha256(session_token), current),
            ).fetchone()
        return bool(row and secrets.compare_digest(str(row["request_hash"]), _sha256(request_token)))

    def revoke(self, session_token: str | None, *, now: int | None = None) -> bool:
        if not session_token:
            return False
        current = int(time.time() if now is None else now)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE login_sessions SET revoked_at = ?
                WHERE session_hash = ? AND revoked_at IS NULL
                """,
                (current, _sha256(session_token)),
            )
        return cursor.rowcount == 1

    def revoke_email(self, email: str, *, now: int | None = None) -> int:
        current = int(time.time() if now is None else now)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE login_sessions SET revoked_at = ?
                WHERE email = ? AND revoked_at IS NULL
                """,
                (current, email.strip().lower()),
            )
        return int(cursor.rowcount)

    def consume_attempt(
        self,
        namespace: str,
        key: str,
        *,
        limit: int,
        window_seconds: int,
        now: int | None = None,
    ) -> AttemptDecision:
        current = int(time.time() if now is None else now)
        window_start = current - (current % int(window_seconds))
        key_hash = _sha256(_canonical_key(namespace, key))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO attempt_windows(namespace, key_hash, window_start, count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(namespace, key_hash, window_start)
                    DO UPDATE SET count = count + 1
                    """,
                    (namespace, key_hash, window_start),
                )
                row = connection.execute(
                    """
                    SELECT count FROM attempt_windows
                    WHERE namespace = ? AND key_hash = ? AND window_start = ?
                    """,
                    (namespace, key_hash, window_start),
                ).fetchone()
                connection.execute(
                    "DELETE FROM attempt_windows WHERE window_start < ?",
                    (window_start - int(window_seconds) * 2,),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        count = int(row["count"])
        return AttemptDecision(
            allowed=count <= int(limit),
            remaining=max(0, int(limit) - count),
            retry_after=max(1, window_start + int(window_seconds) - current),
        )

    def reset_attempt(self, namespace: str, key: str) -> None:
        key_hash = _sha256(_canonical_key(namespace, key))
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM attempt_windows WHERE namespace = ? AND key_hash = ?",
                (namespace, key_hash),
            )


class PlatformIdentity:
    def __init__(self, main_module: ModuleType, app: FastAPI | None = None):
        self.main = main_module
        self.app = app or main_module.app
        configured = os.environ.get("VEKTORYUM_LOGIN_DB", "").strip()
        self.db_path = Path(configured) if configured else Path(main_module.DATA_ROOT) / "login_state.sqlite3"
        self.state = LoginStateStore(self.db_path)
        self.ttl_seconds = _integer_env("VEKTORYUM_LOGIN_TTL_SECONDS", 14 * 24 * 60 * 60, 60)
        self.login_limit = _integer_env("VEKTORYUM_LOGIN_LIMIT", 8)
        self.login_window = _integer_env("VEKTORYUM_LOGIN_WINDOW_SECONDS", 15 * 60, 60)
        self.register_limit = _integer_env("VEKTORYUM_REGISTER_LIMIT", 5)
        self.register_window = _integer_env("VEKTORYUM_REGISTER_WINDOW_SECONDS", 60 * 60, 60)
        environment = os.environ.get("VEKTORYUM_ENVIRONMENT", "development").strip().lower()
        self.cookie_secure = _truthy_env(
            "VEKTORYUM_COOKIE_SECURE",
            default=environment in {"production", "live"},
        )

    def _atomic_users_write(self, users: dict[str, Any]) -> None:
        path = Path(self.main.USERS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(users, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def load_users(self) -> dict[str, Any]:
        path = Path(self.main.USERS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

        email = os.environ.get("VEKTORYUM_ADMIN_EMAIL", "").strip().lower()
        password = os.environ.get("VEKTORYUM_ADMIN_PASSWORD", "")
        if bool(email) != bool(password):
            raise RuntimeError("administrator bootstrap configuration is incomplete")
        users: dict[str, Any] = {}
        if email:
            if "@" not in email or len(password) < 12 or password == "admin123":
                raise RuntimeError("administrator bootstrap configuration is invalid")
            users[email] = {
                "email": email,
                "name": "Vektoryum Yönetici",
                "role": "admin",
                "password": self.main._hash_password(password),
            }
        self._atomic_users_write(users)
        return users

    def save_users(self, users: dict[str, Any]) -> None:
        self._atomic_users_write(users)
        try:
            from app import store

            store.persist(Path(self.main.USERS_FILE), "users.json")
        except Exception:
            pass

    def current_user(self, session_token: str | None) -> dict[str, Any] | None:
        login = self.state.resolve(session_token)
        if login is None:
            return None
        return self.load_users().get(str(login["email"]).lower())

    def _set_cookie_pair(
        self,
        response: Response,
        session_token: str,
        request_token: str,
    ) -> None:
        common = {
            "max_age": self.ttl_seconds,
            "secure": self.cookie_secure,
            "samesite": "lax",
            "path": "/",
        }
        response.set_cookie(SESSION_COOKIE, session_token, httponly=True, **common)
        response.set_cookie(REQUEST_COOKIE, request_token, httponly=False, **common)

    def _set_request_cookie(self, response: Response, request_token: str) -> None:
        response.set_cookie(
            REQUEST_COOKIE,
            request_token,
            max_age=self.ttl_seconds,
            secure=self.cookie_secure,
            httponly=False,
            samesite="lax",
            path="/",
        )

    def _delete_cookies(self, response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(REQUEST_COOKIE, path="/")

    @staticmethod
    def _client_key(request: Request, email: str = "") -> str:
        host = request.client.host if request.client is not None else "unknown"
        return _canonical_key(host, email)

    @staticmethod
    def _raise_limited(decision: AttemptDecision) -> None:
        raise HTTPException(
            status_code=429,
            detail="İstek sınırı aşıldı. Lütfen daha sonra tekrar deneyin.",
            headers={"Retry-After": str(decision.retry_after)},
        )

    def _remove_auth_routes(self) -> None:
        targets = {
            ("/api/auth/register", "POST"),
            ("/api/auth/login", "POST"),
            ("/api/auth/logout", "POST"),
            ("/api/auth/me", "GET"),
        }
        kept = []
        for route in self.app.router.routes:
            path = getattr(route, "path", None)
            methods = set(getattr(route, "methods", set()) or set())
            if any(path == target_path and target_method in methods for target_path, target_method in targets):
                continue
            kept.append(route)
        self.app.router.routes[:] = kept

    def install(self) -> "PlatformIdentity":
        if getattr(self.app.state, "platform_identity", None) is not None:
            return self.app.state.platform_identity

        self.main._load_users = self.load_users
        self.main._save_users = self.save_users
        self.main._current_user = self.current_user
        self.main.SESSIONS = {}
        self._remove_auth_routes()

        async def register(request: Request, response: Response, payload: dict[str, str]):
            email = (payload.get("email") or "").strip().lower()
            name = (payload.get("name") or "").strip()
            password = payload.get("password") or ""
            decision = self.state.consume_attempt(
                "register",
                self._client_key(request, email),
                limit=self.register_limit,
                window_seconds=self.register_window,
            )
            if not decision.allowed:
                self._raise_limited(decision)
            if not email or "@" not in email or len(password) < 8:
                raise HTTPException(status_code=400, detail="Geçerli e-posta ve en az 8 karakter şifre girin.")
            users = self.load_users()
            if email in users:
                raise HTTPException(status_code=409, detail="Bu e-posta zaten kayıtlı.")
            users[email] = {
                "email": email,
                "name": name or email.split("@")[0],
                "role": "user",
                "password": self.main._hash_password(password),
            }
            self.save_users(users)
            session_token, request_token, _expires_at = self.state.create_session(
                email,
                ttl_seconds=self.ttl_seconds,
            )
            self._set_cookie_pair(response, session_token, request_token)
            return {"user": self.main._safe_user(users[email])}

        async def login(request: Request, response: Response, payload: dict[str, str]):
            email = (payload.get("email") or "").strip().lower()
            password = payload.get("password") or ""
            key = self._client_key(request, email)
            decision = self.state.consume_attempt(
                "login",
                key,
                limit=self.login_limit,
                window_seconds=self.login_window,
            )
            if not decision.allowed:
                self._raise_limited(decision)
            user = self.load_users().get(email)
            valid = bool(user and self.main._verify_password(password, user.get("password", "")))
            if user and user.get("role") == "admin" and password == "admin123":
                valid = False
            if not valid:
                raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
            self.state.reset_attempt("login", key)
            session_token, request_token, _expires_at = self.state.create_session(
                email,
                ttl_seconds=self.ttl_seconds,
            )
            self._set_cookie_pair(response, session_token, request_token)
            return {
                "user": self.main._safe_user(user),
                "admin_url": "/admin" if user.get("role") == "admin" else None,
            }

        async def logout(request: Request, response: Response):
            self.state.revoke(request.cookies.get(SESSION_COOKIE))
            self._delete_cookies(response)
            return {"ok": True}

        async def me(request: Request, response: Response):
            session_token = request.cookies.get(SESSION_COOKIE)
            user = self.current_user(session_token)
            if user is None:
                self._delete_cookies(response)
                return {"user": None}
            request_token = self.state.rotate_request_token(session_token)
            if request_token is None:
                self._delete_cookies(response)
                return {"user": None}
            self._set_request_cookie(response, request_token)
            return {"user": self.main._safe_user(user)}

        self.app.add_api_route("/api/auth/register", register, methods=["POST"], summary="Kullanıcı kaydı")
        self.app.add_api_route("/api/auth/login", login, methods=["POST"], summary="Kullanıcı girişi")
        self.app.add_api_route("/api/auth/logout", logout, methods=["POST"], summary="Çıkış")
        self.app.add_api_route("/api/auth/me", me, methods=["GET"], summary="Aktif kullanıcı")

        @self.app.middleware("http")
        async def request_boundary(request: Request, call_next):
            if request.method.upper() == "POST" and request.url.path in PROTECTED_POST_PATHS:
                session_token = request.cookies.get(SESSION_COOKIE)
                if session_token:
                    cookie_token = request.cookies.get(REQUEST_COOKIE)
                    header_token = request.headers.get(REQUEST_HEADER)
                    tokens_match = bool(
                        cookie_token
                        and header_token
                        and secrets.compare_digest(cookie_token, header_token)
                    )
                    if not tokens_match or not self.state.verify_request_token(session_token, header_token):
                        return JSONResponse(
                            status_code=403,
                            content={"detail": "İstek doğrulaması başarısız."},
                        )
            return await call_next(request)

        self.app.state.platform_identity = self
        return self


def install_platform_identity(main_module: ModuleType, app: FastAPI | None = None) -> PlatformIdentity:
    return PlatformIdentity(main_module, app=app).install()


__all__ = [
    "AttemptDecision",
    "LoginStateStore",
    "PlatformIdentity",
    "REQUEST_COOKIE",
    "REQUEST_HEADER",
    "SESSION_COOKIE",
    "install_platform_identity",
]
