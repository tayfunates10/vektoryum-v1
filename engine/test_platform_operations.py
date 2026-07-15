from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.platform_operations import RuntimeState, install_platform_operations, service_mode


def test_invalid_service_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEKTORYUM_SERVICE_MODE", "invalid")
    with pytest.raises(RuntimeError, match="invalid VEKTORYUM_SERVICE_MODE"):
        service_mode()


def test_modes_health_correlation_and_maintenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEKTORYUM_SERVICE_MODE", "beta")
    app = FastAPI()

    @app.post("/write")
    async def write():
        return {"ok": True}

    state = install_platform_operations(app, revision="abc123")
    with TestClient(app) as client:
        live = client.get("/livez")
        assert live.json() == {"status": "ok", "check": "liveness", "mode": "beta", "revision": "abc123"}
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        response = client.post("/write", headers={"X-Correlation-ID": "cid-1"})
        assert response.status_code == 200
        assert response.headers["X-Correlation-ID"] == "cid-1"
        monkeypatch.setenv("VEKTORYUM_SERVICE_MODE", "maintenance")
        blocked = client.post("/write")
        assert blocked.status_code == 503
        assert blocked.json()["status"] == "maintenance"
        assert client.get("/readyz").status_code == 503
        state.accepting_requests = False
        unavailable = client.get("/anything")
        assert unavailable.status_code == 503
        assert unavailable.json()["reason"] == "shutdown_in_progress"


def test_structured_logs_are_json(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("VEKTORYUM_SERVICE_MODE", "live")
    app = FastAPI()
    install_platform_operations(app, revision="r1")
    with caplog.at_level("INFO", logger="vektoryum.operations"):
        with TestClient(app) as client:
            client.get("/livez")
    records = [json.loads(record.message) for record in caplog.records if record.name == "vektoryum.operations"]
    assert {item["event"] for item in records} >= {"request_started", "request_finished"}
    assert all("correlation_id" in item for item in records if item["event"].startswith("request_"))


def test_runtime_state_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEKTORYUM_SERVICE_MODE", "live")
    state = RuntimeState(started_at=1.0)
    assert state.readiness() == (True, [])
    state.accepting_requests = False
    assert state.readiness() == (False, ["shutdown_in_progress"])
