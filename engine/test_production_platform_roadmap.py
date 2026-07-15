from __future__ import annotations

import json
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
REPO_DIR = ENGINE_DIR.parent
ROADMAP_PATH = ENGINE_DIR / "production_platform_roadmap.json"
EXPECTED_PHASES = ["PPC-1", "PPC-2", "PPC-3", "PPC-4"]
EXPECTED_LIMITATIONS = {
    "administrator_bootstrap_not_closed",
    "login_state_lifecycle_not_closed",
    "request_boundary_not_closed",
    "state_sync_integrity_not_closed",
    "job_retention_not_closed",
    "service_modes_not_closed",
    "gated_deploy_not_closed",
}


def _roadmap() -> dict:
    return json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))


def _evidence_path(relative: str) -> Path:
    return (ENGINE_DIR / relative).resolve()


def test_phase_order_and_status() -> None:
    data = _roadmap()
    assert data["schema_version"] == "production-platform-closure-v1"
    phases = data["phases"]
    assert [phase["id"] for phase in phases] == EXPECTED_PHASES
    assert [phase["status"] for phase in phases] == ["complete"] * 4
    assert all(len(phase["acceptance_criteria"]) >= 5 for phase in phases)


def test_limitations_have_one_closure_phase() -> None:
    data = _roadmap()
    phases = {phase["id"]: phase for phase in data["phases"]}
    limitations = data["known_limitations"]
    assert {item["id"] for item in limitations} == EXPECTED_LIMITATIONS
    assert len(limitations) == len({item["id"] for item in limitations})
    for item in limitations:
        assert item["status"] == "closed"
        assert phases[item["closure_phase"]]["status"] == "complete"
        assert _evidence_path(item["evidence"]).is_file()


def test_completed_phase_evidence_exists() -> None:
    completed = [phase for phase in _roadmap()["phases"] if phase["status"] == "complete"]
    assert [phase["id"] for phase in completed] == EXPECTED_PHASES
    for phase in completed:
        assert "test_production_platform_roadmap.py" in phase["evidence"]
        for relative in phase["evidence"]:
            assert _evidence_path(relative).is_file(), relative


def test_scope_separation() -> None:
    excluded = " ".join(_roadmap()["scope"]["excluded"]).lower()
    assert "vector engine" in excluded
    assert "analyzer" in excluded
    assert "billing" in excluded


def test_runtime_inventory() -> None:
    main_text = (ENGINE_DIR / "app/main.py").read_text(encoding="utf-8")
    settings_text = (ENGINE_DIR / "app/settings.py").read_text(encoding="utf-8")
    store_text = (ENGINE_DIR / "app/store.py").read_text(encoding="utf-8")
    persistence_text = (ENGINE_DIR / "app/platform_persistence.py").read_text(encoding="utf-8")
    operations_text = (ENGINE_DIR / "app/platform_operations.py").read_text(encoding="utf-8")
    runtime_text = (ENGINE_DIR / "app/runtime_main.py").read_text(encoding="utf-8")
    docker_text = (REPO_DIR / "Dockerfile").read_text(encoding="utf-8")
    deploy_text = (REPO_DIR / ".github/workflows/huggingface-deploy.yml").read_text(encoding="utf-8")

    assert "FastAPI(" in main_text
    assert "JOBS_ROOT" in main_text
    assert "max_upload_bytes" in settings_text
    assert "max_pixels" in settings_text
    assert "VEKTORYUM_DATASET" in store_text
    assert "atomic_write_json" in persistence_text
    assert "artifact_manifest" in persistence_text
    assert "RemoteSyncObservation" in persistence_text
    assert "ALLOWED_SERVICE_MODES" in operations_text
    assert "X-Correlation-ID" in operations_text
    assert "shutdown_in_progress" in operations_text
    assert "install_platform_identity" in runtime_text
    assert "install_platform_frontend" in runtime_text
    assert "install_platform_operations" in runtime_text
    assert "USER appuser" in docker_text
    assert "HEALTHCHECK" in docker_text
    assert "exec python -m uvicorn" in docker_text
    assert "github.event.workflow_run.head_sha" in deploy_text
    assert "Require SHA to equal current main" in deploy_text
    assert "needs: mandatory-ci" in deploy_text
