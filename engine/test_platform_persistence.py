from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.platform_persistence import AtomicStateStore, JobRegistry, accept_remote_state


def test_atomic_state_integrity_and_tamper_detection(tmp_path: Path) -> None:
    store = AtomicStateStore(tmp_path)
    store.write("users.json", {"revision": 2, "users": [{"id": "u1"}]})
    assert store.read("users.json")["revision"] == 2
    path = tmp_path / "users.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["revision"] = 3
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        store.read("users.json")


def test_job_owner_timestamps_terminal_state_and_manifest_immutable(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs")
    registry.create("j1", "u1", now=10)
    registry.transition("j1", "running", now=20)
    artifact = registry._dir("j1") / "result.svg"
    artifact.write_text("<svg/>", encoding="utf-8")
    finished = registry.finalize("j1", "succeeded", [artifact], now=30)
    assert finished["owner"] == "u1"
    assert finished["created_at"] == 10
    assert finished["updated_at"] == 30
    assert finished["state"] == "succeeded"
    assert finished["artifact_manifest"][0]["name"] == "result.svg"
    with pytest.raises(ValueError, match="terminal job is immutable"):
        registry.transition("j1", "running", now=40)
    with pytest.raises(ValueError, match="artifact manifest is immutable"):
        registry.finalize("j1", "failed", [artifact], now=40)


def test_restart_recovery_separates_complete_incomplete_and_damaged(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs")
    registry.create("complete", "u", now=1)
    artifact = registry._dir("complete") / "a.svg"
    artifact.write_text("ok", encoding="utf-8")
    registry.finalize("complete", "succeeded", [artifact], now=2)
    registry.create("incomplete", "u", now=3)
    registry._dir("damaged").mkdir()
    (registry._dir("damaged") / "job.json").write_text("{}", encoding="utf-8")
    assert registry.recover() == {
        "complete": ["complete"],
        "incomplete": ["incomplete"],
        "damaged": ["damaged"],
    }


def test_cleanup_is_deterministic_for_retention_quota_and_disk_budget(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs")
    for job_id, owner, updated, size in [
        ("old", "u1", 1, 20),
        ("u1-a", "u1", 90, 20),
        ("u1-b", "u1", 91, 20),
        ("u2-a", "u2", 92, 20),
    ]:
        registry.create(job_id, owner, now=updated)
        payload = registry._dir(job_id) / "blob.bin"
        payload.write_bytes(b"x" * size)
        registry.transition(job_id, "running", now=updated)
    removed = registry.cleanup(
        now=100,
        retention_seconds=50,
        user_quota_bytes=300,
        disk_budget_bytes=700,
    )
    assert removed == ["old"]


def test_remote_state_never_replaces_newer_local_and_conflicts_are_observable() -> None:
    local = {"revision": 5, "users": ["new"]}
    stale = accept_remote_state(local, {"revision": 4, "users": ["old"]})
    assert stale.status == "rejected_stale"
    assert stale.reason == "remote_older_than_local"
    conflict = accept_remote_state(local, {"revision": 5, "users": ["different"]})
    assert conflict.status == "conflict"
    assert conflict.reason == "same_revision_different_content"
    assert accept_remote_state(local, {"revision": 6, "users": ["remote"]}).status == "accepted"
