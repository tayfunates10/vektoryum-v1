from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "damaged"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "sha256": _digest(payload)}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(envelope))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def read_checked_json(path: Path) -> Any:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if envelope.get("sha256") != _digest(payload):
        raise ValueError(f"integrity check failed: {path}")
    return payload


class AtomicStateStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def write(self, name: str, payload: Any) -> Path:
        path = self.root / name
        atomic_write_json(path, payload)
        return path

    def read(self, name: str, default: Any = None) -> Any:
        path = self.root / name
        return default if not path.exists() else read_checked_json(path)


@dataclass(frozen=True)
class RemoteSyncObservation:
    status: str
    local_revision: int
    remote_revision: int | None
    reason: str | None = None


def accept_remote_state(local: dict[str, Any], remote: dict[str, Any]) -> RemoteSyncObservation:
    local_rev = int(local.get("revision", 0))
    remote_rev = int(remote.get("revision", 0))
    if remote_rev < local_rev:
        return RemoteSyncObservation("rejected_stale", local_rev, remote_rev, "remote_older_than_local")
    if remote_rev == local_rev and _digest(remote) != _digest(local):
        return RemoteSyncObservation("conflict", local_rev, remote_rev, "same_revision_different_content")
    return RemoteSyncObservation("accepted", local_rev, remote_rev)


class JobRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _meta(self, job_id: str) -> Path:
        return self._dir(job_id) / "job.json"

    def create(self, job_id: str, owner: str, now: int | None = None) -> dict[str, Any]:
        ts = int(time.time() if now is None else now)
        data = {
            "job_id": job_id,
            "owner": owner,
            "created_at": ts,
            "updated_at": ts,
            "state": "queued",
            "artifact_manifest": None,
        }
        atomic_write_json(self._meta(job_id), data)
        return data

    def load(self, job_id: str) -> dict[str, Any]:
        return read_checked_json(self._meta(job_id))

    def transition(self, job_id: str, state: str, now: int | None = None) -> dict[str, Any]:
        data = self.load(job_id)
        if data["state"] in TERMINAL_STATES:
            raise ValueError("terminal job is immutable")
        data["state"] = state
        data["updated_at"] = int(time.time() if now is None else now)
        atomic_write_json(self._meta(job_id), data)
        return data

    def finalize(self, job_id: str, state: str, artifacts: Iterable[Path], now: int | None = None) -> dict[str, Any]:
        if state not in TERMINAL_STATES:
            raise ValueError("final state must be terminal")
        data = self.load(job_id)
        if data["artifact_manifest"] is not None:
            raise ValueError("artifact manifest is immutable")
        manifest = []
        for path in sorted((Path(p) for p in artifacts), key=lambda p: p.name):
            manifest.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        data["state"] = state
        data["updated_at"] = int(time.time() if now is None else now)
        data["artifact_manifest"] = manifest
        atomic_write_json(self._meta(job_id), data)
        return data

    def recover(self) -> dict[str, list[str]]:
        result = {"complete": [], "incomplete": [], "damaged": []}
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                data = read_checked_json(child / "job.json")
                bucket = "complete" if data.get("state") in TERMINAL_STATES else "incomplete"
            except Exception:
                bucket = "damaged"
            result[bucket].append(child.name)
        return result

    def cleanup(self, *, now: int, retention_seconds: int, user_quota_bytes: int, disk_budget_bytes: int) -> list[str]:
        rows: list[tuple[int, str, int, str]] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                data = read_checked_json(child / "job.json")
            except Exception:
                continue
            size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
            rows.append((int(data.get("updated_at", 0)), child.name, size, str(data.get("owner", ""))))
        victims: set[str] = {job_id for updated, job_id, _, _ in rows if now - updated > retention_seconds}
        for owner in sorted({owner for *_, owner in rows}):
            total = sum(size for _, job_id, size, row_owner in rows if row_owner == owner and job_id not in victims)
            for _, job_id, size, row_owner in sorted(rows):
                if row_owner == owner and job_id not in victims and total > user_quota_bytes:
                    victims.add(job_id); total -= size
        total = sum(size for _, job_id, size, _ in rows if job_id not in victims)
        for _, job_id, size, _ in sorted(rows):
            if job_id not in victims and total > disk_budget_bytes:
                victims.add(job_id); total -= size
        for job_id in sorted(victims):
            shutil.rmtree(self._dir(job_id), ignore_errors=True)
        return sorted(victims)
