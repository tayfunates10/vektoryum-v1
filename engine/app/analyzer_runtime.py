"""Small job-scoped handoff for the final analyzer review verdict.

The pipeline and exact final evaluator run in different thread-pool calls. Job
directories are request-unique, so a bounded registry keyed by resolved job path
carries only the auto decision to the final exported ``<job_id>.svg`` evaluation.
Candidate and transform-journal SVGs never consume the entry.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any


_MAX_PENDING_JOBS = 1024
_LOCK = Lock()
_PENDING: OrderedDict[str, dict[str, Any]] = OrderedDict()


def register_job_auto_decision(job_dir: Path, decision: dict[str, Any]) -> None:
    key = str(Path(job_dir).resolve())
    with _LOCK:
        _PENDING[key] = deepcopy(decision)
        _PENDING.move_to_end(key)
        while len(_PENDING) > _MAX_PENDING_JOBS:
            _PENDING.popitem(last=False)


def take_final_svg_auto_decision(svg_path: Path) -> dict[str, Any] | None:
    path = Path(svg_path).resolve()
    parent = path.parent
    # Production export naming contract: <job_dir>/<job_dir.name>.svg.
    # Earlier candidate/journal evaluations use different filenames and must not
    # consume the final review decision.
    if path.name != f"{parent.name}.svg":
        return None
    with _LOCK:
        decision = _PENDING.pop(str(parent), None)
    return deepcopy(decision) if decision is not None else None


def clear_job_auto_decision(job_dir: Path) -> None:
    with _LOCK:
        _PENDING.pop(str(Path(job_dir).resolve()), None)


__all__ = [
    "clear_job_auto_decision",
    "register_job_auto_decision",
    "take_final_svg_auto_decision",
]
