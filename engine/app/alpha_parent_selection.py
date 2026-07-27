"""Compatibility wrapper that exposes alpha-parent trial diagnostics.

The production implementation remains byte-for-byte in
``app.alpha_parent_selection_base``. This module preserves its public/private
namespace and wraps only the orchestration boundary to attach JSON-safe trial
measurements. Candidate choice and artifact bytes are unchanged.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

from app import alpha_parent_selection_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def alpha_parent_trial_snapshot(trials: object) -> list[dict[str, Any]]:
    if not isinstance(trials, list):
        return []
    output: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, dict):
            continue
        candidate = trial.get("candidate") if isinstance(trial.get("candidate"), dict) else {}
        output.append(
            {
                "trial_index": index,
                "candidate_name": str(candidate.get("name") or "unknown"),
                "candidate_engine": str(candidate.get("engine") or "unknown"),
                "rendered_ok": bool(trial.get("rendered_ok")),
                "fidelity_score": (
                    None
                    if trial.get("fidelity_score") is None
                    else round(float(trial["fidelity_score"]), 8)
                ),
                "edge_f1": (
                    None
                    if trial.get("edge_f1") is None
                    else round(float(trial["edge_f1"]), 8)
                ),
                "path_count": int(trial.get("path_count") or 0),
                "byte_size": int(trial.get("byte_size") or 0),
            }
        )
    return output


def _select(result: dict[str, Any], source_path: Path, job_dir: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {"trials": [], "chosen": None}
    original_choose = _base.choose_alpha_parent_trial

    def capturing_choose(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
        captured["trials"] = alpha_parent_trial_snapshot(trials)
        chosen = original_choose(trials)
        if isinstance(chosen, dict):
            candidate = chosen.get("candidate") if isinstance(chosen.get("candidate"), dict) else {}
            captured["chosen"] = str(candidate.get("name") or "unknown")
        return chosen

    _base.choose_alpha_parent_trial = capturing_choose
    try:
        selected = _base._select(result, Path(source_path), Path(job_dir))
    finally:
        _base.choose_alpha_parent_trial = original_choose

    diagnostics = {
        "schema": "vektoryum-alpha-parent-trial-diagnostics-v1",
        "trial_count": len(captured["trials"]),
        "chosen_candidate_name": captured["chosen"],
        "trials": captured["trials"],
    }
    selected["alpha_parent_trial_diagnostics"] = diagnostics
    return selected


def wrap_run_pipeline_selecting_alpha_parent(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_alpha_parent_selected__", False):
        return original

    @wraps(original)
    def wrapped(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ) -> dict[str, Any]:
        result = original(
            image,
            original_path,
            trace_mode,
            job_dir,
            refine=refine,
            edge_cleanup=edge_cleanup,
        )
        return _select(result, Path(original_path), Path(job_dir))

    wrapped.__vektoryum_alpha_parent_selected__ = True
    return wrapped
