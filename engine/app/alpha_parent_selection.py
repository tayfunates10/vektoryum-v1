"""Compatibility wrapper with engine-diverse alpha-parent trials.

The production implementation remains byte-for-byte in
``app.alpha_parent_selection_base``. This module preserves its public/private
namespace, keeps the established quality margins, and changes only the bounded
trial shortlist: when multiple vector engines exist, at least one candidate from
an engine different from the current winner is measured after the real source-
alpha transform. JSON-safe trial evidence is attached for regression analysis.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

from app import alpha_parent_selection_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def _identity(candidate: dict[str, Any]) -> str | None:
    path = _base._path(candidate)
    return None if path is None else str(path.resolve())


def _rendered_pool(result: dict[str, Any]) -> list[dict[str, Any]]:
    best = result.get("best")
    pool: list[dict[str, Any]] = []
    if isinstance(best, dict):
        pool.append(best)
    pool.extend(
        item
        for item in result.get("scored") or []
        if isinstance(item, dict)
        and item.get("rendered_ok")
        and not (item.get("score_details") or {}).get("has_bitmap")
    )
    unique: dict[str, dict[str, Any]] = {}
    for item in pool:
        identity = _identity(item)
        if identity is not None:
            unique.setdefault(identity, item)
    return list(unique.values())


def engine_diverse_shortlist(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve the legacy shortlist and guarantee one alternate engine trial.

    The old shortlist can collapse to only gradient variants because the current
    winner is simultaneously the leanest and highest-fidelity candidate. For
    transparent flat artwork that means a VTracer parent is never measured after
    source-alpha reconstruction. This function starts from the exact legacy list,
    then appends or substitutes only one highest-fidelity alternate-engine parent.
    Established alpha selection margins remain unchanged.
    """
    legacy = list(_base._shortlist(result))
    best = result.get("best")
    if not isinstance(best, dict):
        return legacy
    best_engine = str(best.get("engine") or "")
    pool = _rendered_pool(result)
    alternates = [
        item for item in pool if str(item.get("engine") or "") != best_engine
    ]
    if not alternates:
        return legacy
    alternate = max(
        alternates,
        key=lambda item: (
            float(item.get("fidelity_score") or 0.0),
            float(_base._edge_f1(item) or 0.0),
            -_base._path_count(item),
        ),
    )
    alternate_identity = _identity(alternate)
    identities = {_identity(item) for item in legacy}
    if alternate_identity in identities:
        return legacy

    if len(legacy) < _base._MAX_TRIALS:
        legacy.append(alternate)
    else:
        replaceable = [
            (index, item)
            for index, item in enumerate(legacy)
            if item is not best
            and str(item.get("engine") or "") == best_engine
        ]
        if not replaceable:
            return legacy
        replace_index, _ = min(
            replaceable,
            key=lambda pair: (
                float(pair[1].get("fidelity_score") or 0.0),
                float(_base._edge_f1(pair[1]) or 0.0),
            ),
        )
        legacy[replace_index] = alternate

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in legacy:
        identity = _identity(item)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        output.append(item)
        if len(output) == _base._MAX_TRIALS:
            break
    return output


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
    original_shortlist = _base._shortlist

    def capturing_choose(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
        captured["trials"] = alpha_parent_trial_snapshot(trials)
        chosen = original_choose(trials)
        if isinstance(chosen, dict):
            candidate = chosen.get("candidate") if isinstance(chosen.get("candidate"), dict) else {}
            captured["chosen"] = str(candidate.get("name") or "unknown")
        return chosen

    _base.choose_alpha_parent_trial = capturing_choose
    _base._shortlist = engine_diverse_shortlist
    try:
        selected = _base._select(result, Path(source_path), Path(job_dir))
    finally:
        _base.choose_alpha_parent_trial = original_choose
        _base._shortlist = original_shortlist

    diagnostics = {
        "schema": "vektoryum-alpha-parent-trial-diagnostics-v1",
        "shortlist_policy": "legacy_plus_highest_fidelity_alternate_engine",
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
