"""Engine-diverse, evidence-rich alpha-parent selection.

The established production implementation is retained byte-for-byte in
``app.alpha_parent_selection_base``. This compatibility module re-exports its
namespace and keeps the same quality margins/journal gates, while making the
bounded parent shortlist engine-diverse and recording fail-closed trial outcomes.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from PIL import Image
import numpy as np

from app import alpha_parent_selection_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_legacy_shortlist = _base._shortlist
_ERROR_LIMIT = 180
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-~+]+){2,}")


def _identity(candidate: dict[str, Any]) -> str | None:
    path = _base._path(candidate)
    return None if path is None else str(path.resolve())


def _candidate_snapshot(candidate: object) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    return {
        "candidate_name": str(candidate.get("name") or "unknown"),
        "candidate_engine": str(candidate.get("engine") or "unknown"),
        "fidelity_score_before_alpha": (
            None
            if candidate.get("fidelity_score") is None
            else round(float(candidate["fidelity_score"]), 8)
        ),
        "edge_f1_before_alpha": (
            None
            if _base._edge_f1(candidate) is None
            else round(float(_base._edge_f1(candidate)), 8)
        ),
        "path_count_before_alpha": _base._path_count(candidate),
    }


def _safe_error(exc: BaseException) -> dict[str, str]:
    text = _ABSOLUTE_PATH.sub("<redacted-path>", str(exc))[:_ERROR_LIMIT]
    return {"error_class": type(exc).__name__, "error_message": text}


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
    """Keep the legacy bounded list but force one alternate-engine measurement."""
    legacy = list(_legacy_shortlist(result))
    best = result.get("best")
    if not isinstance(best, dict):
        return legacy
    best_engine = str(best.get("engine") or "")
    alternates = [
        item
        for item in _rendered_pool(result)
        if str(item.get("engine") or "") != best_engine
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
    if alternate_identity in {_identity(item) for item in legacy}:
        return legacy

    if len(legacy) < _base._MAX_TRIALS:
        legacy.append(alternate)
    else:
        replaceable = [
            (index, item)
            for index, item in enumerate(legacy)
            if item is not best and str(item.get("engine") or "") == best_engine
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


def _attach_diagnostics(
    result: dict[str, Any],
    *,
    status: str,
    shortlist: list[dict[str, Any]] | None = None,
    trials: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    chosen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chosen_candidate = (
        chosen.get("candidate")
        if isinstance(chosen, dict) and isinstance(chosen.get("candidate"), dict)
        else {}
    )
    result["alpha_parent_trial_diagnostics"] = {
        "schema": "vektoryum-alpha-parent-trial-diagnostics-v2",
        "shortlist_policy": "legacy_plus_highest_fidelity_alternate_engine",
        "status": status,
        "shortlist_count": len(shortlist or []),
        "shortlist": [
            {"shortlist_index": index, **_candidate_snapshot(candidate)}
            for index, candidate in enumerate(shortlist or [], start=1)
        ],
        "trial_count": len(trials or []),
        "chosen_candidate_name": str(chosen_candidate.get("name") or "") or None,
        "failures": failures or [],
        "trials": alpha_parent_trial_snapshot(trials or []),
    }
    return result


def _select(result: dict[str, Any], source_path: Path, job_dir: Path) -> dict[str, Any]:
    from app.alpha_pipeline_retry import alpha_remediation_enabled

    if not alpha_remediation_enabled():
        return _attach_diagnostics(result, status="legacy_pass_not_authorized")

    best = result.get("best")
    mode = str(result.get("mode_used") or "")
    if not isinstance(best, dict) or mode not in _base._SUPPORTED_MODES:
        return _attach_diagnostics(result, status="unsupported_result_or_mode")
    try:
        with Image.open(source_path) as source:
            levels = np.unique(np.asarray(source.convert("RGBA"))[:, :, 3])
    except (OSError, ValueError) as exc:
        return _attach_diagnostics(
            result,
            status="source_alpha_read_failed",
            failures=[{"stage": "source_alpha_read", **_safe_error(exc)}],
        )
    if len(levels) <= 1 or len(levels) > _base._MAX_ALPHA_LEVELS:
        return _attach_diagnostics(result, status="alpha_level_count_not_supported")

    shortlist = engine_diverse_shortlist(result)
    if len(shortlist) <= 1:
        return _attach_diagnostics(
            result,
            status="shortlist_insufficient",
            shortlist=shortlist,
        )

    from app import alpha_svg_mask
    from app.pipeline import score_candidate

    trials: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="alpha-parent-", dir=job_dir) as temporary:
        for index, candidate in enumerate(shortlist):
            parent = _base._path(candidate)
            candidate_fields = {
                "shortlist_index": index + 1,
                **_candidate_snapshot(candidate),
            }
            if parent is None:
                failures.append({"stage": "parent_path", **candidate_fields})
                continue
            trial_path = Path(temporary) / f"trial-{index}.svg"
            shutil.copy2(parent, trial_path)
            try:
                alpha_svg_mask.apply_source_alpha_mask(trial_path, source_path, mode)
            except Exception as exc:  # noqa: BLE001 - evidence, fail closed
                failures.append(
                    {"stage": "alpha_transform", **candidate_fields, **_safe_error(exc)}
                )
                continue
            try:
                rescored = score_candidate(
                    {**candidate, "svg_path": trial_path},
                    source_path,
                    result.get("analysis") or {},
                    mode,
                )
            except Exception as exc:  # noqa: BLE001 - evidence, fail closed
                failures.append(
                    {"stage": "rescore", **candidate_fields, **_safe_error(exc)}
                )
                continue
            if not isinstance(rescored, dict):
                failures.append({"stage": "rescore_no_result", **candidate_fields})
                continue
            trials.append(
                {
                    "candidate": candidate,
                    "rendered_ok": bool(rescored.get("rendered_ok")),
                    "fidelity_score": rescored.get("fidelity_score"),
                    "edge_f1": _base._edge_f1(rescored),
                    "path_count": _base._path_count(rescored),
                    "byte_size": trial_path.stat().st_size,
                }
            )

    chosen = _base.choose_alpha_parent_trial(trials)
    if chosen is None:
        return _attach_diagnostics(
            result,
            status="no_measured_parent",
            shortlist=shortlist,
            trials=trials,
            failures=failures,
        )
    candidate = chosen["candidate"]
    selected_path = _base._path(candidate)
    best_path = _base._path(best)
    if selected_path is None or best_path is None:
        return _attach_diagnostics(
            result,
            status="selected_or_best_path_missing",
            shortlist=shortlist,
            trials=trials,
            failures=failures,
            chosen=chosen,
        )
    if selected_path.resolve() == best_path.resolve():
        return _attach_diagnostics(
            result,
            status="original_parent_retained",
            shortlist=shortlist,
            trials=trials,
            failures=failures,
            chosen=chosen,
        )

    merged_journal, selection_stage, rejection = _base._bind_parent_selection_journal(
        result,
        best_path=best_path,
        selected_path=selected_path,
        source_path=source_path,
        mode=mode,
        trial=chosen,
    )
    if rejection is not None or merged_journal is None:
        result["alpha_parent_selection"] = {
            "status": "preserved_original",
            "reason": rejection or "journal_not_available",
            "source_alpha_level_count": int(len(levels)),
            "original_candidate_name": str(best.get("name") or best_path.stem),
            "proposed_candidate_name": str(candidate.get("name") or selected_path.stem),
            "trial_count": len(trials),
            "journal_stage": selection_stage,
        }
        return _attach_diagnostics(
            result,
            status="alternate_parent_journal_rejected",
            shortlist=shortlist,
            trials=trials,
            failures=failures,
            chosen=chosen,
        )

    result["best"] = candidate
    result["transform_journal"] = merged_journal
    result["selection_reason"] = (
        f"{result.get('selection_reason') or 'selected'}+alpha_parent_editability"
    )
    result["alpha_parent_selection"] = {
        "status": "selected",
        "source_alpha_level_count": int(len(levels)),
        "original_candidate_name": str(best.get("name") or best_path.stem),
        "selected_candidate_name": str(candidate.get("name") or selected_path.stem),
        "trial_count": len(trials),
        "selected_fidelity_score": float(chosen["fidelity_score"]),
        "selected_edge_f1": chosen.get("edge_f1"),
        "selected_path_count_after_alpha": int(chosen["path_count"]),
        "selected_byte_size_after_alpha": int(chosen["byte_size"]),
        "fidelity_margin": _base._FIDELITY_MARGIN,
        "edge_f1_margin": _base._EDGE_MARGIN,
        "journal_stage_status": str((selection_stage or {}).get("status") or "accepted"),
        "journal_chain_valid": True,
    }
    return _attach_diagnostics(
        result,
        status="alternate_parent_selected",
        shortlist=shortlist,
        trials=trials,
        failures=failures,
        chosen=chosen,
    )


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
