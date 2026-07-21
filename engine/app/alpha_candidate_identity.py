"""Stable candidate-identity contract for post-selection alpha finalization."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_ALPHA_SUFFIX = "_alpha"


def _candidate_names(result: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for collection_name in ("results", "scored"):
        for candidate in result.get(collection_name) or []:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _restore_selected_candidate_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Keep artifact transforms from masquerading as new engine candidates."""
    report = result.get("alpha_mask_report")
    best = result.get("best")
    if not (
        isinstance(report, dict)
        and report.get("applied") is True
        and isinstance(best, dict)
    ):
        return result

    finalized_name = best.get("name")
    if not isinstance(finalized_name, str) or not finalized_name.endswith(_ALPHA_SUFFIX):
        raise RuntimeError("source_alpha_candidate_identity_missing_suffix")

    source_name = finalized_name[: -len(_ALPHA_SUFFIX)]
    if not source_name:
        raise RuntimeError("source_alpha_candidate_identity_empty")

    known_names = _candidate_names(result)
    if source_name not in known_names:
        raise RuntimeError(
            "source_alpha_candidate_identity_unbound:"
            f"{source_name} not in production candidates"
        )

    best["name"] = source_name
    for candidate in result.get("scored") or []:
        if not isinstance(candidate, dict):
            continue
        if candidate is best or (
            candidate.get("name") == finalized_name
            and candidate.get("alpha_mask_report") is report
        ):
            candidate["name"] = source_name

    report["source_candidate_name"] = source_name
    report["finalization_stage"] = "source_alpha_vector_mask"
    result["candidate_identity"] = {
        "status": "preserved",
        "source_candidate_name": source_name,
        "artifact_transform": "source_alpha_vector_mask",
    }
    return result


def wrap_run_pipeline_preserving_candidate_identity(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Restore the engine candidate name after the alpha artifact transform."""
    if getattr(original, "__vektoryum_candidate_identity_preserved__", False):
        return original

    @wraps(original)
    def identity_preserving_pipeline(*args, **kwargs) -> dict[str, Any]:
        return _restore_selected_candidate_identity(original(*args, **kwargs))

    identity_preserving_pipeline.__vektoryum_candidate_identity_preserved__ = True
    return identity_preserving_pipeline
