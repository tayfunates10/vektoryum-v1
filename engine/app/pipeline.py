"""Pipeline public facade with AI-2 selection/lifecycle policy.

The established pipeline implementation lives in :mod:`app.pipeline_core`.
This facade keeps that implementation byte-identical while applying the small
AI-2 manager-remediation policies at its public extension points:

* connected-component safety filters candidate selection without rewriting
  measured total/fidelity scores;
* the pre-existing closed-filled-cycle structural invariant is an eligibility
  tier, never a numeric score offset;
* neutral-palette restoration happens after candidate generation and before
  scoring, so scoring/evaluation itself is byte-pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app import pipeline_core as _core
from app.neutral_palette import (
    detect_neutral_luminance_bands,
    geometric_preserves_neutral_palette,
    restore_layered_neutral_svg_palette,
)

# Preserve the existing public/compatibility surface, including internal helpers
# imported by the repository's regression tests.
for _name in dir(_core):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, getattr(_core, _name))

_base_select_best = _core.select_best
_base_produce_candidate = _core.produce_candidate
_base_produce_and_score_job = _core._produce_and_score_job
_base_apply_editability_preference = _core._apply_editability_preference
_base_refine_best = _core.refine_best
_base_refit_one = _core._refit_one
_base_apply_boundary_refit = _core._apply_boundary_refit


def _selection_tier(candidate: dict[str, Any]) -> int:
    """Return independent eligibility tier; lower is safer.

    Tier 0: component-safe / component gate not applicable.
    Tier 1: applicable CC fail or needs_review.
    Tier 2: structurally invalid open filled cycle.

    The tier is deliberately separate from total/fidelity score so quality
    measurements stay numerically honest.  Within a tier the established
    selector remains unchanged.
    """
    component = candidate.get("component_quality") or {}
    if bool(component.get("open_required_cycle")):
        return 2
    if bool(
        candidate.get(
            "selection_safe",
            not candidate.get("selection_disqualified", False),
        )
    ):
        return 0
    return 1


def _selection_pool(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only the best available eligibility tier, preserving legacy rank."""
    if not scored:
        return scored
    best_tier = min(_selection_tier(candidate) for candidate in scored)
    pool = [candidate for candidate in scored if _selection_tier(candidate) == best_tier]
    return pool or scored


def select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
    """Run the legacy selector inside the independent safety/structure pool."""
    legacy_chosen, legacy_raw, legacy_reason = _base_select_best(scored, mode)
    pool = _selection_pool(scored)
    if len(pool) == len(scored):
        return legacy_chosen, legacy_raw, legacy_reason
    chosen, _safe_raw, reason = _base_select_best(pool, mode)
    return chosen, legacy_raw, f"component_integrity_guard+{reason}"


def produce_candidate(
    name: str,
    spec: dict[str, Any],
    preprocessed_path: Path,
    mode: str,
    job_dir: Path,
    original_path: Path | None = None,
    palette_cap: int | None = None,
) -> dict[str, Any]:
    """Finalize narrow neutral-palette repair before any candidate scoring."""
    result = _base_produce_candidate(
        name,
        spec,
        preprocessed_path,
        mode,
        job_dir,
        original_path=original_path,
        palette_cap=palette_cap,
    )
    result.setdefault(
        "neutral_palette_restore",
        {"applied": False, "reason": "not_layered_neutral_geometric"},
    )
    if not result.get("success") or original_path is None or mode != "geometric_logo":
        return result

    neutral_report = detect_neutral_luminance_bands(original_path)
    if not geometric_preserves_neutral_palette(mode, neutral_report):
        return result

    # This is candidate construction, not scoring. The returned SVG is the
    # exact immutable artifact score_vector_candidate() will observe.
    restore_report = restore_layered_neutral_svg_palette(
        Path(result["svg_path"]), Path(original_path), mode=mode
    )
    result["neutral_palette_restore"] = restore_report
    return result


def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Picklable facade entrypoint so worker processes install the same policy."""
    return _base_produce_and_score_job(args)


def _apply_editability_preference(
    scored: list[dict[str, Any]], current_best: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    pool = _selection_pool(scored)
    if current_best not in pool:
        current_best = max(pool, key=_core._fidelity_rank_key)
    return _base_apply_editability_preference(pool, current_best)


def refine_best(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    preprocessed_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    refined, info = _base_refine_best(
        best, mode, analysis, original_path, preprocessed_path, job_dir, scored
    )
    if _selection_tier(refined) > _selection_tier(best):
        return best, {
            **info,
            "applied": False,
            "component_integrity_rejected": refined.get("name"),
        }
    return refined, info


def _refit_one(
    cand: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    job_dir: Path,
) -> dict[str, Any] | None:
    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)
    if refined is not None and _selection_tier(refined) > _selection_tier(cand):
        return None
    return refined


def _apply_boundary_refit(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate, info = _base_apply_boundary_refit(
        best, mode, analysis, original_path, job_dir, scored
    )
    if _selection_tier(candidate) > _selection_tier(best):
        return best, {
            **info,
            "applied": False,
            "reason": "component_integrity_regression",
        }
    return candidate, info


# pipeline_core functions resolve these names from their own module globals at
# call time; install the policy hooks before exporting run_pipeline. In
# particular, the worker entrypoint is patched so spawn/forkserver children
# import app.pipeline and install this same policy before candidate work starts.
_core.select_best = select_best
_core.produce_candidate = produce_candidate
_core._produce_and_score_job = _produce_and_score_job
_core._apply_editability_preference = _apply_editability_preference
_core.refine_best = refine_best
_core._refit_one = _refit_one
_core._apply_boundary_refit = _apply_boundary_refit

run_pipeline = _core.run_pipeline
WorkerFailure = _core.WorkerFailure


def __getattr__(name: str) -> Any:
    return getattr(_core, name)
