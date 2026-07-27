"""Compatibility wrapper adding alpha-parent diagnostics to QA evidence."""
from __future__ import annotations

from typing import Any

from engine.regression import output_quality_root_cause_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_original_pipeline_snapshot = _base.pipeline_snapshot

_ALPHA_PARENT_FIELDS = (
    "schema",
    "trial_count",
    "chosen_candidate_name",
)
_ALPHA_TRIAL_FIELDS = (
    "trial_index",
    "candidate_name",
    "candidate_engine",
    "rendered_ok",
    "fidelity_score",
    "edge_f1",
    "path_count",
    "byte_size",
)


def _safe_alpha_parent_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = _base._safe_mapping(value, _ALPHA_PARENT_FIELDS)
    trials: list[dict[str, Any]] = []
    for trial in value.get("trials") or []:
        snapshot = _base._safe_mapping(trial, _ALPHA_TRIAL_FIELDS)
        if snapshot:
            trials.append(snapshot)
    result["trials"] = trials[:8]
    return result


def pipeline_snapshot(output: object) -> dict[str, Any]:
    snapshot = _original_pipeline_snapshot(output)
    if isinstance(output, dict):
        alpha_parent = _safe_alpha_parent_diagnostics(
            output.get("alpha_parent_trial_diagnostics")
        )
        if alpha_parent:
            snapshot["alpha_parent_trials"] = alpha_parent
        alpha_selection = output.get("alpha_parent_selection")
        if isinstance(alpha_selection, dict):
            snapshot["alpha_parent_selection"] = _base._safe_mapping(
                alpha_selection,
                (
                    "status",
                    "reason",
                    "source_alpha_level_count",
                    "original_candidate_name",
                    "proposed_candidate_name",
                    "selected_candidate_name",
                    "trial_count",
                    "selected_fidelity_score",
                    "selected_edge_f1",
                    "selected_path_count_after_alpha",
                    "selected_byte_size_after_alpha",
                    "fidelity_margin",
                    "edge_f1_margin",
                ),
            )
    return snapshot


_base.pipeline_snapshot = pipeline_snapshot
