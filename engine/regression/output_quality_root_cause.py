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
    "shortlist_policy",
    "status",
    "shortlist_count",
    "trial_count",
    "chosen_candidate_name",
)
_ALPHA_SHORTLIST_FIELDS = (
    "shortlist_index",
    "candidate_name",
    "candidate_engine",
    "fidelity_score_before_alpha",
    "edge_f1_before_alpha",
    "path_count_before_alpha",
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
_ALPHA_FAILURE_FIELDS = (
    "stage",
    "shortlist_index",
    "candidate_name",
    "candidate_engine",
    "fidelity_score_before_alpha",
    "edge_f1_before_alpha",
    "path_count_before_alpha",
    "error_class",
    "error_message",
)


def _safe_list(value: object, fields: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        snapshot = _base._safe_mapping(item, fields)
        if snapshot:
            output.append(snapshot)
        if len(output) == limit:
            break
    return output


def _safe_alpha_parent_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = _base._safe_mapping(value, _ALPHA_PARENT_FIELDS)
    result["shortlist"] = _safe_list(value.get("shortlist"), _ALPHA_SHORTLIST_FIELDS)
    result["trials"] = _safe_list(value.get("trials"), _ALPHA_TRIAL_FIELDS)
    result["failures"] = _safe_list(value.get("failures"), _ALPHA_FAILURE_FIELDS)
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


if __name__ == "__main__":
    _base.main()
