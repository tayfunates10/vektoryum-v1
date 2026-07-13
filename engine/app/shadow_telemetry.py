"""FAZ 4.2 shadow telemetry adapter.

Consumes an existing ``run_pipeline`` result and compares the production winner with
the fail-closed multi-hypothesis shadow winner. It never mutates the pipeline result
or changes the exported artifact.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from app.multi_hypothesis import HypothesisPolicy, select_shadow_hypothesis


def _candidate_snapshot(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    details = candidate.get("score_details") or {}
    path = candidate.get("svg_path")
    sha256 = None
    if path:
        try:
            sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            sha256 = None
    return {
        "name": candidate.get("name"),
        "engine": candidate.get("engine"),
        "fidelity_score": candidate.get("fidelity_score"),
        "total_score": candidate.get("total_score"),
        "edge_f1": details.get("edge_f1"),
        "path_count": details.get("path_count"),
        "rendered_ok": bool(candidate.get("rendered_ok")),
        "svg_sha256": sha256,
    }


def build_shadow_telemetry(
    pipeline_result: dict[str, Any],
    policy: HypothesisPolicy | None = None,
) -> dict[str, Any]:
    """Build an auditable shadow comparison without mutating production state."""
    scored = list(pipeline_result.get("scored") or [])
    production = pipeline_result.get("best")
    before = copy.deepcopy(pipeline_result)

    shadow = select_shadow_hypothesis(scored, policy=policy)
    shadow_source = shadow.pop("winner_source", None)
    production_snapshot = _candidate_snapshot(production)
    shadow_snapshot = _candidate_snapshot(shadow_source)

    production_name = production_snapshot.get("name") if production_snapshot else None
    shadow_name = shadow_snapshot.get("name") if shadow_snapshot else None
    same_winner = production_name is not None and production_name == shadow_name

    telemetry = {
        "schema_version": "faz4.2-shadow-v1",
        "mode_used": pipeline_result.get("mode_used"),
        "production_selection_reason": pipeline_result.get("selection_reason"),
        "production_winner": production_snapshot,
        "shadow_winner": shadow_snapshot,
        "same_winner": same_winner,
        "decision": "agreement" if same_winner else (
            "shadow_no_winner" if shadow_name is None else "disagreement"
        ),
        "shadow_report": shadow,
        "candidate_count": len(scored),
    }

    if before != pipeline_result:
        raise RuntimeError("shadow telemetry mutated pipeline_result")
    return telemetry


def append_shadow_telemetry(path: Path, telemetry: dict[str, Any]) -> None:
    """Append one deterministic JSONL record; caller controls persistence policy."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(telemetry, ensure_ascii=False, sort_keys=True) + "\n")
