from __future__ import annotations

import copy
from pathlib import Path


def _candidate(tmp_path: Path, name: str, fidelity: float, paths: int, edge: float = 0.99) -> dict:
    svg = tmp_path / f"{name}.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 0L1 1Z"/></svg>', encoding="utf-8")
    return {
        "name": name,
        "engine": "vtracer",
        "svg_path": svg,
        "fidelity_score": fidelity,
        "total_score": fidelity,
        "rendered_ok": True,
        "score_details": {"path_count": paths, "edge_f1": edge},
    }


def test_shadow_telemetry_does_not_mutate_pipeline_result(tmp_path: Path) -> None:
    from app.shadow_telemetry import build_shadow_telemetry

    production = _candidate(tmp_path, "production", 98.0, 100)
    lean = _candidate(tmp_path, "lean", 97.4, 40)
    pipe = {
        "mode_used": "logo_color",
        "selection_reason": "highest_fidelity",
        "best": production,
        "scored": [production, lean],
    }
    before = copy.deepcopy(pipe)
    report = build_shadow_telemetry(pipe)
    assert pipe == before
    assert report["production_winner"]["name"] == "production"
    assert report["shadow_winner"]["name"] == "lean"
    assert report["decision"] == "disagreement"
    assert report["same_winner"] is False


def test_shadow_telemetry_agreement_is_explicit(tmp_path: Path) -> None:
    from app.shadow_telemetry import build_shadow_telemetry

    winner = _candidate(tmp_path, "winner", 99.0, 30)
    pipe = {
        "mode_used": "geometric_logo",
        "selection_reason": "highest_fidelity",
        "best": winner,
        "scored": [winner],
    }
    report = build_shadow_telemetry(pipe)
    assert report["decision"] == "agreement"
    assert report["same_winner"] is True
    assert report["production_winner"]["svg_sha256"]


def test_shadow_no_winner_is_fail_closed(tmp_path: Path) -> None:
    from app.shadow_telemetry import build_shadow_telemetry

    failed = _candidate(tmp_path, "failed", 99.0, 20)
    failed["rendered_ok"] = False
    pipe = {
        "mode_used": "logo_color",
        "selection_reason": "highest_total_score",
        "best": failed,
        "scored": [failed],
    }
    report = build_shadow_telemetry(pipe)
    assert report["decision"] == "shadow_no_winner"
    assert report["shadow_winner"] is None
    assert report["shadow_report"]["status"] == "no_eligible_hypothesis"


def test_jsonl_append_is_deterministic(tmp_path: Path) -> None:
    from app.shadow_telemetry import append_shadow_telemetry

    target = tmp_path / "audit" / "shadow.jsonl"
    record = {"z": 1, "a": "test"}
    append_shadow_telemetry(target, record)
    append_shadow_telemetry(target, record)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == lines[1]
    assert lines[0].startswith('{"a"')
