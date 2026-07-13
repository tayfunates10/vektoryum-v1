from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def _candidate(
    name: str,
    *,
    engine: str = "vtracer",
    fidelity: float = 95.0,
    edge: float = 0.95,
    paths: int = 100,
    total: float = 90.0,
    rendered: bool = True,
    hard: list[str] | None = None,
    unmeasured: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "engine": engine,
        "fidelity_score": fidelity,
        "total_score": total,
        "rendered_ok": rendered,
        "score_details": {"edge_f1": edge, "path_count": paths},
        "final_artifact_report": {
            "hard_fail_codes": hard or [],
            "unmeasured_required": unmeasured or [],
        },
    }


def test_fail_closed_rejects_hard_fail_and_unmeasured_candidates() -> None:
    from app.multi_hypothesis import select_shadow_hypothesis

    result = select_shadow_hypothesis([
        _candidate("bad_alpha", fidelity=99.0, hard=["alpha_iou_below_min"]),
        _candidate("unknown_metric", fidelity=98.0, unmeasured=["alpha_fidelity"]),
        _candidate("good", fidelity=94.0),
    ])
    assert result["winner"] == "good"
    rejected = {item["name"]: item["reasons"] for item in result["rejected"]}
    assert "alpha_iou_below_min" in rejected["bad_alpha"]
    assert "unmeasured:alpha_fidelity" in rejected["unknown_metric"]


def test_near_equal_quality_prefers_materially_simpler_candidate() -> None:
    from app.multi_hypothesis import select_shadow_hypothesis

    result = select_shadow_hypothesis([
        _candidate("detail", fidelity=97.0, edge=0.98, paths=1000),
        _candidate("lean", fidelity=96.2, edge=0.97, paths=400),
    ])
    assert result["winner"] == "lean"
    assert result["reason"] == "safe_editability_preference"


def test_edge_loss_blocks_simpler_candidate() -> None:
    from app.multi_hypothesis import select_shadow_hypothesis

    result = select_shadow_hypothesis([
        _candidate("detail", fidelity=97.0, edge=0.98, paths=1000),
        _candidate("broken_lean", fidelity=96.8, edge=0.90, paths=300),
    ])
    assert result["winner"] == "detail"


def test_family_classification_and_pareto_report_are_deterministic() -> None:
    from app.multi_hypothesis import select_shadow_hypothesis

    candidates = [
        _candidate("region", engine="vtracer", fidelity=95.0, paths=500),
        _candidate("gradient", engine="gradient", fidelity=96.0, paths=20),
        _candidate("centerline_clean", engine="autotrace", fidelity=94.0, paths=80),
    ]
    first = select_shadow_hypothesis(candidates)
    second = select_shadow_hypothesis(list(reversed(candidates)))
    assert first["winner"] == second["winner"] == "gradient"
    assert first["families_seen"] == ["centerline", "gradient", "region"]
    assert first["family_best"] == {
        "centerline": "centerline_clean",
        "gradient": "gradient",
        "region": "region",
    }


def test_no_eligible_candidate_returns_explicit_fail_closed_status() -> None:
    from app.multi_hypothesis import select_shadow_hypothesis

    result = select_shadow_hypothesis([
        _candidate("render_fail", rendered=False),
        _candidate("low", fidelity=50.0),
    ])
    assert result["status"] == "no_eligible_hypothesis"
    assert result["winner"] is None
    assert result["reason"] == "fail_closed"
