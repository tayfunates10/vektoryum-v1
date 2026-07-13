"""FAZ 4.3 feature-flag and production immutability contracts."""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def _candidate(tmp_path: Path, name: str, fidelity: float, paths: int) -> dict:
    svg = tmp_path / f"{name}.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>', encoding="utf-8")
    return {
        "name": name,
        "engine": "vtracer",
        "svg_path": svg,
        "rendered_ok": True,
        "fidelity_score": fidelity,
        "total_score": fidelity,
        "score_details": {"edge_f1": 0.99, "path_count": paths},
    }


def test_flag_defaults_off_and_returns_same_object(tmp_path: Path) -> None:
    from app.shadow_runtime import maybe_attach_shadow_telemetry, shadow_selector_enabled

    best = _candidate(tmp_path, "prod", 96.0, 10)
    pipe = {"best": best, "scored": [best], "mode_used": "logo_color"}
    assert shadow_selector_enabled({}) is False
    result = maybe_attach_shadow_telemetry(pipe, env={})
    assert result is pipe
    assert "shadow_telemetry" not in pipe


def test_flag_on_attaches_report_without_replacing_production_winner(tmp_path: Path) -> None:
    from app.shadow_runtime import maybe_attach_shadow_telemetry

    prod = _candidate(tmp_path, "prod", 96.0, 10)
    shadow = _candidate(tmp_path, "lean", 95.7, 3)
    pipe = {
        "best": prod,
        "scored": [prod, shadow],
        "mode_used": "logo_color",
        "selection_reason": "highest_fidelity",
    }
    result = maybe_attach_shadow_telemetry(pipe, env={"VEKTORYUM_SHADOW_SELECTOR": "on"})
    assert result is not pipe
    assert result["best"] is prod
    assert pipe["best"] is prod
    assert "shadow_telemetry" not in pipe
    assert result["shadow_telemetry"]["production_winner"]["name"] == "prod"
    assert result["shadow_telemetry"]["shadow_winner"]["name"] in {"prod", "lean"}


def test_enabled_runtime_can_append_deterministic_audit(tmp_path: Path) -> None:
    from app.shadow_runtime import maybe_attach_shadow_telemetry

    prod = _candidate(tmp_path, "prod", 97.0, 6)
    pipe = {"best": prod, "scored": [prod], "mode_used": "geometric_logo"}
    audit = tmp_path / "audit" / "shadow.jsonl"
    result = maybe_attach_shadow_telemetry(
        pipe,
        env={"VEKTORYUM_SHADOW_SELECTOR": "true"},
        audit_path=audit,
    )
    assert result["shadow_telemetry"]["decision"] == "agreement"
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"decision": "agreement"' in lines[0]


def test_shadow_error_isolated_from_production(monkeypatch, tmp_path: Path) -> None:
    import app.shadow_runtime as runtime

    prod = _candidate(tmp_path, "prod", 97.0, 6)
    pipe = {"best": prod, "scored": [prod], "mode_used": "logo_color"}

    def explode(_pipeline_result):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "build_shadow_telemetry", explode)
    result = runtime.maybe_attach_shadow_telemetry(
        pipe, env={"VEKTORYUM_SHADOW_SELECTOR": "1"}
    )
    assert result["best"] is prod
    assert result["shadow_telemetry"] == {
        "schema_version": "faz4.3-shadow-runtime-v1",
        "status": "telemetry_error",
        "error_type": "RuntimeError",
    }
