from __future__ import annotations

import json
from pathlib import Path

from app.main import ALLOWED_MODES
from app.vector_engines import build_vector_candidates


ENGINE_DIR = Path(__file__).resolve().parent
ROADMAP_PATH = ENGINE_DIR / "core_vector_engine_roadmap.json"
EXPECTED_PHASES = ["CVE-1", "CVE-2", "CVE-3", "CVE-4"]
EXPECTED_LIMITATIONS = {
    "centerline_placeholder",
    "polygon_flattening_cutouts",
    "photo_fidelity_ceiling",
    "optional_render_backends",
}
VALID_PHASE_STATUS = {"complete", "pending"}
VALID_LIMIT_STATUS = {
    "open",
    "accepted_product_limit",
    "accepted_with_fail_closed_fallback",
}


def _roadmap() -> dict:
    return json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))


def test_roadmap_is_finite_ordered_and_schema_pinned() -> None:
    data = _roadmap()

    assert data["schema_version"] == "core-vector-engine-closure-v1"
    phases = data["phases"]
    assert [phase["id"] for phase in phases] == EXPECTED_PHASES
    assert len({phase["id"] for phase in phases}) == len(phases)
    assert all(phase["status"] in VALID_PHASE_STATUS for phase in phases)

    statuses = [phase["status"] for phase in phases]
    assert statuses == ["complete", "pending", "pending", "pending"]
    assert all(len(phase["acceptance_criteria"]) >= 4 for phase in phases)
    assert all(all(isinstance(item, str) and item.strip() for item in phase["acceptance_criteria"])
               for phase in phases)


def test_completed_phase_evidence_exists_and_is_not_self_declared_only() -> None:
    completed = [phase for phase in _roadmap()["phases"] if phase["status"] == "complete"]
    assert [phase["id"] for phase in completed] == ["CVE-1"]

    for phase in completed:
        evidence = phase["evidence"]
        assert len(evidence) >= 3
        assert "test_core_vector_engine_roadmap.py" in evidence
        for relative in evidence:
            path = ENGINE_DIR / relative
            assert path.is_file(), f"missing evidence for {phase['id']}: {relative}"


def test_manifest_modes_match_public_production_modes() -> None:
    data = _roadmap()
    declared = data["production_modes"]

    assert len(declared) == len(set(declared))
    assert set(declared) == set(ALLOWED_MODES) - {"auto"}
    assert "auto" not in declared


def test_every_mode_has_a_known_non_optional_candidate() -> None:
    data = _roadmap()
    known_engines = set(data["known_engines"])

    for mode in data["production_modes"]:
        candidates = build_vector_candidates(mode)
        assert candidates, f"{mode}: empty candidate plan"
        assert len(candidates) == len(set(candidates)), f"{mode}: duplicate candidate names"

        mandatory = [spec for spec in candidates.values() if not spec.get("optional", False)]
        assert mandatory, f"{mode}: no non-optional candidate"
        for name, spec in candidates.items():
            assert spec.get("engine") in known_engines, (
                f"{mode}/{name}: unknown engine {spec.get('engine')!r}"
            )


def test_every_known_limitation_has_exactly_one_closure_phase() -> None:
    data = _roadmap()
    phases = {phase["id"]: phase for phase in data["phases"]}
    limitations = data["known_limitations"]

    assert {item["id"] for item in limitations} == EXPECTED_LIMITATIONS
    assert len(limitations) == len({item["id"] for item in limitations})
    for item in limitations:
        assert item["status"] in VALID_LIMIT_STATUS
        assert item["closure_phase"] in phases
        assert (ENGINE_DIR / item["evidence"]).is_file()
        if item["status"] == "open":
            assert phases[item["closure_phase"]]["status"] == "pending"


def test_scope_keeps_ai_analyzer_closure_separate() -> None:
    excluded = " ".join(_roadmap()["scope"]["excluded"]).lower()
    assert "ai analyzer" in excluded
    assert "classification" in excluded
    assert "confidence" in excluded
