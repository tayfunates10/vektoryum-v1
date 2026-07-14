from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.main import ALLOWED_MODES
import app.analyzer as analyzer


ENGINE_DIR = Path(__file__).resolve().parent
ROADMAP_PATH = ENGINE_DIR / "ai_analyzer_roadmap.json"
EXPECTED_PHASES = ["AA-1", "AA-2", "AA-3", "AA-4"]
EXPECTED_AUTO_MODES = {
    "geometric_logo",
    "minimal_ai",
    "logo_color",
    "single_color",
    "lineart",
    "photo_poster",
}
EXPECTED_EXPLICIT_ONLY = {"flat_logo", "centerline"}
EXPECTED_LIMITATIONS = {
    "missing_confidence_score",
    "unversioned_feature_thresholds",
    "no_fail_closed_abstention_contract",
    "no_labeled_analyzer_release_corpus",
}


def _roadmap() -> dict:
    return json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))


def _geometric_logo() -> Image.Image:
    width, height = 800, 600
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    arr[60:540, 60:90] = 0
    arr[60:540, 710:740] = 0
    arr[60:90, 60:740] = 0
    arr[510:540, 60:740] = 0
    arr[160:360, 180:420] = (255, 0, 0)
    arr[160:440, 470:660] = 0
    return Image.fromarray(arr, "RGB")


def _color_logo() -> Image.Image:
    width, height = 800, 600
    yy, xx = np.mgrid[0:height, 0:width]
    red = (xx / width * 255).astype(np.uint8)
    green = (yy / height * 255).astype(np.uint8)
    blue = ((xx + yy) / (width + height) * 255).astype(np.uint8)
    arr = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    palette = [
        (220, 30, 30),
        (30, 160, 60),
        (40, 80, 200),
        (240, 200, 20),
        (150, 40, 160),
        (240, 130, 20),
    ]
    for index, color in enumerate(palette):
        cy = 150 + (index % 2) * 250
        cx = 120 + (index % 3) * 250
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 90**2
        arr[mask] = color
    return Image.fromarray(arr, "RGB")


def test_roadmap_is_finite_ordered_and_schema_pinned() -> None:
    data = _roadmap()
    assert data["schema_version"] == "ai-analyzer-closure-v1"
    phases = data["phases"]
    assert [phase["id"] for phase in phases] == EXPECTED_PHASES
    assert [phase["status"] for phase in phases] == [
        "complete",
        "pending",
        "pending",
        "pending",
    ]
    assert len({phase["id"] for phase in phases}) == len(phases)
    assert all(len(phase["acceptance_criteria"]) >= 4 for phase in phases)


def test_mode_sets_partition_public_non_auto_modes() -> None:
    data = _roadmap()
    public = set(data["public_trace_modes"])
    auto_modes = set(data["auto_recommendation_modes"])
    explicit_only = set(data["explicit_only_modes"])

    assert public == set(ALLOWED_MODES)
    assert auto_modes == EXPECTED_AUTO_MODES
    assert explicit_only == EXPECTED_EXPLICIT_ONLY
    assert "auto" not in auto_modes
    assert "auto" not in explicit_only
    assert auto_modes.isdisjoint(explicit_only)
    assert auto_modes | explicit_only == public - {"auto"}


def test_known_limitations_have_one_pending_closure_phase() -> None:
    data = _roadmap()
    phases = {phase["id"]: phase for phase in data["phases"]}
    limitations = data["known_limitations"]

    assert {item["id"] for item in limitations} == EXPECTED_LIMITATIONS
    assert len(limitations) == len({item["id"] for item in limitations})
    for item in limitations:
        assert item["status"] == "open"
        assert phases[item["closure_phase"]]["status"] == "pending"
        assert (ENGINE_DIR / item["evidence"]).is_file()


def test_completed_phase_evidence_exists_and_is_not_self_declared_only() -> None:
    completed = [phase for phase in _roadmap()["phases"] if phase["status"] == "complete"]
    assert [phase["id"] for phase in completed] == ["AA-1"]
    evidence = completed[0]["evidence"]
    assert "test_ai_analyzer_roadmap.py" in evidence
    assert len(evidence) >= 4
    for relative in evidence:
        assert (ENGINE_DIR / relative).is_file(), relative


def test_public_report_contract_and_seed_decisions_are_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(analyzer, "calculate_semantic_edge_stats", lambda _image: None)
    required_fields = set(_roadmap()["public_report_fields"])

    geometric_first = analyzer.analyze_image_from_mem(_geometric_logo())
    geometric_second = analyzer.analyze_image_from_mem(_geometric_logo())
    color = analyzer.analyze_image_from_mem(_color_logo())

    assert required_fields <= set(geometric_first)
    assert required_fields <= set(color)
    assert {key: geometric_first[key] for key in required_fields} == {
        key: geometric_second[key] for key in required_fields
    }
    assert geometric_first["recommended_mode"] == "geometric_logo"
    assert color["recommended_mode"] == "logo_color"
    assert geometric_first["detected_type"] == geometric_first["recommended_mode"]
    assert color["detected_type"] == color["recommended_mode"]
    assert geometric_first["recommended_mode"] in EXPECTED_AUTO_MODES
    assert color["recommended_mode"] in EXPECTED_AUTO_MODES


def test_scope_keeps_core_engine_and_saas_closure_separate() -> None:
    excluded = " ".join(_roadmap()["scope"]["excluded"]).lower()
    assert "core vector engine roadmap" in excluded
    assert "billing" in excluded
    assert "authentication" in excluded
