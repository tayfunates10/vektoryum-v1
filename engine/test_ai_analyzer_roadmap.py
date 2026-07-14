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
EXPECTED_DECISION_FIELDS = {
    "schema_version",
    "status",
    "requested_mode",
    "recommended_mode",
    "execution_mode",
    "abstained",
    "fallback_applied",
    "reason_codes",
    "confidence",
    "runner_up_mode",
    "runner_up_margin",
    "verified_recommendation_digest",
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


def test_phase_order_and_status() -> None:
    data = _roadmap()
    assert data["schema_version"] == "ai-analyzer-closure-v1"
    phases = data["phases"]
    assert [phase["id"] for phase in phases] == EXPECTED_PHASES
    assert [phase["status"] for phase in phases] == [
        "complete",
        "complete",
        "complete",
        "pending",
    ]
    assert len({phase["id"] for phase in phases}) == len(phases)
    assert all(len(phase["acceptance_criteria"]) >= 4 for phase in phases)


def test_mode_sets() -> None:
    data = _roadmap()
    public = set(data["public_trace_modes"])
    automatic = set(data["auto_recommendation_modes"])
    manual = set(data["explicit_only_modes"])
    assert public == set(ALLOWED_MODES)
    assert automatic == EXPECTED_AUTO_MODES
    assert manual == EXPECTED_EXPLICIT_ONLY
    assert automatic.isdisjoint(manual)
    assert automatic | manual == public - {"auto"}
    assert set(data["auto_decision_fields"]) == EXPECTED_DECISION_FIELDS


def test_limitation_status_matches_phase() -> None:
    data = _roadmap()
    phases = {phase["id"]: phase for phase in data["phases"]}
    limitations = data["known_limitations"]
    assert {item["id"] for item in limitations} == EXPECTED_LIMITATIONS
    assert len(limitations) == len({item["id"] for item in limitations})
    for item in limitations:
        assert item["status"] in {"open", "closed"}
        assert item["closure_phase"] in phases
        assert (ENGINE_DIR / item["evidence"]).is_file()
        expected = "pending" if item["status"] == "open" else "complete"
        assert phases[item["closure_phase"]]["status"] == expected


def test_completed_evidence_files() -> None:
    completed = [phase for phase in _roadmap()["phases"] if phase["status"] == "complete"]
    assert [phase["id"] for phase in completed] == ["AA-1", "AA-2", "AA-3"]
    for phase in completed:
        assert "test_ai_analyzer_roadmap.py" in phase["evidence"]
        assert len(phase["evidence"]) >= 4
        for relative in phase["evidence"]:
            assert (ENGINE_DIR / relative).is_file()


def test_public_fields_and_seed_decisions(monkeypatch) -> None:
    monkeypatch.setattr(analyzer, "calculate_semantic_edge_stats", lambda _image: None)
    required = set(_roadmap()["public_report_fields"])
    first = analyzer.analyze_image_from_mem(_geometric_logo())
    second = analyzer.analyze_image_from_mem(_geometric_logo())
    color = analyzer.analyze_image_from_mem(_color_logo())
    assert required <= set(first)
    assert required <= set(color)
    assert {key: first[key] for key in required} == {key: second[key] for key in required}
    assert first["recommended_mode"] == "geometric_logo"
    assert color["recommended_mode"] == "logo_color"
    assert first["detected_type"] == first["recommended_mode"]
    assert color["detected_type"] == color["recommended_mode"]
    assert first["analyzer_contract"]["status"] == "valid"
    assert color["analyzer_contract"]["status"] == "valid"


def test_scope_separation() -> None:
    excluded = " ".join(_roadmap()["scope"]["excluded"]).lower()
    assert "core vector engine roadmap" in excluded
    assert "billing" in excluded
    assert "authentication" in excluded
