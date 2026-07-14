from __future__ import annotations

from pathlib import Path

from app.quality import basic_svg_quality_check
from app.scoring import _parse_svg_stats
from core_release_contract import PRODUCTION_MODES
from core_release_runner import _fixture, _has_open_required_cycle, _score_snapshot_match


def test_every_explicit_mode_has_a_deterministic_fixture(tmp_path) -> None:
    for mode in PRODUCTION_MODES:
        first = tmp_path / f"{mode}-1.png"
        second = tmp_path / f"{mode}-2.png"
        a = _fixture(mode, first)
        b = _fixture(mode, second)
        assert a["source_sha256"] == b["source_sha256"]
        assert first.read_bytes() == second.read_bytes()
        assert a["palette"]
        assert a["image_class"]


def test_open_filled_cycle_is_rejected_but_open_stroke_is_allowed(tmp_path) -> None:
    filled = tmp_path / "filled.svg"
    filled.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path fill="#000" d="M 1 1 L 9 1 L 9 9"/>'
        "</svg>",
        encoding="utf-8",
    )
    assert _has_open_required_cycle(filled) is True

    stroke = tmp_path / "stroke.svg"
    stroke.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path fill="none" stroke="#000" d="M 1 1 L 9 9"/>'
        "</svg>",
        encoding="utf-8",
    )
    assert _has_open_required_cycle(stroke) is False


def test_score_snapshot_must_match_the_exact_svg(tmp_path) -> None:
    svg = tmp_path / "artifact.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path fill="#000000" d="M 1 1 L 9 1 L 9 9 L 1 9 Z"/>'
        "</svg>",
        encoding="utf-8",
    )
    stats = _parse_svg_stats(svg)
    assert _score_snapshot_match({"score_details": stats}, svg) is True
    stale = dict(stats)
    stale["path_count"] += 1
    assert _score_snapshot_match({"score_details": stale}, svg) is False


def test_photo_poster_is_always_needs_review_even_with_high_scores() -> None:
    report = basic_svg_quality_check(
        score_details={
            "path_count": 800,
            "node_count": 6000,
            "unique_colors": 48,
            "has_bitmap": False,
        },
        mode="photo_poster",
        total_score=99.0,
        fidelity_score=99.0,
        structure_report={
            "ink_recall": 1.0,
            "ink_precision": 1.0,
            "components_original": 4,
            "components_rendered": 4,
            "component_delta": 0,
        },
    )
    assert report["status"] == "needs_review"
    assert any("accepted product limit" in warning for warning in report["warnings"])
