from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from app.component_quality import has_open_required_cycle  # noqa: E402
from app.pipeline import _FillCycleJournal  # noqa: E402
from app.svg_lifecycle import close_implicit_fill_cycles  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fill_only_open_subpaths_are_closed_byte_minimally(tmp_path: Path) -> None:
    svg = tmp_path / "fill.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path fill="#000" d="M2 2L18 2L18 18L2 18 M5 5L5 15L15 15L15 5"/>'
        '</svg>',
        encoding="utf-8",
    )
    before = svg.read_text(encoding="utf-8")
    assert has_open_required_cycle(svg) is True
    report = close_implicit_fill_cycles(svg)
    after = svg.read_text(encoding="utf-8")
    assert report == {
        "applied": True,
        "reason": "explicit_fill_cycle_normalization",
        "paths_changed": 1,
        "subpaths_closed": 2,
    }
    assert after.count("Z") - before.count("Z") == 2
    assert len(after.encode("utf-8")) == len(before.encode("utf-8")) + 2
    assert has_open_required_cycle(svg) is False


def test_visible_stroke_path_is_never_closed_or_rewritten(tmp_path: Path) -> None:
    svg = tmp_path / "stroke.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path fill="#000" stroke="#f00" d="M2 2L18 2L18 18"/>'
        '</svg>',
        encoding="utf-8",
    )
    before_sha = _sha(svg)
    report = close_implicit_fill_cycles(svg)
    assert report["applied"] is False
    assert report["reason"] == "no_implicit_fill_cycles"
    assert _sha(svg) == before_sha


def test_already_closed_fill_is_idempotent(tmp_path: Path) -> None:
    svg = tmp_path / "closed.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path fill="#000" d="M2 2L18 2L18 18Z"/>'
        '</svg>',
        encoding="utf-8",
    )
    before_sha = _sha(svg)
    first = close_implicit_fill_cycles(svg)
    second = close_implicit_fill_cycles(svg)
    assert first["applied"] is False
    assert second["applied"] is False
    assert _sha(svg) == before_sha


def test_exact_render_fill_cycle_repair_is_journal_accepted(tmp_path: Path) -> None:
    """Implicit-vs-explicit fill closure may repair structure without changing pixels."""
    svg = tmp_path / "implicit.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">'
        '<path fill="#000" d="M2 2L18 2L18 18L2 18"/>'
        '</svg>',
        encoding="utf-8",
    )
    # Deliberately unrelated source pixels: acceptance must be based on exact
    # parent/candidate render preservation plus structural repair, not on a
    # newly invented source-fidelity threshold.
    source_rgb = np.full((20, 20, 3), 127, dtype=np.uint8)
    journal = _FillCycleJournal.build(
        svg,
        source_rgb,
        image_class="clean_logo",
        required_metrics=set(),
    )
    accepted, report, stage = journal.run_in_place(
        "explicit_fill_cycle_normalization",
        svg,
        close_implicit_fill_cycles,
    )
    assert accepted is True
    assert report["applied"] is True
    assert stage["status"] == "accepted"
    assert stage["reason_codes"] == ["metrics_non_regressing"]
    assert stage["before_metrics"]["render_rgb_sha256"] == stage["after_metrics"]["render_rgb_sha256"]
    assert has_open_required_cycle(svg) is False
