from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from app import pipeline
from app.transform_journal import TransformJournal


def _candidate(name: str, fidelity: float, *, safe: bool = True, eligible: bool = True):
    return {
        "name": name,
        "fidelity_score": fidelity,
        "total_score": fidelity,
        "selection_safe": safe,
        "selection_disqualified": not safe,
        "final_eligible": eligible,
        "score_details": {},
    }


def test_source_palette_exact_dominates_lower_fidelity_safe_legacy():
    legacy = _candidate("geo_standard", 98.49)
    exact = _candidate("source_palette_exact", 100.0)
    with patch.object(pipeline, "_base_select_best", return_value=(legacy, legacy, "highest_total_score")):
        chosen, raw, reason = pipeline.select_best([legacy, exact], "geometric_logo")
    assert chosen is exact
    assert raw is legacy
    assert reason == "source_palette_exact_dominance"


def test_nonexact_source_palette_keeps_safe_legacy_ranking():
    legacy = _candidate("geo_standard", 98.49)
    approximate = _candidate("source_palette_exact", 99.5)
    with patch.object(pipeline, "_base_select_best", return_value=(legacy, legacy, "highest_total_score")):
        chosen, raw, reason = pipeline.select_best([legacy, approximate], "geometric_logo")
    assert chosen is legacy
    assert raw is legacy
    assert reason == "highest_total_score"


def test_gradient_plus_alpha_downstream_reuses_required_render_for_alpha_proof(monkeypatch):
    import app.fidelity as fidelity
    import app.source_truth as source_truth

    source = np.full((128, 128, 3), 255, dtype=np.uint8)
    source[24:104, 24:104] = (227, 0, 11)

    def rgba_render(_path: Path, width: int, height: int) -> np.ndarray:
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = source[:height, :width]
        rgba[:, :, 3] = 255
        return rgba

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", rgba_render)
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><rect x="24" y="24" width="80" height="80" fill="#e3000b"/></svg>'
    with TemporaryDirectory() as td:
        root = Path(td)
        parent = root / "p.svg"
        candidate = root / "c.svg"
        parent.write_bytes(svg)
        candidate.write_bytes(svg.replace(b"</svg>", b"<metadata>same</metadata></svg>"))
        journal = TransformJournal(parent, source, required_metrics={"gradient_fidelity", "alpha_fidelity"})
        accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)
    assert accepted == candidate
    assert stage["required_unmeasured"] == []
    assert stage["render_comparison"] is not None
    assert stage["alpha_comparison"] is not None


def test_alpha_only_downstream_remains_fail_closed():
    source = np.full((128, 128, 3), 255, dtype=np.uint8)
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><rect x="24" y="24" width="80" height="80" fill="#e3000b" fill-opacity="0.5"/></svg>'
    with TemporaryDirectory() as td:
        root = Path(td)
        parent = root / "p.svg"
        candidate = root / "c.svg"
        parent.write_bytes(svg)
        candidate.write_bytes(svg.replace(b"</svg>", b"<metadata>same</metadata></svg>"))
        journal = TransformJournal(parent, source, required_metrics={"alpha_fidelity"})
        accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)
    assert accepted == parent
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert stage["alpha_comparison"] is None
