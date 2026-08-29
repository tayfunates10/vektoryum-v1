from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from app import component_align


def test_component_align_skips_path_parsing_above_budget(monkeypatch, tmp_path: Path) -> None:
    fake_svgpathtools = types.ModuleType("svgpathtools")

    def _unexpected_parse_path(_d: str):
        raise AssertionError("parse_path must not run above component-align complexity budget")

    fake_svgpathtools.parse_path = _unexpected_parse_path  # type: ignore[attr-defined]
    fake_path_mod = types.ModuleType("svgpathtools.path")
    fake_path_mod.transform = lambda path, matrix: path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "svgpathtools", fake_svgpathtools)
    monkeypatch.setitem(sys.modules, "svgpathtools.path", fake_path_mod)

    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(component_align, "load_reference_rgb", lambda *_a, **_k: (reference, (8, 8)))
    monkeypatch.setattr(component_align, "render_svg_to_rgb", lambda *_a, **_k: reference.copy())
    monkeypatch.setattr(
        component_align,
        "_component_class_report",
        lambda *_a, **_k: {
            "component_min_iou": 0.5,
            "weak_components": [{"iou": 0.5, "dx": 1.0, "dy": 1.0, "scale": 1.0, "bbox": [0, 0, 2, 2]}],
        },
    )
    monkeypatch.setattr(component_align, "compute_fidelity", lambda *_a, **_k: {"fidelity_score": 1.0})

    svg_path = tmp_path / "large.svg"
    paths = "".join('<path d="M0 0L1 1"/>' for _ in range(component_align._ALIGN_PARSE_PATH_BUDGET + 1))
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">{paths}</svg>',
        encoding="utf-8",
    )

    report = component_align.apply_component_align(svg_path, tmp_path / "source.png", tmp_path / "out.svg")

    assert report == {
        "applied": False,
        "reason": "component_align_complexity_budget",
        "path_count": component_align._ALIGN_PARSE_PATH_BUDGET + 1,
        "path_budget": component_align._ALIGN_PARSE_PATH_BUDGET,
    }
    assert not (tmp_path / "out.svg").exists()
