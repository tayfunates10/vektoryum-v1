from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import source_truth
from app.fidelity import score_svg_fidelity
from app.refine_cache import RefinementCache
from app.transform_journal import _measure_svg_bytes


def _fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    side = 32
    source = np.full((side, side, 3), 255, dtype=np.uint8)
    source[8:24, 8:24] = 0
    original = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(original)
    svg = tmp_path / "candidate.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
        'viewBox="0 0 32 32"><rect width="32" height="32" fill="#fff"/>'
        '<rect x="8" y="8" width="16" height="16" fill="#000"/></svg>',
        encoding="utf-8",
    )
    return original, svg, source


def test_scoring_and_journal_share_only_the_same_coarse_rgb_render(tmp_path: Path) -> None:
    original, svg, source = _fixture(tmp_path)
    calls: list[tuple[int, int]] = []
    cache = RefinementCache(source, max_renders=2)

    def real_render(_path: Path, width: int, height: int) -> np.ndarray:
        calls.append((width, height))
        if (width, height) == (source.shape[1], source.shape[0]):
            return source.copy()
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)

    cache._real_render = real_render
    try:
        fidelity = score_svg_fidelity(svg, original, render_fn=cache.render)
        assert fidelity is not None
        metric = _measure_svg_bytes(
            svg.read_bytes(), source, max_side=512, render_fn=cache.render,
        )
        assert "ssim" in metric
        assert "seam_ratio" in metric
        assert "component_delta" in metric
        assert cache.render_calls == 2
        assert cache.render_misses == 1
        assert cache.render_hits == 1
        assert calls == [(32, 32)]
    finally:
        cache.close()


def test_alpha_required_measurement_keeps_rgba_proof_outside_rgb_cache(
    tmp_path: Path, monkeypatch,
) -> None:
    _original, svg, source = _fixture(tmp_path)
    rgba = np.dstack((source, np.full(source.shape[:2], 255, dtype=np.uint8)))
    rgb_calls = 0

    def rgba_render(_path: Path, _width: int, _height: int) -> np.ndarray:
        return rgba.copy()

    def cached_rgb(_path: Path, _width: int, _height: int) -> np.ndarray:
        nonlocal rgb_calls
        rgb_calls += 1
        return source.copy()

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", rgba_render)
    metric = _measure_svg_bytes(
        svg.read_bytes(),
        source,
        max_side=512,
        required_metrics={"alpha_fidelity"},
        measure_alpha=True,
        render_fn=cached_rgb,
    )
    assert metric["alpha_fidelity_status"] == "measured"
    assert "alpha_fidelity" not in metric["required_unmeasured"]
    assert rgb_calls == 0
