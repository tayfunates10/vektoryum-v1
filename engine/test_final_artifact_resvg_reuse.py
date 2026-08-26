from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import app.final_artifact_evaluator as evaluator
from app.fidelity import _rgb_on_white
from app.source_truth import composite_rgba


def _png_bytes(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def test_resvg_rgba_white_composite_matches_fidelity_exactly() -> None:
    values = np.arange(256, dtype=np.uint8)
    channel, alpha = np.meshgrid(values, values)
    rgba = np.stack([channel, channel, channel, alpha], axis=2)

    expected = _rgb_on_white(Image.fromarray(rgba, mode="RGBA"))
    actual = composite_rgba(rgba, 255)

    assert np.array_equal(actual, expected)


def test_primary_resvg_render_returns_rgba_and_same_white_base(monkeypatch) -> None:
    rgba = np.asarray(
        [
            [[10, 20, 30, 0], [40, 50, 60, 64]],
            [[70, 80, 90, 128], [100, 110, 120, 255]],
        ],
        dtype=np.uint8,
    )
    calls: list[tuple[str, int, int]] = []

    def svg_to_bytes(*, svg_path: str, width: int, height: int) -> bytes:
        calls.append((svg_path, width, height))
        return _png_bytes(rgba)

    monkeypatch.setitem(sys.modules, "resvg_py", SimpleNamespace(svg_to_bytes=svg_to_bytes))

    rendered, parity_base = evaluator._render_rgba_with_resvg_base(Path("candidate.svg"), 2, 2)

    assert calls == [("candidate.svg", 2, 2)]
    assert rendered is not None
    assert parity_base is not None
    assert np.array_equal(rendered, rgba)
    assert np.array_equal(parity_base, composite_rgba(rgba, 255))


def test_cross_renderer_parity_reuses_precomputed_resvg_base(monkeypatch) -> None:
    base = np.full((3, 4, 3), 127, dtype=np.uint8)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("precomputed resvg base must avoid another resvg render")

    monkeypatch.setattr(evaluator, "_render_resvg_py", unexpected)
    monkeypatch.setattr(evaluator, "_render_resvg", unexpected)
    monkeypatch.setattr(evaluator, "_render_cairosvg", lambda *_args, **_kwargs: base.copy())
    monkeypatch.setattr(evaluator, "_render_svglib", lambda *_args, **_kwargs: base.copy())

    report = evaluator._cross_renderer_parity(
        Path("candidate.svg"), 4, 3, resvg_base=base,
    )

    assert report == {"resvg": True, "cairosvg": 0.0, "svglib": 0.0}


def test_resvg_failure_preserves_existing_fallback_paths(monkeypatch) -> None:
    fallback_rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    fallback_rgb = np.full((2, 2, 3), 200, dtype=np.uint8)

    def fail_svg_to_bytes(**_kwargs):
        raise RuntimeError("resvg unavailable")

    monkeypatch.setitem(sys.modules, "resvg_py", SimpleNamespace(svg_to_bytes=fail_svg_to_bytes))
    monkeypatch.setattr(evaluator, "render_svg_to_rgba", lambda *_args, **_kwargs: fallback_rgba.copy())

    rendered, parity_base = evaluator._render_rgba_with_resvg_base(Path("candidate.svg"), 2, 2)
    assert np.array_equal(rendered, fallback_rgba)
    assert parity_base is None

    calls: list[str] = []

    def resvg_py_fallback(*_args, **_kwargs):
        calls.append("resvg_py")
        return None

    def resvg_cli_fallback(*_args, **_kwargs):
        calls.append("resvg_cli")
        return fallback_rgb.copy()

    monkeypatch.setattr(evaluator, "_render_resvg_py", resvg_py_fallback)
    monkeypatch.setattr(evaluator, "_render_resvg", resvg_cli_fallback)
    monkeypatch.setattr(evaluator, "_render_cairosvg", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator, "_render_svglib", lambda *_args, **_kwargs: None)

    report = evaluator._cross_renderer_parity(Path("candidate.svg"), 2, 2)

    assert calls == ["resvg_py", "resvg_cli"]
    assert report == {"resvg": True, "cairosvg": None, "svglib": None}
