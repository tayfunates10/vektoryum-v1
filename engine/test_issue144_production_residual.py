from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import cairosvg
import numpy as np
import pytest
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from app.pipeline import (  # noqa: E402
    _pareto_front,
    _selection_pool,
    build_vector_candidates,
)
from app.source_palette_vector import (  # noqa: E402
    SourcePaletteNotApplicable,
    vectorize_source_palette_paths,
)


def _render_svg(path: Path, width: int, height: int) -> np.ndarray:
    png = cairosvg.svg2png(url=str(path), output_width=width, output_height=height)
    rendered = Image.open(io.BytesIO(png)).convert("RGBA")
    background = Image.new("RGBA", rendered.size, (255, 255, 255, 255))
    background.alpha_composite(rendered)
    return np.asarray(background.convert("RGB"), dtype=np.uint8)


def _candidate(
    name: str,
    *,
    fidelity: float,
    ssim: float,
    delta: float,
    edge: float,
    recall: float = 1.0,
    precision: float = 1.0,
    min_iou: float = 1.0,
    safe: bool = True,
) -> dict:
    return {
        "name": name,
        "rendered_ok": True,
        "fidelity_score": fidelity,
        "selection_safe": safe,
        "selection_disqualified": not safe,
        "score_details": {
            "ssim": ssim,
            "mean_delta_e": delta,
            "edge_f1": edge,
        },
        "component_quality": {
            "applicable": True,
            "source_cc_recall": recall,
            "render_cc_precision": precision,
            "min_true_cc_iou": min_iou,
        },
    }


def test_exact_palette_candidate_is_real_vector_and_native_pixel_exact(tmp_path: Path) -> None:
    rgb = np.full((24, 24, 3), 255, dtype=np.uint8)
    rgb[2:22, 2:22] = (24, 24, 24)
    rgb[6:18, 6:18] = (235, 235, 235)
    rgb[9:15, 9:15] = (190, 25, 35)
    rgb[11:13, 11:13] = (255, 255, 255)
    source = tmp_path / "source.png"
    Image.fromarray(rgb, mode="RGB").save(source)

    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    report = vectorize_source_palette_paths(source, first, max_colors=6)
    vectorize_source_palette_paths(source, second, max_colors=6)

    text = first.read_text(encoding="utf-8")
    assert report["source_color_count"] == 4
    assert report["palette_cap"] == 6
    assert report["raster_embedded"] is False
    assert report["path_count"] == 4
    assert "<image" not in text.lower()
    assert "base64" not in text.lower()
    assert "data:" not in text.lower()
    assert text.count("<path") == 4
    assert text.count("Z") >= 4
    assert np.array_equal(_render_svg(first, 24, 24), rgb)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_exact_palette_candidate_rejects_alpha_and_over_cap(tmp_path: Path) -> None:
    alpha = np.zeros((8, 8, 4), dtype=np.uint8)
    alpha[:, :, :3] = (20, 20, 20)
    alpha[:, :, 3] = 255
    alpha[0, 0, 3] = 128
    alpha_path = tmp_path / "alpha.png"
    Image.fromarray(alpha, mode="RGBA").save(alpha_path)
    with pytest.raises(SourcePaletteNotApplicable, match="source_has_alpha"):
        vectorize_source_palette_paths(alpha_path, tmp_path / "alpha.svg", max_colors=6)

    palette = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(8):
        palette[index, :, :] = (index * 20, index * 20, index * 20)
    palette_path = tmp_path / "palette.png"
    Image.fromarray(palette, mode="RGB").save(palette_path)
    with pytest.raises(SourcePaletteNotApplicable, match="source_palette_exceeds_cap"):
        vectorize_source_palette_paths(palette_path, tmp_path / "palette.svg", max_colors=6)


def test_source_palette_candidate_family_is_narrowly_scoped() -> None:
    assert "source_palette_exact" in build_vector_candidates("geometric_logo")
    assert "source_palette_exact" in build_vector_candidates("single_color")
    assert "source_palette_exact" not in build_vector_candidates("logo_color")
    assert "source_palette_exact" not in build_vector_candidates("logo_gradient")
    assert "source_palette_exact" not in build_vector_candidates("photo_poster")


def test_safe_pareto_front_removes_fidelity_dominated_candidate() -> None:
    exact = _candidate(
        "exact",
        fidelity=100.0,
        ssim=1.0,
        delta=0.0,
        edge=1.0,
    )
    dominated = _candidate(
        "dominated",
        fidelity=99.0,
        ssim=0.99,
        delta=0.2,
        edge=0.99,
    )
    tradeoff = _candidate(
        "tradeoff",
        fidelity=99.5,
        ssim=1.0,
        delta=0.0,
        edge=0.98,
    )
    front = _pareto_front([dominated, exact, tradeoff])
    assert exact in front
    assert tradeoff in front
    assert dominated not in front


def test_unsafe_high_global_score_cannot_enter_safe_pareto_pool() -> None:
    unsafe = _candidate(
        "unsafe",
        fidelity=100.0,
        ssim=1.0,
        delta=0.0,
        edge=1.0,
        recall=0.8,
        min_iou=0.0,
        safe=False,
    )
    safe = _candidate(
        "safe",
        fidelity=99.0,
        ssim=0.99,
        delta=0.1,
        edge=0.99,
    )
    assert _selection_pool([unsafe, safe]) == [safe]


def test_all_unsafe_candidates_preserve_legacy_pool_order() -> None:
    first = _candidate(
        "first",
        fidelity=90.0,
        ssim=0.9,
        delta=1.0,
        edge=0.9,
        recall=0.5,
        safe=False,
    )
    second = _candidate(
        "second",
        fidelity=80.0,
        ssim=0.8,
        delta=2.0,
        edge=0.8,
        recall=0.5,
        safe=False,
    )
    assert _selection_pool([first, second]) == [first, second]
