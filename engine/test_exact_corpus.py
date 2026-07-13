"""Exact korpus açıklaması, oracle SVG'si ve gerçek input byte uyumu."""
from __future__ import annotations

import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR / "regression"))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def test_vector_oracles_and_required_fixture_fields() -> None:
    from exact_corpus import t1_topology, t2_gradient_alpha, t3_micro_detail

    for fixture in (t1_topology(), t2_gradient_alpha(), t3_micro_detail()):
        assert fixture.oracle_svg
        assert fixture.input_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert fixture.input_mime == "image/png"
        assert fixture.reference_rgba.shape[:2] == fixture.rgb.shape[:2]
        assert fixture.protected_rois
        assert fixture.complexity_budget["max_paths"] > 0
        root = ET.fromstring(fixture.oracle_svg)
        assert _local(root.tag) == "svg"


def test_t2_description_matches_oracle_and_alpha_input() -> None:
    from exact_corpus import t2_gradient_alpha

    fixture = t2_gradient_alpha()
    root = ET.fromstring(fixture.oracle_svg)
    names = [_local(element.tag) for element in root.iter()]
    assert names.count("linearGradient") == 2
    assert names.count("radialGradient") == 1
    assert names.count("mask") == 1
    assert any(element.get("opacity") not in (None, "1") for element in root.iter())
    semantics = fixture.oracle["source_semantics"]
    assert semantics["linear_gradient_definitions"] == 2
    assert semantics["radial_gradient_definitions"] == 1
    assert fixture.alpha is not None
    assert int(fixture.alpha.min()) < int(fixture.alpha.max())
    with Image.open(io.BytesIO(fixture.input_bytes)) as image:
        decoded_alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    assert np.array_equal(decoded_alpha, fixture.alpha)


def test_t5_is_actual_q32_jpeg_not_clean_rgb() -> None:
    from exact_corpus import all_fixtures, t5_lowres_jpeg

    fixture = t5_lowres_jpeg()
    assert fixture.input_mime == "image/jpeg"
    assert fixture.input_bytes[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(fixture.input_bytes)) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert np.array_equal(decoded, fixture.rgb)
    assert np.mean(np.abs(decoded.astype(float) - fixture.reference_rgba[:, :, :3])) > 0.1
    corpus_t5 = all_fixtures()[-1]
    assert corpus_t5.input_bytes[:2] == b"\xff\xd8"
    assert np.array_equal(corpus_t5.rgb, decoded)


def test_dot_names_use_diameter_consistently() -> None:
    from exact_corpus import t1_topology, t3_micro_detail

    assert {key for key in t1_topology().protected_rois if key.startswith("dot_")} == {
        "dot_diameter_2px", "dot_diameter_4px", "dot_diameter_6px",
    }
    assert {key for key in t3_micro_detail().protected_rois if key.startswith("dot_")} == {
        "dot_diameter_3px", "dot_diameter_5px", "dot_diameter_7px",
    }


def test_gradient_segmentation_uses_nearest_seed_without_one_pixel_expansion() -> None:
    from app.gradient_vectorize import _edge_based_segments, _fit_linear_gradient

    n = 128
    rgb = np.full((n, n, 3), 255, np.uint8)
    yy, xx = np.indices((80, 80))
    rgb[24:104, 24:104, 0] = 180 - (xx + yy) // 4
    rgb[24:104, 24:104, 1] = 40 + (xx + yy) // 6
    rgb[24:104, 24:104, 2] = 90 + (xx + yy) // 5

    labels = _edge_based_segments(rgb)
    foreground = labels == labels[64, 64]
    ys, xs = np.where(foreground)
    assert (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) == (24, 103, 24, 103)
    assert not np.any(np.all(rgb[foreground] == 255, axis=1))

    gradient = _fit_linear_gradient(ys, xs, rgb[ys, xs].astype(np.float32))
    assert gradient is not None
    offsets = [offset for offset, _color in gradient[-1]]
    assert offsets[0] == 0.0 and offsets[-1] == 1.0
