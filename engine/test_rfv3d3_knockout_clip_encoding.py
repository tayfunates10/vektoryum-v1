"""RFV-3D3 — knockout clip geometrisi kodlaması sözleşmesi.

``rect`` kodlaması bugünkü kanıtlanmış davranıştır ve değişmemelidir. ``path-transform``
YALNIZ bir fallback'tir: aynı birleşmiş dikdörtgenleri tam sayı raster uzayında tek bir
path olarak yazar, raster→user dönüşümünü clipPath'in kendi transform'una taşır ve
böylece bayt/node maliyetini düşürür. Bu testler geometrinin birebir aynı kaldığını,
kazancın gerçek olduğunu ve rect yolunun bozulmadığını ölçer.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from app.alpha_candidate_knockout import (
    _CLIP_GEOMETRY_ENCODINGS,
    _build_reconstruction_tree,
    _clip_raster_path_data,
    _local_name,
    _write_tree_to_temp,
)
from app.source_truth import render_svg_to_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_VIEW = 400
_RASTER = 200


def _parent_root() -> tuple[ET.Element, ET.Element]:
    root = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "viewBox": f"0 0 {_VIEW} {_VIEW}",
            "width": str(_VIEW),
            "height": str(_VIEW),
        },
    )
    canvas = ET.SubElement(
        root,
        f"{{{_SVG_NS}}}rect",
        {"x": "0", "y": "0", "width": str(_VIEW), "height": str(_VIEW), "fill": "#ffffff"},
    )
    ET.SubElement(
        root,
        f"{{{_SVG_NS}}}path",
        {"d": f"M0 0H{_VIEW}V{_VIEW}H0Z", "fill": "#123456"},
    )
    return root, canvas


def _quantized_alpha() -> tuple[np.ndarray, dict[int, float]]:
    """Çok sayıda satır-koşusu üreten deterministik yumuşak alfa alanı."""
    ys, xs = np.mgrid[0:_RASTER, 0:_RASTER]
    radius = np.sqrt((xs - _RASTER / 2) ** 2 + (ys - _RASTER / 2) ** 2)
    alpha = np.clip(255.0 - radius * 2.4, 0.0, 255.0)
    quantized = (np.round(alpha / 8.0) * 8.0).astype(np.uint8)
    levels = {int(value): int(value) / 255.0 for value in np.unique(quantized) if int(value) > 0}
    return quantized, levels


def _render_alpha(root: ET.Element, tmp_path: Path, name: str) -> np.ndarray:
    target = tmp_path / f"{name}.svg"
    target.write_bytes(b"<svg/>")
    written = _write_tree_to_temp(root, target)
    try:
        rendered = render_svg_to_rgba(written, _RASTER, _RASTER)
    finally:
        written.unlink(missing_ok=True)
    assert rendered is not None, f"{name} render edilemedi"
    return np.asarray(rendered[:, :, 3], dtype=np.uint8)


def _serialized_size(root: ET.Element) -> int:
    return len(ET.tostring(root, encoding="utf-8"))


def test_clip_encoding_order_is_rect_first() -> None:
    assert _CLIP_GEOMETRY_ENCODINGS == ("rect", "path-transform")


def test_rect_encoding_is_default_and_unchanged() -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    default_tree, default_geometry = _build_reconstruction_tree(root, canvas, quantized, levels)
    explicit_tree, _ = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect"
    )
    assert ET.tostring(default_tree) == ET.tostring(explicit_tree)
    assert default_geometry["reconstruction_clip_encoding"] == "rect"

    clips = [
        element for element in default_tree.iter()
        if _local_name(str(element.tag)).lower() == "clippath"
    ]
    assert clips, "rect kodlaması clipPath üretmeli"
    for clip in clips:
        assert clip.get("transform") is None
        assert all(_local_name(str(child.tag)).lower() == "rect" for child in clip)


def test_path_transform_encoding_preserves_geometry(tmp_path: Path) -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    rect_tree, rect_geometry = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect"
    )
    path_tree, path_geometry = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="path-transform"
    )

    assert path_geometry["reconstruction_clip_encoding"] == "path-transform"
    # Aynı bölgeler kodlanır: clip/use sayısı ve kapsanan dikdörtgen sayısı değişmez.
    assert path_geometry["reconstruction_clip_count"] == rect_geometry["reconstruction_clip_count"]
    assert path_geometry["reconstruction_use_count"] == rect_geometry["reconstruction_use_count"]
    assert (
        path_geometry["reconstruction_rectangle_count"]
        == rect_geometry["reconstruction_rectangle_count"]
    )

    rect_alpha = _render_alpha(rect_tree, tmp_path, "rect")
    path_alpha = _render_alpha(path_tree, tmp_path, "path")
    intersection = int(np.count_nonzero((rect_alpha > 127) & (path_alpha > 127)))
    union = int(np.count_nonzero((rect_alpha > 127) | (path_alpha > 127)))
    assert union > 0
    assert intersection / union == pytest.approx(1.0, abs=1e-9)
    mae = float(np.abs(rect_alpha.astype(np.float32) - path_alpha.astype(np.float32)).mean() / 255.0)
    assert mae == pytest.approx(0.0, abs=1e-9)


def test_path_transform_encoding_reduces_markup_cost() -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    rect_tree, _ = _build_reconstruction_tree(root, canvas, quantized, levels, clip_encoding="rect")
    path_tree, _ = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="path-transform"
    )
    rect_size = _serialized_size(rect_tree)
    path_size = _serialized_size(path_tree)
    # Ölçülen kazanç bu alanda ~5x'tir; sözleşme olarak en az %40 talep edilir, aksi
    # halde fallback bütçe reddini çözemez ve sessizce işe yaramaz hâle gelir.
    assert path_size < rect_size * 0.60, (rect_size, path_size)

    rect_elements = sum(
        1 for element in rect_tree.iter()
        if _local_name(str(element.tag)).lower() == "rect"
    )
    path_elements = sum(
        1 for element in path_tree.iter()
        if _local_name(str(element.tag)).lower() == "rect"
    )
    assert path_elements < rect_elements


def test_clip_path_data_skips_degenerate_rectangles() -> None:
    assert _clip_raster_path_data([(1, 2, 3, 4)]) == "M1 2h3v4h-3z"
    assert _clip_raster_path_data([(1, 2, 0, 4), (5, 6, 3, 0)]) == ""
    assert _clip_raster_path_data([]) == ""
