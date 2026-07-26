"""RFV-3D3 — knockout clip geometrisi kodlaması sözleşmesi.

``rect`` kodlaması bugünkü kanıtlanmış davranıştır ve değişmemelidir. ``rect-transform``
YALNIZ bir fallback'tir: aynı ``<rect>`` elementlerini tam sayı raster koordinatlarıyla
yazar ve raster→user dönüşümünü clipPath'in kendi transform'una taşır.

Element türü kritiktir. Clip geometrisini ``<path>`` ile yazmak CI'da ölçüldü ve
reddedildi: ``_path_node_counts`` defs içindeki ``<path>``'leri de saydığı için aday
değişmemiş artwork kimlik sözleşmesini ihlal ediyordu (190/9221 → 445/112181,
``source_alpha_candidate_knockout_candidate_geometry_changed``). Bu yüzden aşağıdaki
kimlik testi her iki kodlama için de zorunludur.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from app.alpha_candidate_knockout import (
    _CLIP_GEOMETRY_ENCODINGS,
    _build_reconstruction_tree,
    _local_name,
    _path_node_counts,
    _write_tree_to_temp,
)
from app.source_truth import render_svg_to_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
# Kesirli ölçek bilinçlidir: gerçek vakalarda user-space koordinatları uzun ondalıklı
# yazılır ve markup maliyetini asıl bu belirler.
_VIEW = 997
_RASTER = 613


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
    alpha = np.clip(255.0 - radius * 0.8, 0.0, 255.0)
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
    assert _CLIP_GEOMETRY_ENCODINGS == ("rect", "rect-transform")


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


@pytest.mark.parametrize("clip_encoding", _CLIP_GEOMETRY_ENCODINGS)
def test_clip_geometry_never_changes_artwork_identity(clip_encoding: str) -> None:
    """Her iki kodlama da parent'ın path/node sayısını AYNEN korumalıdır.

    CI'da ölçülen gerçek başarısızlık buydu: clip geometrisi <path> ile yazıldığında
    path sayısı 190→445, node sayısı 9.221→112.181 oldu ve değişmemiş kimlik kapısı
    adayı reddetti.
    """
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    parent_counts = _path_node_counts(root)
    tree, _geometry = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding=clip_encoding
    )
    assert _path_node_counts(tree) == parent_counts

    clip_children = [
        _local_name(str(child.tag)).lower()
        for element in tree.iter()
        if _local_name(str(element.tag)).lower() == "clippath"
        for child in element
    ]
    assert clip_children, "clipPath içeriği boş olmamalı"
    assert set(clip_children) == {"rect"}


def test_rect_transform_encoding_preserves_geometry(tmp_path: Path) -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    rect_tree, rect_geometry = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect"
    )
    transform_tree, transform_geometry = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect-transform"
    )

    assert transform_geometry["reconstruction_clip_encoding"] == "rect-transform"
    for key in (
        "reconstruction_clip_count",
        "reconstruction_use_count",
        "reconstruction_rectangle_count",
    ):
        assert transform_geometry[key] == rect_geometry[key]

    rect_alpha = _render_alpha(rect_tree, tmp_path, "rect")
    transform_alpha = _render_alpha(transform_tree, tmp_path, "rect-transform")
    intersection = int(np.count_nonzero((rect_alpha > 127) & (transform_alpha > 127)))
    union = int(np.count_nonzero((rect_alpha > 127) | (transform_alpha > 127)))
    assert union > 0
    assert intersection / union == pytest.approx(1.0, abs=1e-9)
    mae = float(
        np.abs(rect_alpha.astype(np.float32) - transform_alpha.astype(np.float32)).mean() / 255.0
    )
    assert mae == pytest.approx(0.0, abs=1e-9)


def test_rect_transform_encoding_reduces_markup_cost() -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    rect_tree, _ = _build_reconstruction_tree(root, canvas, quantized, levels, clip_encoding="rect")
    transform_tree, _ = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect-transform"
    )
    rect_size = _serialized_size(rect_tree)
    transform_size = _serialized_size(transform_tree)
    # Fallback'in var olma sebebi bütçe reddini çözmektir; kazanç anlamlı olmazsa
    # sessizce işe yaramaz hâle gelir. Ölçülen oran bu alanda ~0,45'tir.
    assert transform_size < rect_size * 0.60, (rect_size, transform_size)


def test_transform_is_only_present_for_transform_encoding() -> None:
    root, canvas = _parent_root()
    quantized, levels = _quantized_alpha()
    tree, _ = _build_reconstruction_tree(
        root, canvas, quantized, levels, clip_encoding="rect-transform"
    )
    clips = [
        element for element in tree.iter()
        if _local_name(str(element.tag)).lower() == "clippath"
    ]
    assert clips
    for clip in clips:
        transform = clip.get("transform")
        assert transform is not None and transform.startswith("translate(")
        for child in clip:
            for attribute in ("x", "y", "width", "height"):
                value = str(child.get(attribute))
                assert value.lstrip("-").isdigit(), (attribute, value)
