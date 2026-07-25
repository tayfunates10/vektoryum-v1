"""FAZ 3D4 — kümülatif eşik alfa maskesi encoding testleri.

Ayrık seviye poligonları (``q == k``) komşu alfa seviyeleri arasında ortak sınır
paylaşır; değerlendirme çözünürlüğüne ölçeklenince bu sınırın AA'sı siyah tabanla
karışıp bir dikiş (seam_gap) bırakır. Kümülatif eşikler (``q >= k``, düşük→yüksek
gri) her bölgeyi bir alttaki bölgeyle örttüğü için sınır AA'sı komşu gri ile
karışır (gerçek alfa rampası) → dikiş kapanır. Bu testler bu davranışı ve
deterministik/byte-birebir/even-odd/alfa-tam özelliklerini sabitler.

Testler sentetiktir; fixture adına veya vaka numarasına bağlı değildir.
"""
from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from app.alpha_candidate_knockout import _write_tree_to_temp
from app.alpha_candidate_painter import (
    _cumulative_threshold_children,
    _painter_loops,
    _painter_polygon_children,
)
from app.source_truth import render_svg_to_rgba

_SVG = "http://www.w3.org/2000/svg"


def _qname(name: str) -> str:
    return f"{{{_SVG}}}{name}"


def _mask_svg(children: list[ET.Element], side: int) -> ET.Element:
    root = ET.Element(
        _qname("svg"),
        {"xmlns": _SVG, "width": str(side), "height": str(side),
         "viewBox": f"0 0 {side} {side}"},
    )
    defs = ET.SubElement(root, _qname("defs"))
    mask = ET.SubElement(
        defs, _qname("mask"),
        {"id": "m", "maskUnits": "userSpaceOnUse",
         "maskContentUnits": "userSpaceOnUse", "x": "0", "y": "0",
         "width": str(side), "height": str(side)},
    )
    ET.SubElement(mask, _qname("rect"),
                  {"x": "0", "y": "0", "width": str(side), "height": str(side),
                   "fill": "rgb(0,0,0)"})
    for child in children:
        mask.append(child)
    ET.SubElement(root, _qname("rect"),
                  {"x": "0", "y": "0", "width": str(side), "height": str(side),
                   "fill": "rgb(255,0,0)", "mask": "url(#m)"})
    return root


def _render_alpha(children: list[ET.Element], side: int, render_side: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "probe.svg"
        target.write_text("<svg/>")
        svg = _write_tree_to_temp(_mask_svg(children, side), target)
        rendered = render_svg_to_rgba(svg, render_side, render_side)
        svg.unlink(missing_ok=True)
    assert rendered is not None
    return rendered[:, :, 3].astype(np.float32)


def _path_d_list(children: list[ET.Element]) -> list[str]:
    return [child.get("d", "") for child in children]


def _diagonal_levels(side: int, levels: int) -> tuple[np.ndarray, dict[int, float]]:
    yy, xx = np.mgrid[0:side, 0:side]
    ramp = (xx + yy).astype(np.float32)
    ramp /= ramp.max()
    steps = levels - 1
    quantized = np.clip(np.round(ramp * steps), 0, steps).astype(np.int32)
    opacity = {k: k / float(steps) for k in range(1, levels)}
    return quantized, opacity


class CumulativeThresholdEncodingTests(unittest.TestCase):
    def test_deterministic_and_byte_identical(self) -> None:
        quantized, opacity = _diagonal_levels(40, 4)
        first, stats_a = _cumulative_threshold_children(quantized, opacity, _qname)
        second, stats_b = _cumulative_threshold_children(quantized, opacity, _qname)
        self.assertEqual(_path_d_list(first), _path_d_list(second))
        self.assertEqual(stats_a, stats_b)
        # Her pozitif seviye için tek even-odd path; integer koordinat, float yok.
        self.assertEqual(stats_a["cumulative_threshold_count"], len(first))
        for child in first:
            self.assertEqual(child.get("fill-rule"), "evenodd")
            self.assertNotIn(".", child.get("d", ""))

    def test_regions_are_nested_supersets(self) -> None:
        # region_k = (q>=k); daha yüksek k daha küçük alan → gri arttıkça sınır küçülür.
        quantized, opacity = _diagonal_levels(40, 4)
        areas = [int(np.count_nonzero(quantized >= k)) for k in (1, 2, 3)]
        self.assertTrue(areas[0] > areas[1] > areas[2] > 0)

    def test_native_alpha_is_exact_quantized_gray(self) -> None:
        # Ölçüm (native) ızgarasında maske luminance'ı tam kuantize gri olmalı.
        side = 40
        quantized, opacity = _diagonal_levels(side, 4)
        children, _ = _cumulative_threshold_children(quantized, opacity, _qname)
        rendered = _render_alpha(children, side, side)
        gray = np.array([0, 85, 170, 255], dtype=np.float32)[quantized]
        self.assertLessEqual(float(np.abs(rendered - gray).max()), 1.0)

    def test_eliminates_adjacent_level_seam_vs_disjoint(self) -> None:
        # Kesirli yukarı-ölçekte kümülatif kodlama seviye sınırında ideal alfa
        # rampasının ALTINA ayrık poligonlardan çok daha az düşer (seam kapanır).
        side = 48
        quantized, opacity = _diagonal_levels(side, 4)
        cumulative, _ = _cumulative_threshold_children(quantized, opacity, _qname)
        disjoint = _painter_polygon_children(_painter_loops(quantized, opacity), _qname)
        gray = np.array([0, 85, 170, 255], dtype=np.float32)[quantized]
        for render_side in (256, 512):
            ideal = cv2.resize(gray, (render_side, render_side),
                               interpolation=cv2.INTER_AREA)
            cum_dip = float(-(_render_alpha(cumulative, side, render_side) - ideal).min())
            dis_dip = float(-(_render_alpha(disjoint, side, render_side) - ideal).min())
            # Ayrık kodlama en az iki katı derin dikiş açar; kümülatif sınırlı kalır.
            self.assertLess(cum_dip, dis_dip * 0.6,
                            f"@{render_side}: cum_dip={cum_dip} dis_dip={dis_dip}")
            self.assertLess(cum_dip, 20.0, f"@{render_side}: cum_dip={cum_dip}")

    def test_preserves_evenodd_transparent_hole(self) -> None:
        # İçinde kapalı şeffaf (q=0) göl olan seviye-2 gövde: göl native'de siyah
        # kalmalı (even-odd delik korunur), gövde dolu gri olmalı.
        side = 32
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[6:26, 6:26] = 2
        quantized[13:19, 13:19] = 0  # kapalı şeffaf göl
        opacity = {2: 2 / 3.0}
        children, _ = _cumulative_threshold_children(quantized, opacity, _qname)
        rendered = _render_alpha(children, side, side)
        self.assertLessEqual(float(rendered[15, 15]), 1.0)   # göl şeffaf
        self.assertGreaterEqual(float(rendered[9, 9]), 169.0)  # gövde ~170

    def test_empty_positive_alpha_returns_none(self) -> None:
        quantized = np.zeros((16, 16), dtype=np.int32)
        opacity: dict[int, float] = {0: 0.0}
        self.assertIsNone(_cumulative_threshold_children(quantized, opacity, _qname))


if __name__ == "__main__":
    unittest.main()
