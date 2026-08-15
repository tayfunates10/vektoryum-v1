"""Paint-deficit destek katmanı geometri kodlaması sözleşmesi.

Bu aile bayt bütçesine takılıp reddediliyordu (public-15: 617 139 B destek
katmanı, 13 348 rect, ~46 B/rect). Kodlamayı ``<rect>`` yerine tek bir bağıl
``<path>`` verisine taşımak baytı düşürür; bu testler kazancın **ölçülerek**
seçildiğini ve pikselin değişmediğini sabitler.
"""
from __future__ import annotations

import io
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from app.alpha_candidate_paint_deficit import (
    _compact_rectangle_path_data,
    _emit_paint_deficit_support_geometry,
    _qname,
)

_SVG_NS = "http://www.w3.org/2000/svg"


def _render(markup: str) -> np.ndarray:
    import resvg_py

    png = resvg_py.svg_to_bytes(svg_string=markup)
    return np.array(Image.open(io.BytesIO(bytes(png))).convert("RGBA"), dtype=np.int16)


def _document(body: str, size: int = 96) -> str:
    return (
        f'<svg xmlns="{_SVG_NS}" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">{body}</svg>'
    )


def _staircase_layers() -> list[tuple[np.ndarray, list[tuple[int, int, int, int]]]]:
    """Bitişik, ayrık ve tekrar eden dikdörtgenler içeren gerçekçi bir küme."""
    first = [(x, x, 3, 5) for x in range(0, 60, 4)]
    first += [(10, 40, 25, 6), (36, 40, 25, 6)]  # kenar paylaşan iki komşu
    second = [(x, 70, 2, 9) for x in range(5, 80, 3)]
    return [
        (np.asarray([20, 90, 200], dtype=np.uint8), first),
        (np.asarray([200, 40, 60], dtype=np.uint8), second),
    ]


class CompactRectanglePathDataTest(unittest.TestCase):
    def test_relative_encoding_tracks_closepath_cursor(self) -> None:
        # ``z`` imleci alt-yolun BAŞLANGICINA döndürür; ikinci ``m`` bu yüzden
        # (10, 10) noktasına göre bağıldır.
        data = _compact_rectangle_path_data([(10, 10, 4, 3), (12, 15, 2, 2)])
        self.assertEqual(data, "m10 10h4v3h-4zm2 5h2v2h-2z")

    def test_empty_input_yields_empty_data(self) -> None:
        self.assertEqual(_compact_rectangle_path_data([]), "")


class SupportGeometryEncodingTest(unittest.TestCase):
    def _emit(self, layers: list) -> tuple[ET.Element, dict]:
        support = ET.Element(_qname("g"))
        stats = _emit_paint_deficit_support_geometry(support, layers)
        return support, stats

    def test_path_form_is_selected_and_reports_measured_saving(self) -> None:
        support, stats = self._emit(_staircase_layers())
        self.assertEqual(stats["paint_deficit_support_geometry_encoding"], "path")
        self.assertLess(
            stats["paint_deficit_support_path_form_bytes"],
            stats["paint_deficit_support_rect_form_bytes"],
        )
        self.assertEqual(
            stats["paint_deficit_support_geometry_saved_bytes"],
            stats["paint_deficit_support_rect_form_bytes"]
            - stats["paint_deficit_support_path_form_bytes"],
        )
        # Seçilen kodlama gerçekten ağaca yazılmış olmalı.
        self.assertEqual(len(support.findall(_qname("path"))), 2)
        self.assertEqual(support.findall(_qname("rect")), [])

    def test_selection_is_byte_driven_and_never_worse_than_rect_form(self) -> None:
        # Sözleşme, belirli bir girdinin hangi biçimi seçtiği değil, seçimin
        # ÖLÇÜLEN bayta bağlı olması: seçilen biçim asla <rect> biçiminden
        # büyük olamaz, kazanç negatif olamaz.
        cases = [
            [(np.asarray([0, 0, 0], dtype=np.uint8), [(0, 0, 1, 1)])],
            [(np.asarray([7, 7, 7], dtype=np.uint8), [(4, 9, 12, 3), (40, 2, 1, 1)])],
            _staircase_layers(),
        ]
        for layers in cases:
            with self.subTest(rect_count=sum(len(items) for _c, items in layers)):
                support, stats = self._emit(layers)
                encoding = stats["paint_deficit_support_geometry_encoding"]
                rect_bytes = stats["paint_deficit_support_rect_form_bytes"]
                path_bytes = stats["paint_deficit_support_path_form_bytes"]
                chosen = path_bytes if encoding == "path" else rect_bytes
                self.assertEqual(chosen, min(rect_bytes, path_bytes))
                self.assertLessEqual(chosen, rect_bytes)
                self.assertGreaterEqual(
                    stats["paint_deficit_support_geometry_saved_bytes"], 0
                )
                self.assertTrue(len(support))

    def test_both_encodings_render_identical_pixels(self) -> None:
        layers = _staircase_layers()
        rect_support = ET.Element(_qname("g"))
        for color, rectangles in layers:
            group = ET.SubElement(
                rect_support,
                _qname("g"),
                {"fill": f"rgb({color[0]},{color[1]},{color[2]})"},
            )
            for x, y, width, height in rectangles:
                ET.SubElement(
                    group,
                    _qname("rect"),
                    {
                        "x": str(x),
                        "y": str(y),
                        "width": str(width),
                        "height": str(height),
                    },
                )
        path_support, _ = self._emit(layers)

        def markup(node: ET.Element) -> str:
            return _document(ET.tostring(node, encoding="unicode"))

        rendered_rect = _render(markup(rect_support))
        rendered_path = _render(markup(path_support))
        self.assertEqual(rendered_rect.shape, rendered_path.shape)
        self.assertEqual(int(np.abs(rendered_rect - rendered_path).max()), 0)


if __name__ == "__main__":
    unittest.main()
