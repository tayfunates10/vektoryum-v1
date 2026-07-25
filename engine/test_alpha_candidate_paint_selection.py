from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from app.alpha_candidate_paint_selection import (
    comparison_canvas_rgb,
    expand_non_canvas_paint,
    parse_solid_rgb,
)

_SVG = "http://www.w3.org/2000/svg"


class AlphaCandidatePaintSelectionTests(unittest.TestCase):
    def test_solid_color_parser_accepts_trace_output_forms(self) -> None:
        self.assertEqual(parse_solid_rgb("#fff"), (255, 255, 255))
        self.assertEqual(parse_solid_rgb("#FFFFFF"), (255, 255, 255))
        self.assertEqual(parse_solid_rgb("rgb(255, 0, 0)"), (255, 0, 0))
        self.assertEqual(parse_solid_rgb("rgb(100%, 0%, 0%)"), (255, 0, 0))
        self.assertIsNone(parse_solid_rgb("url(#gradient)"))

    def test_only_paint_contrasting_with_proven_canvas_is_expanded(self) -> None:
        canvas = ET.Element(f"{{{_SVG}}}path", {"fill": "#FFFFFF"})
        paint = ET.Element(f"{{{_SVG}}}g")
        black = ET.SubElement(
            paint,
            f"{{{_SVG}}}path",
            {"d": "M0 0H10V10H0Z", "fill": "#000000"},
        )
        white_detail = ET.SubElement(
            paint,
            f"{{{_SVG}}}path",
            {"d": "M2 2H8V8H2Z", "fill": "#FFFFFF"},
        )

        expanded, skipped, canvas_rgb = expand_non_canvas_paint(
            paint,
            2.0,
            canvas,
        )

        self.assertEqual(canvas_rgb, (255, 255, 255))
        self.assertEqual(expanded, 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(black.get("stroke"), "#000000")
        self.assertEqual(black.get("stroke-width"), "2")
        self.assertIsNone(white_detail.get("stroke"))
        self.assertEqual(white_detail.get("d"), "M2 2H8V8H2Z")

    def test_ambiguous_canvas_fails_safe_without_exclusion(self) -> None:
        canvas = ET.Element(f"{{{_SVG}}}g")
        ET.SubElement(canvas, f"{{{_SVG}}}path", {"fill": "#FFFFFF"})
        ET.SubElement(canvas, f"{{{_SVG}}}path", {"fill": "#000000"})
        paint = ET.Element(f"{{{_SVG}}}g")
        white = ET.SubElement(
            paint,
            f"{{{_SVG}}}path",
            {"d": "M0 0H10V10H0Z", "fill": "#FFFFFF"},
        )

        self.assertIsNone(comparison_canvas_rgb(canvas))
        expanded, skipped, canvas_rgb = expand_non_canvas_paint(
            paint,
            1.0,
            canvas,
        )
        self.assertIsNone(canvas_rgb)
        self.assertEqual(expanded, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(white.get("stroke"), "#FFFFFF")


if __name__ == "__main__":
    unittest.main()
