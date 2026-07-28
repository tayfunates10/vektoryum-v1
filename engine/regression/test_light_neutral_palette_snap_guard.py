from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from app.geometry_cleanup import consolidate_svg_palette


def _fills(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        str(element.get("fill")).lower()
        for element in root.iter()
        if element.tag.split("}")[-1] == "path" and element.get("fill")
    ]


class BroadLightNeutralCanonicalSnapTests(unittest.TestCase):
    def test_substantial_light_neutral_fill_is_not_snapped_to_white(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#ffffff" d="M0 0H100V100H0Z"/>
        <path fill="#ebebeb" d="M20 20H80V80H20Z"/>
        <path fill="#141414" d="M10 10H18V90H10Z"/>
        <path fill="#fa0a0a" d="M45 10H55V90H45Z"/>
        </svg>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "badge.svg"
            path.write_text(svg, encoding="utf-8")
            report = consolidate_svg_palette(
                path,
                max_colors=6,
                canonical=[(0, 0, 0), (255, 255, 255), (255, 0, 0)],
            )

            fills = _fills(path)
            self.assertIn("#ffffff", fills)
            self.assertIn("#ebebeb", fills)
            self.assertIn("#000000", fills)
            self.assertIn("#ff0000", fills)
            guard = report.get("light_neutral_snap_guard")
            self.assertIsInstance(guard, dict)
            self.assertTrue(guard["white_snap_removed"])
            self.assertEqual(guard["evidence"][0]["rgb"], [235, 235, 235])

    def test_tiny_neutral_antialias_path_keeps_legacy_white_snap(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#ffffff" d="M0 0H100V100H0Z"/>
        <path fill="#ebebeb" d="M49 49H51V51H49Z"/>
        </svg>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aa.svg"
            path.write_text(svg, encoding="utf-8")
            report = consolidate_svg_palette(
                path,
                max_colors=6,
                canonical=[(0, 0, 0), (255, 255, 255), (255, 0, 0)],
            )

            self.assertNotIn("#ebebeb", _fills(path))
            self.assertNotIn("light_neutral_snap_guard", report)

    def test_no_exact_white_region_keeps_legacy_behavior(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#fafafa" d="M0 0H100V100H0Z"/>
        <path fill="#ebebeb" d="M20 20H80V80H20Z"/>
        </svg>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "offwhite.svg"
            path.write_text(svg, encoding="utf-8")
            report = consolidate_svg_palette(
                path,
                max_colors=6,
                canonical=[(0, 0, 0), (255, 255, 255), (255, 0, 0)],
            )

            self.assertNotIn("light_neutral_snap_guard", report)


if __name__ == "__main__":
    unittest.main()
