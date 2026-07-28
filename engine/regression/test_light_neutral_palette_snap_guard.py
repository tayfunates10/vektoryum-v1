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


class BroadNeutralCanonicalSnapTests(unittest.TestCase):
    def test_substantial_dark_and_light_neutral_fills_keep_source_tones(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#ffffff" d="M0 0H100V100H0Z"/>
        <path fill="#191919" d="M10 10H90V90H10Z"/>
        <path fill="#ebebeb" d="M20 20H80V80H20Z"/>
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
            self.assertIn("#191919", fills)
            self.assertIn("#ebebeb", fills)
            self.assertIn("#ff0000", fills)
            guard = report.get("neutral_snap_guard")
            self.assertIsInstance(guard, dict)
            self.assertTrue(guard["black_snap_removed"])
            self.assertTrue(guard["white_snap_removed"])
            self.assertEqual(guard["evidence"]["dark"][0]["rgb"], [25, 25, 25])
            self.assertEqual(guard["evidence"]["light"][0]["rgb"], [235, 235, 235])

    def test_tiny_light_neutral_antialias_path_keeps_legacy_white_snap(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#ffffff" d="M0 0H100V100H0Z"/>
        <path fill="#ebebeb" d="M49 49H51V51H49Z"/>
        </svg>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aa-light.svg"
            path.write_text(svg, encoding="utf-8")
            report = consolidate_svg_palette(
                path,
                max_colors=6,
                canonical=[(0, 0, 0), (255, 255, 255), (255, 0, 0)],
            )

            self.assertNotIn("#ebebeb", _fills(path))
            self.assertNotIn("neutral_snap_guard", report)

    def test_tiny_dark_neutral_antialias_path_keeps_legacy_black_snap(self) -> None:
        svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <path fill="#ffffff" d="M0 0H100V100H0Z"/>
        <path fill="#181818" d="M49 49H51V51H49Z"/>
        </svg>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aa-dark.svg"
            path.write_text(svg, encoding="utf-8")
            report = consolidate_svg_palette(
                path,
                max_colors=6,
                canonical=[(0, 0, 0), (255, 255, 255), (255, 0, 0)],
            )

            self.assertIn("#000000", _fills(path))
            self.assertNotIn("neutral_snap_guard", report)

    def test_no_exact_white_region_disables_only_light_guard(self) -> None:
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
            guard = report.get("neutral_snap_guard")
            if guard:
                self.assertFalse(guard["white_snap_removed"])


if __name__ == "__main__":
    unittest.main()
