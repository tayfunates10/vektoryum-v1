from __future__ import annotations

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from app.alpha_candidate_knockout import _render_root
from app.alpha_candidate_paint_deficit import (
    _fixed_alpha_levels,
    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)

SVG_NS = "http://www.w3.org/2000/svg"


def qname(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


class PaintDeficitCandidateTests(unittest.TestCase):
    def _root(self):
        root = ET.Element(
            qname("svg"),
            {"viewBox": "0 0 4 4", "width": "4", "height": "4"},
        )
        canvas = ET.SubElement(
            root,
            qname("rect"),
            {"x": "0", "y": "0", "width": "4", "height": "4", "fill": "white"},
        )
        ET.SubElement(
            root,
            qname("rect"),
            {"x": "0", "y": "0", "width": "2", "height": "4", "fill": "black"},
        )
        ET.SubElement(
            root,
            qname("rect"),
            {"x": "2", "y": "0", "width": "1", "height": "4", "fill": "white"},
        )
        return root, canvas

    def _source(self):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :3, :3] = 0
        rgba[:, :3, 3] = 255
        return rgba

    def test_fixed_q24_is_deterministic_and_bounded(self):
        alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
        q1, o1 = _fixed_alpha_levels(alpha)
        q2, o2 = _fixed_alpha_levels(alpha)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(o1, o2)
        self.assertLessEqual(len(np.unique(q1)), 24)
        self.assertEqual(int(q1[0, 0]), 0)
        self.assertEqual(int(q1[-1, -1]), 23)

    def test_deficit_detects_opaque_white_artwork_hole(self):
        source = self._source()
        artwork = np.zeros_like(source)
        artwork[:, :2, :3] = 0
        artwork[:, :2, 3] = 255
        artwork[:, 2:, :3] = 255
        artwork[:, 2:, 3] = 255
        labels, _palette, stats = _paint_deficit_labels(source, artwork)
        self.assertEqual(stats["paint_deficit_pixel_count"], 4)
        self.assertEqual(stats["paint_deficit_opaque_artwork_count"], 4)
        self.assertTrue(np.all(labels[:, 2] > 0))

    def test_builder_is_vector_only_deterministic_and_repairs_paint(self):
        root, canvas = self._root()
        source = self._source()
        tree1, report1 = build_paint_deficit_reconstruction_tree(
            root, canvas, source, "txn-fixed"
        )
        root2 = copy.deepcopy(root)
        canvas2 = list(root2)[0]
        tree2, report2 = build_paint_deficit_reconstruction_tree(
            root2, canvas2, source, "txn-fixed"
        )
        bytes1 = ET.tostring(tree1)
        bytes2 = ET.tostring(tree2)
        self.assertEqual(bytes1, bytes2)
        self.assertEqual(report1, report2)
        self.assertNotIn(b"<image", bytes1)
        self.assertNotIn(b"data:", bytes1)
        self.assertEqual(report1["paint_deficit_pixel_count"], 4)
        rendered = _render_root(tree1, 4, 4)
        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertTrue(np.all(rendered[:, :3, 3] == 255))
        self.assertTrue(np.all(rendered[:, :3, :3] < 20))
        self.assertTrue(np.all(rendered[:, 3, 3] == 0))

    def test_production_module_has_no_fixture_specific_branch(self):
        text = Path(
            "engine/app/alpha_candidate_paint_deficit.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("class_reklam", text)
        self.assertNotIn("qualification-public", text)


if __name__ == "__main__":
    unittest.main()