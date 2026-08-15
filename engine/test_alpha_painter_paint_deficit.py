from __future__ import annotations

import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from app.alpha_candidate_knockout import _render_root
from app.alpha_candidate_paint_deficit import (
    _anchored_source_component_mask,
    _fixed_alpha_levels,
    _build_paint_deficit_support,
    _paint_deficit_labels,
    build_paint_deficit_reconstruction_tree,
)
from app.alpha_candidate_painter import _paint_deficit_support_retry_eligible

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
        self.assertEqual(stats["source_component_count"], 1)
        self.assertEqual(stats["anchored_source_component_count"], 1)
        self.assertEqual(stats["detached_source_component_count"], 0)
        self.assertTrue(np.all(labels[:, 2] > 0))

    def test_detached_source_component_remains_fail_closed(self):
        source = np.zeros((6, 10, 4), dtype=np.uint8)
        source[1:5, 1:4, :3] = 0
        source[1:5, 1:4, 3] = 255
        source[1:5, 7:9, :3] = 0
        source[1:5, 7:9, 3] = 255

        artwork = np.zeros_like(source)
        artwork[1:5, 1:3, :3] = 0
        artwork[1:5, 1:3, 3] = 255
        artwork[1:5, 3:4, :3] = 255
        artwork[1:5, 3:4, 3] = 255

        anchored, component_stats = _anchored_source_component_mask(
            source[:, :, 3], artwork[:, :, 3]
        )
        labels, _palette, stats = _paint_deficit_labels(source, artwork)

        self.assertEqual(component_stats["source_component_count"], 2)
        self.assertEqual(component_stats["anchored_source_component_count"], 1)
        self.assertEqual(component_stats["detached_source_component_count"], 1)
        self.assertTrue(np.all(anchored[1:5, 1:4]))
        self.assertFalse(np.any(anchored[1:5, 7:9]))
        self.assertTrue(np.all(labels[1:5, 3] > 0))
        self.assertFalse(np.any(labels[1:5, 7:9]))
        self.assertEqual(stats["detached_source_component_count"], 1)

    def test_base_delta_support_is_render_exact_and_smaller(self):
        labels = np.fromfunction(
            lambda y, x: ((x + y) % 2) + 1,
            (8, 8),
            dtype=int,
        ).astype(np.int32)
        palette = np.asarray([[32, 32, 32], [224, 32, 32]], dtype=np.uint8)
        transform = "translate(0 0) scale(1 1)"
        legacy, legacy_stats = _build_paint_deficit_support(
            labels,
            palette,
            transform,
            "txn-support-exact",
            support_encoding="per_label",
        )
        compact, compact_stats = _build_paint_deficit_support(
            labels,
            palette,
            transform,
            "txn-support-exact",
            support_encoding="base_delta",
        )

        self.assertEqual(compact_stats["paint_deficit_palette_count"], 2)
        self.assertEqual(compact_stats["paint_deficit_support_encoding"], "base_delta")
        self.assertGreater(compact_stats["paint_deficit_support_saved_bytes"], 0)
        self.assertLess(
            compact_stats["paint_deficit_support_serialized_bytes"],
            legacy_stats["paint_deficit_support_serialized_bytes"],
        )
        self.assertLess(
            compact_stats["paint_deficit_support_rect_count"],
            legacy_stats["paint_deficit_support_rect_count"],
        )

        legacy_root = ET.Element(
            qname("svg"),
            {"viewBox": "0 0 8 8", "width": "8", "height": "8"},
        )
        legacy_root.append(copy.deepcopy(legacy))
        compact_root = ET.Element(
            qname("svg"),
            {"viewBox": "0 0 8 8", "width": "8", "height": "8"},
        )
        compact_root.append(copy.deepcopy(compact))
        legacy_rgba = _render_root(legacy_root, 8, 8)
        compact_rgba = _render_root(compact_root, 8, 8)
        self.assertIsNotNone(legacy_rgba)
        self.assertIsNotNone(compact_rgba)
        np.testing.assert_array_equal(legacy_rgba, compact_rgba)

    def test_explicit_default_support_encoding_is_byte_identical(self):
        root, canvas = self._root()
        source = self._source()
        default_tree, default_report = build_paint_deficit_reconstruction_tree(
            root,
            canvas,
            source,
            "txn-default-support",
            mask_encoding="cumulative",
            levels=8,
        )
        root2 = copy.deepcopy(root)
        canvas2 = list(root2)[0]
        explicit_tree, explicit_report = build_paint_deficit_reconstruction_tree(
            root2,
            canvas2,
            source,
            "txn-default-support",
            mask_encoding="cumulative",
            levels=8,
            support_encoding="per_label",
        )
        self.assertEqual(ET.tostring(default_tree), ET.tostring(explicit_tree))
        self.assertEqual(default_report, explicit_report)

    def test_support_retry_requires_complete_byte_rejected_legacy_ladder(self):
        labels = [
            "paint-deficit-q24",
            "paint-deficit-cumulative",
            "paint-deficit-cumulative-q20",
            "paint-deficit-cumulative-q16",
            "paint-deficit-cumulative-q12",
            "paint-deficit-cumulative-q8",
        ]
        attempts = [
            {
                "encoding_label": label,
                "exact_or_quantized": "paint_deficit",
                "status": "byte_rejected",
            }
            for label in labels
        ]
        self.assertTrue(_paint_deficit_support_retry_eligible(attempts))
        self.assertFalse(_paint_deficit_support_retry_eligible(attempts[:-1]))

        quality_rejected = [dict(entry) for entry in attempts]
        quality_rejected[-1]["status"] = "evaluator_rejected"
        self.assertFalse(_paint_deficit_support_retry_eligible(quality_rejected))

        mixed_family = [dict(entry) for entry in attempts]
        mixed_family[-1]["exact_or_quantized"] = "quantized"
        self.assertFalse(_paint_deficit_support_retry_eligible(mixed_family))

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
        self.assertEqual(report1["anchored_source_component_count"], 1)
        self.assertEqual(report1["detached_source_component_count"], 0)
        rendered = _render_root(tree1, 4, 4)
        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertTrue(np.all(rendered[:, :3, 3] == 255))
        self.assertTrue(np.all(rendered[:, :3, :3] < 20))
        self.assertTrue(np.all(rendered[:, 3, 3] == 0))

    def test_cumulative_builder_is_vector_only_and_applies_alpha_once(self):
        root, canvas = self._root()
        source = self._source()
        tree1, report1 = build_paint_deficit_reconstruction_tree(
            root, canvas, source, "txn-cumulative", mask_encoding="cumulative"
        )
        root2 = copy.deepcopy(root)
        canvas2 = list(root2)[0]
        tree2, report2 = build_paint_deficit_reconstruction_tree(
            root2, canvas2, source, "txn-cumulative", mask_encoding="cumulative"
        )
        bytes1 = ET.tostring(tree1)
        self.assertEqual(bytes1, ET.tostring(tree2))
        self.assertEqual(report1, report2)
        self.assertNotIn(b"<image", bytes1)
        self.assertNotIn(b"data:", bytes1)
        self.assertEqual(
            report1["reconstruction_mask_encoding"],
            "paint-deficit-cumulative-threshold",
        )
        self.assertGreaterEqual(
            report1["reconstruction_cumulative_threshold_count"], 1
        )
        rendered = _render_root(tree1, 4, 4)
        self.assertIsNotNone(rendered)
        assert rendered is not None
        self.assertTrue(np.all(rendered[:, :3, 3] == 255))
        self.assertTrue(np.all(rendered[:, :3, :3] < 20))
        self.assertTrue(np.all(rendered[:, 3, 3] == 0))

    def test_unknown_mask_encoding_fails_closed(self):
        root, canvas = self._root()
        source = self._source()
        with self.assertRaises(RuntimeError):
            build_paint_deficit_reconstruction_tree(
                root, canvas, source, "txn-bad", mask_encoding="spline"
            )

    def test_production_module_has_no_fixture_specific_branch(self):
        module_path = (
            Path(__file__).resolve().parent
            / "app"
            / "alpha_candidate_paint_deficit.py"
        )
        text = module_path.read_text(encoding="utf-8")
        self.assertNotIn("class_reklam", text)
        self.assertNotIn("qualification-public", text)


if __name__ == "__main__":
    unittest.main()
