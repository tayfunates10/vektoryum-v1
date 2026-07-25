from __future__ import annotations

import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_artwork_identity import artwork_fingerprint
from app.alpha_candidate_painter import (
    _painter_contour_children,
    _painter_loops,
    _painter_polygon_children,
    _painter_rect_children,
    apply_candidate_painter_reconstruction,
    build_painter_reconstruction_tree,
)
from app.alpha_svg_mask import _quantize_alpha
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba

SVG_NS = "http://www.w3.org/2000/svg"
Q = lambda name: f"{{{SVG_NS}}}{name}"
GRAYS = (0, 1, 32, 64, 128, 192, 254, 255)


def _render_bytes(data: bytes, width: int, height: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.svg"
        path.write_bytes(data)
        rgba = render_svg_to_rgba(path, width, height)
    if rgba is None:
        raise AssertionError("production renderer unavailable")
    return rgba


def _render_tree(root: ET.Element, width: int, height: int) -> np.ndarray:
    ET.register_namespace("", SVG_NS)
    return _render_bytes(
        ET.tostring(root, encoding="utf-8", xml_declaration=True),
        width,
        height,
    )


def _root(
    *,
    canvas_fill: str = "#B4141E",
    viewbox: tuple[float, float, float, float] = (0.0, 0.0, 64.0, 64.0),
) -> tuple[ET.Element, ET.Element]:
    vx, vy, vw, vh = viewbox
    root = ET.Element(
        Q("svg"),
        {
            "width": "64",
            "height": "64",
            "viewBox": f"{vx:g} {vy:g} {vw:g} {vh:g}",
        },
    )
    canvas = ET.SubElement(
        root,
        Q("rect"),
        {
            "x": f"{vx:g}",
            "y": f"{vy:g}",
            "width": f"{vw:g}",
            "height": f"{vh:g}",
            "fill": canvas_fill,
        },
    )
    ET.SubElement(
        root,
        Q("circle"),
        {
            "cx": f"{vx + vw / 2:g}",
            "cy": f"{vy + vh / 2:g}",
            "r": f"{min(vw, vh) / 8:g}",
            "fill": canvas_fill,
        },
    )
    return root, canvas


def _soft_support(width: int, height: int) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radius = min(width, height) * 0.34
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((radius + 1.5 - distance) * 96.0, 0.0, 255.0)
    alpha[(xx < width * 0.18) & (yy < height * 0.18)] = 0.0
    return alpha.astype(np.uint8)


def _mask_probe(gray: int, mask_type: str | None = None) -> bytes:
    attr = ""
    if mask_type is not None:
        attr = f' mask-type="{mask_type}"'
    return f'''<svg xmlns="{SVG_NS}" width="32" height="32" viewBox="0 0 32 32">
      <defs><mask id="m" maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" x="0" y="0" width="32" height="32"{attr}>
        <rect x="0" y="0" width="32" height="32" fill="rgb(0,0,0)"/>
        <rect x="0" y="0" width="32" height="32" fill="rgb({gray},{gray},{gray})"/>
      </mask></defs>
      <rect x="0" y="0" width="32" height="32" fill="#fff" mask="url(#m)"/>
    </svg>'''.encode()


class PainterMaskTransferTests(unittest.TestCase):
    def test_implicit_and_explicit_luminance_transfer_are_byte_exact(self) -> None:
        for gray in GRAYS:
            for mask_type in (None, "luminance"):
                rendered = _render_bytes(_mask_probe(gray, mask_type), 32, 32)
                self.assertEqual(int(rendered[16, 16, 3]), gray)

    def test_alpha_mask_type_uses_shape_alpha_not_gray(self) -> None:
        for gray in (0, 128, 255):
            rendered = _render_bytes(_mask_probe(gray, "alpha"), 32, 32)
            self.assertEqual(int(rendered[16, 16, 3]), 255)


class PainterCanvasUnderpaintTests(unittest.TestCase):
    def _build(self, encoding: str = "polygon", viewbox=(0.0, 0.0, 64.0, 64.0)):
        root, canvas = _root(viewbox=viewbox)
        alpha = _soft_support(64, 64)
        quantized, opacity = _quantize_alpha(alpha)
        tree, stats = build_painter_reconstruction_tree(
            root,
            canvas,
            quantized,
            opacity,
            1.0,
            mask_encoding=encoding,
            transaction_id=f"faz3d-{encoding}",
        )
        return root, canvas, alpha, tree, stats

    def test_proven_canvas_is_retained_only_inside_source_alpha_mask(self) -> None:
        _root_node, _canvas, alpha, tree, stats = self._build()
        rendered = _render_tree(tree, 64, 64)
        metrics = alpha_plane_metrics(alpha, rendered[:, :, 3])
        self.assertGreaterEqual(metrics["alpha_iou"], 0.995, metrics)
        self.assertLessEqual(metrics["alpha_mae"], 0.005, metrics)
        self.assertFalse(stats["comparison_canvas_knocked_out"])
        self.assertTrue(stats["comparison_canvas_retained_under_mask"])
        data = ET.tostring(tree)
        self.assertIn(b"comparison-canvas-underpaint", data)
        self.assertIn(b"comparison-canvas-v1", data)
        self.assertNotIn(b"data-vektoryum-candidate-geometry-knockout", data)
        self.assertNotIn(b"<image", data.lower())
        self.assertNotIn(b"data:image", data.lower())

    def test_underpaint_is_excluded_from_artwork_fingerprint(self) -> None:
        root, canvas, _alpha, tree, _stats = self._build()
        transaction = "faz3d-polygon"
        parent = artwork_fingerprint(root, transaction, (canvas,))
        candidate = artwork_fingerprint(tree, transaction)
        self.assertEqual(parent, candidate)

    def test_fractional_nonzero_viewbox_uses_cell_edge_mapping(self) -> None:
        _root_node, _canvas, alpha, tree, _stats = self._build(
            viewbox=(5.5, -2.25, 91.75, 47.5)
        )
        # Explicit anisotropic mapping; default meet would letterbox.
        tree.set("preserveAspectRatio", "none")
        rendered = _render_tree(tree, 64, 64)
        metrics = alpha_plane_metrics(alpha, rendered[:, :, 3])
        self.assertGreaterEqual(metrics["alpha_iou"], 0.995, metrics)
        self.assertLessEqual(metrics["alpha_mae"], 0.005, metrics)

    def test_polygon_rect_and_contour_underpaint_are_native_equivalent(self) -> None:
        renders = {}
        for encoding in ("polygon", "rect", "contour"):
            _root_node, _canvas, alpha, tree, stats = self._build(encoding)
            rendered = _render_tree(tree, 64, 64)
            metrics = alpha_plane_metrics(alpha, rendered[:, :, 3])
            self.assertGreaterEqual(metrics["alpha_iou"], 0.995, (encoding, metrics))
            self.assertTrue(stats["comparison_canvas_retained_under_mask"])
            renders[encoding] = rendered[:, :, 3]
        self.assertTrue(np.array_equal(renders["polygon"], renders["contour"]))
        self.assertTrue(np.array_equal(renders["polygon"], renders["rect"]))

    def test_output_tree_is_deterministic(self) -> None:
        first = self._build()[3]
        second = self._build()[3]
        first_bytes = ET.tostring(first)
        second_bytes = ET.tostring(second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            hashlib.sha256(first_bytes).hexdigest(),
            hashlib.sha256(second_bytes).hexdigest(),
        )

    def test_absent_canvas_path_remains_masked_without_underpaint(self) -> None:
        root, _canvas = _root()
        root.remove(list(root)[0])
        ET.SubElement(
            root,
            Q("rect"),
            {"x": "0", "y": "0", "width": "64", "height": "64", "fill": "#B4141E"},
        )
        alpha = _soft_support(64, 64)
        quantized, opacity = _quantize_alpha(alpha)
        tree, stats = build_painter_reconstruction_tree(
            root, None, quantized, opacity, 1.0,
            mask_encoding="polygon", transaction_id="faz3d-absent"
        )
        metrics = alpha_plane_metrics(alpha, _render_tree(tree, 64, 64)[:, :, 3])
        self.assertGreaterEqual(metrics["alpha_iou"], 0.995, metrics)
        self.assertFalse(stats["comparison_canvas_retained_under_mask"])

    def test_end_to_end_sparse_artwork_uses_masked_canvas_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_path = base / "source.png"
            svg_path = base / "candidate.svg"
            alpha = _soft_support(96, 72)
            rgba = np.zeros((72, 96, 4), dtype=np.uint8)
            rgba[:, :, :3] = (180, 20, 30)
            rgba[:, :, 3] = alpha
            Image.fromarray(rgba, mode="RGBA").save(source_path)
            original = (
                '<?xml version="1.0" encoding="utf-8"?>'
                f'<svg xmlns="{SVG_NS}" width="96" height="72" viewBox="0 0 96 72">'
                '<path d="M0 0H96V72H0Z" fill="#B4141E"/>'
                '<circle cx="48" cy="36" r="8" fill="#B4141E"/>'
                '</svg>'
            ).encode()
            svg_path.write_bytes(original)
            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )
            self.assertTrue(report["applied"])
            self.assertTrue(report["comparison_canvas_retained_under_mask"])
            self.assertFalse(report["comparison_canvas_knocked_out"])
            self.assertGreaterEqual(
                report["candidate_support_stroke_width_pixels"], 1.5
            )
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)
            self.assertLessEqual(report["painter_native_alpha_mae"], 0.005)
            self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.995)
            self.assertGreaterEqual(report["final_evaluator_alpha_iou"], 0.995)
            self.assertTrue(report["artwork_identity_preserved"])
            text = svg_path.read_bytes().lower()
            self.assertNotIn(b"<image", text)
            self.assertNotIn(b"data:image", text)


if __name__ == "__main__":
    unittest.main()
