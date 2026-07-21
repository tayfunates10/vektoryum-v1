"""Canvas-independent (color-agnostic) comparison-background classification and
its three-way painter integration: proven knockout / absent no-knockout /
ambiguous fail-closed. No white-canvas assumption, no ``canvas_not_proven``
crash, no filename or case-id special casing.
"""
from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_background import classify_comparison_background
from app.alpha_candidate_painter import apply_candidate_painter_reconstruction
from app.alpha_svg_mask import _journal_source_rgb
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba
from app.transform_journal import TransformJournal

_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _NS)


def _q(name: str) -> str:
    return f"{{{_NS}}}{name}"


def _root(*children: ET.Element) -> ET.Element:
    root = ET.Element(
        _q("svg"),
        {"xmlns": _NS, "width": "64", "height": "64", "viewBox": "0 0 64 64"},
    )
    for child in children:
        root.append(child)
    return root


def _rect(x: int, y: int, w: int, h: int, fill: str) -> ET.Element:
    return ET.Element(
        _q("rect"),
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "fill": fill},
    )


def _path(d: str, fill: str, **extra: str) -> ET.Element:
    attrib = {"d": d, "fill": fill}
    attrib.update(extra)
    return ET.Element(_q("path"), attrib)


def _center_opaque_source() -> np.ndarray:
    """Transparent border, opaque coloured centre 20..44."""
    source = np.zeros((64, 64, 4), dtype=np.uint8)
    source[20:44, 20:44, :3] = (200, 30, 40)
    source[20:44, 20:44, 3] = 255
    return source


_CENTER_SQ = "M20 20H44V44H20Z"


class BackgroundClassifierTests(unittest.TestCase):
    def test_white_background_is_proven(self) -> None:
        root = _root(_rect(0, 0, 64, 64, "#ffffff"), _path(_CENTER_SQ, "#000000"))
        status, element = classify_comparison_background(
            root, _center_opaque_source(), 64, 64
        )
        self.assertEqual(status, "proven")
        self.assertIsNotNone(element)

    def test_background_classification_is_color_invariant(self) -> None:
        # Aynı geometri, farklı background rengi → aynı sınıflandırma sonucu.
        source = _center_opaque_source()
        for fill in ("#ffffff", "#000000", "#3b7fd0", "#12ff88"):
            with self.subTest(fill=fill):
                root = _root(_rect(0, 0, 64, 64, fill), _path(_CENTER_SQ, "#e02040"))
                status, element = classify_comparison_background(root, source, 64, 64)
                self.assertEqual(status, "proven", fill)
                self.assertEqual(element.get("fill"), fill)

    def test_gradient_background_is_proven(self) -> None:
        defs = ET.Element(_q("defs"))
        grad = ET.SubElement(
            defs, _q("linearGradient"), {"id": "g", "x1": "0", "x2": "1"}
        )
        ET.SubElement(grad, _q("stop"), {"offset": "0", "stop-color": "#ff0000"})
        ET.SubElement(grad, _q("stop"), {"offset": "1", "stop-color": "#0000ff"})
        root = _root(defs, _rect(0, 0, 64, 64, "url(#g)"), _path(_CENTER_SQ, "#101010"))
        status, _element = classify_comparison_background(
            root, _center_opaque_source(), 64, 64
        )
        self.assertEqual(status, "proven")

    def test_no_background_is_absent(self) -> None:
        root = _root(_path(_CENTER_SQ, "#c8202a"))
        status, element = classify_comparison_background(
            root, _center_opaque_source(), 64, 64
        )
        self.assertEqual(status, "absent")
        self.assertIsNone(element)

    def test_border_touching_artwork_is_not_background(self) -> None:
        # Opak çerçeve (kenara değer) + şeffaf merkez: çerçeve artwork'tür,
        # şeffaf bölgeyi doldurmaz → background sayılmamalı.
        source = np.zeros((64, 64, 4), dtype=np.uint8)
        source[:, :, :3] = (20, 20, 20)
        source[:, :, 3] = 255
        source[16:48, 16:48, 3] = 0
        frame = _path(
            "M0 0H64V64H0Z M16 16V48H48V16Z", "#141414", **{"fill-rule": "evenodd"}
        )
        status, element = classify_comparison_background(_root(frame), source, 64, 64)
        self.assertEqual(status, "absent")
        self.assertIsNone(element)

    def test_two_full_backgrounds_are_ambiguous(self) -> None:
        root = _root(
            _rect(0, 0, 64, 64, "#ffffff"),
            _rect(0, 0, 64, 64, "#eeeeee"),
            _path(_CENTER_SQ, "#000000"),
        )
        status, element = classify_comparison_background(
            root, _center_opaque_source(), 64, 64
        )
        self.assertEqual(status, "ambiguous")
        self.assertIsNone(element)

    def test_opaque_source_has_no_background(self) -> None:
        opaque = np.zeros((64, 64, 4), dtype=np.uint8)
        opaque[:, :, 3] = 255
        root = _root(_rect(0, 0, 64, 64, "#ffffff"))
        status, element = classify_comparison_background(root, opaque, 64, 64)
        self.assertEqual(status, "absent")
        self.assertIsNone(element)


def _soft_ring_source(path: Path) -> np.ndarray:
    """Class-like glyph: opaque body, enclosed transparent counter, soft ring."""
    height, width = 96, 128
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = (180, 20, 30)
    body = np.zeros((height, width), dtype=bool)
    body[20:76, 24:104] = True
    counter = np.zeros((height, width), dtype=bool)
    counter[38:58, 52:76] = True
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[body] = 255.0
    alpha[counter] = 0.0
    for distance, value in ((1, 176.0), (2, 64.0)):
        grown = np.zeros_like(body)
        grown[20 - distance : 76 + distance, 24 - distance : 104 + distance] = True
        ring = grown & ~body & (alpha == 0.0)
        alpha[ring] = value
        shrunk = np.zeros_like(counter)
        shrunk[38 + distance : 58 - distance, 52 + distance : 76 - distance] = True
        inner_ring = counter & ~shrunk & (alpha == 0.0)
        alpha[inner_ring] = value
        counter = shrunk
    rgba[:, :, 3] = alpha.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return rgba


def _split_background_candidate(path: Path) -> bytes:
    """Opaque trace whose white comparison background is split in two halves.

    No single element fills the transparent region, so the background is
    classified ``absent`` and source alpha is reconstructed over the unchanged
    paint without any knockout (the canvas-independent path).
    """
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96" '
        'viewBox="0 0 128 96">'
        '<path d="M0 0H64V96H0Z" fill="#FFFFFF"/>'
        '<path d="M64 0H128V96H64Z" fill="#FFFFFF"/>'
        '<path d="M23 19H105V77H23Z" fill="#B4141E"/>'
        '<path d="M53 39H75V57H53Z" fill="#FFFFFF"/>'
        '</svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


def _double_background_candidate(path: Path) -> bytes:
    """Two full-canvas backgrounds → ambiguous → fail closed, bytes unchanged."""
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96" '
        'viewBox="0 0 128 96">'
        '<path d="M0 0H128V96H0Z" fill="#FFFFFF"/>'
        '<path d="M0 0H128V96H0Z" fill="#FEFEFE"/>'
        '<path d="M23 19H105V77H23Z" fill="#B4141E"/>'
        '<path d="M53 39H75V57H53Z" fill="#FFFFFF"/>'
        '</svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


class CanvasIndependentPainterTests(unittest.TestCase):
    def test_absent_background_reconstructs_without_knockout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = _soft_ring_source(source_path)
            _split_background_candidate(svg_path)
            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )
            # Canvas-independent yol: knockout uygulanmadı ama crash da yok.
            self.assertEqual(report["comparison_background_status"], "absent")
            self.assertFalse(report["comparison_canvas_knocked_out"])
            self.assertTrue(report["comparison_background_color_agnostic"])
            # d verisi ve identity korunur; raster/dataURI yok.
            text = svg_path.read_bytes()
            self.assertIn(b"M23 19H105V77H23Z", text)
            self.assertNotIn(b"<image", text.lower())
            self.assertNotIn(b"data:image", text.lower())
            # Native alfa reconstruction gerçek eşiklerden geçer.
            rendered = render_svg_to_rgba(svg_path, 128, 96)
            assert rendered is not None
            native = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(native["alpha_iou"], 0.995, native)
            self.assertLessEqual(native["alpha_mae"], 0.005, native)

    def test_absent_background_accepted_by_real_transform_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            parent_path = root / "parent.svg"
            final_path = root / "final.svg"
            _soft_ring_source(source_path)
            original = _split_background_candidate(parent_path)
            final_path.write_bytes(original)
            report = apply_candidate_painter_reconstruction(
                final_path, source_path, "logo_color"
            )
            with Image.open(source_path) as image:
                journal_source = _journal_source_rgb(image)
            journal = TransformJournal(
                parent_path,
                journal_source,
                image_class="clean_logo",
                required_metrics=set(),
            )
            accepted, stage = journal.consider_candidate(
                "source_alpha_vector_mask",
                parent_path,
                final_path,
                transform_report=report,
            )
            self.assertEqual(accepted, final_path, stage.get("reason_codes"))
            self.assertEqual(stage.get("status"), "accepted")

    def test_absent_background_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            first = root / "first.svg"
            second = root / "second.svg"
            _soft_ring_source(source_path)
            _split_background_candidate(first)
            _split_background_candidate(second)
            report_one = apply_candidate_painter_reconstruction(
                first, source_path, "logo_color"
            )
            report_two = apply_candidate_painter_reconstruction(
                second, source_path, "logo_color"
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                report_one["after_byte_size"], report_two["after_byte_size"]
            )

    def test_never_raises_canvas_not_proven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            _split_background_candidate(svg_path)
            try:
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            except RuntimeError as exc:
                self.assertNotIn("canvas_not_proven", str(exc))

    def test_ambiguous_background_fails_closed_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            original = _double_background_candidate(svg_path)
            with self.assertRaisesRegex(
                RuntimeError, "source_alpha_candidate_painter_background_ambiguous"
            ):
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            self.assertEqual(svg_path.read_bytes(), original)

    def test_absent_background_preserves_identity_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            _split_background_candidate(svg_path)
            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )
            # Candidate identity: path/node sayıları parent ile birebir.
            self.assertEqual(
                report["preflight_parent_path_count"], report["preserved_path_count"]
            )
            self.assertEqual(
                report["preflight_parent_node_count"], report["preserved_node_count"]
            )
            self.assertTrue(report["candidate_identity_preserved"])
            self.assertTrue(report["candidate_path_data_preserved"])
            # Byte bütçesi değişmemiş journal formülünden geliyor ve aşılmıyor.
            self.assertLessEqual(
                report["after_byte_size"], report["preflight_byte_limit"]
            )
            text = svg_path.read_bytes()
            for d in (b"M23 19H105V77H23Z", b"M53 39H75V57H53Z"):
                self.assertIn(d, text)

    def test_gradient_stops_preserved_through_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            # Gradient tanımlı, split beyaz background (Case B). Painter kabul
            # etse de etmese de gradient stop'ları birebir korunmalı.
            original = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96" '
                'viewBox="0 0 128 96">'
                '<defs><linearGradient id="grad" x1="0" x2="1">'
                '<stop offset="0" stop-color="#B4141E"/>'
                '<stop offset="1" stop-color="#7A0E14"/>'
                '</linearGradient></defs>'
                '<path d="M0 0H64V96H0Z" fill="#FFFFFF"/>'
                '<path d="M64 0H128V96H64Z" fill="#FFFFFF"/>'
                '<path d="M23 19H105V77H23Z" fill="url(#grad)"/>'
                '<path d="M53 39H75V57H53Z" fill="#FFFFFF"/>'
                '</svg>'
            ).encode("utf-8")
            svg_path.write_bytes(original)
            try:
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            except RuntimeError:
                # Fail-closed ise baytlar birebir korunmalı.
                self.assertEqual(svg_path.read_bytes(), original)
                return
            text = svg_path.read_bytes()
            self.assertIn(b'<stop offset="0" stop-color="#B4141E"', text)
            self.assertIn(b'<stop offset="1" stop-color="#7A0E14"', text)
            self.assertIn(b'fill="url(#grad)"', text)

    def test_absent_background_report_sha_is_stable(self) -> None:
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            first = root / "first.svg"
            second = root / "second.svg"
            _soft_ring_source(source_path)
            _split_background_candidate(first)
            _split_background_candidate(second)
            report_one = apply_candidate_painter_reconstruction(
                first, source_path, "logo_color"
            )
            report_two = apply_candidate_painter_reconstruction(
                second, source_path, "logo_color"
            )
            svg_sha_one = hashlib.sha256(first.read_bytes()).hexdigest()
            svg_sha_two = hashlib.sha256(second.read_bytes()).hexdigest()
            self.assertEqual(svg_sha_one, svg_sha_two)

            def _report_sha(report: dict) -> str:
                return hashlib.sha256(
                    json.dumps(report, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()

            self.assertEqual(_report_sha(report_one), _report_sha(report_two))


if __name__ == "__main__":
    unittest.main()
