from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_painter import apply_candidate_painter_reconstruction
from app.alpha_mask_contour import loop_signed_area, trace_cell_contours
from app.alpha_svg_mask import _journal_source_rgb
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba
from app.transform_journal import TransformJournal


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
    # Yumuşak halka: gövde dışına ve sayaç içine 2 hücrelik degrade.
    for distance, value in ((1, 176.0), (2, 64.0)):
        grown = np.zeros_like(body)
        grown[20 - distance : 76 + distance, 24 - distance : 104 + distance] = True
        ring = grown & ~body & (alpha == 0.0)
        alpha[ring] = value
        shrunk = np.zeros_like(counter)
        shrunk[
            38 + distance : 58 - distance, 52 + distance : 76 - distance
        ] = True
        inner_ring = counter & ~shrunk & (alpha == 0.0)
        alpha[inner_ring] = value
        counter = shrunk
    rgba[:, :, 3] = alpha.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)
    return rgba


def _covering_candidate(path: Path) -> bytes:
    """Opaque trace output: comparison canvas + glyph body + counter patch.

    Gerçek tracer gibi gövde yolu yumuşak sınırın kısmi piksellerine bir hücre
    taşar; kalan halka desteğini ölçülen en küçük same-color stroke tamamlar.
    """
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96" '
        'viewBox="0 0 128 96">'
        '<path d="M0 0H128V96H0Z" fill="#FFFFFF"/>'
        '<path d="M23 19H105V77H23Z" fill="#B4141E"/>'
        '<path d="M53 39H75V57H53Z" fill="#FFFFFF"/>'
        '</svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


class PainterContourTests(unittest.TestCase):
    def test_contours_cover_random_grids_exactly(self) -> None:
        rng = np.random.default_rng(11)

        def rasterize_evenodd(loops, height, width):
            segments = []
            for corners in loops:
                closed = corners + [corners[0]]
                for (x0, y0), (x1, y1) in zip(closed, closed[1:]):
                    if x0 == x1:
                        segments.append((x0, min(y0, y1), max(y0, y1)))
            grid = np.zeros((height, width), dtype=bool)
            for cell_y in range(height):
                center_y = cell_y + 0.5
                spans = sorted(
                    x for x, y0, y1 in segments if y0 < center_y < y1
                )
                for cell_x in range(width):
                    crossings = sum(1 for x in spans if x > cell_x + 0.5)
                    grid[cell_y, cell_x] = (crossings % 2) == 1
            return grid

        for _trial in range(12):
            mask = rng.random((13, 17)) < 0.45
            loops = trace_cell_contours(mask)
            if not mask.any():
                self.assertEqual(loops, [])
                continue
            self.assertTrue(
                np.array_equal(rasterize_evenodd(loops, 13, 17), mask)
            )
            corners = sum(len(loop) for loop in loops)
            self.assertGreater(corners, 0)
            self.assertTrue(all(loop_signed_area(loop) != 0 for loop in loops))

    def test_contours_are_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        mask = rng.random((21, 19)) < 0.5
        first = trace_cell_contours(mask)
        second = trace_cell_contours(mask.copy())
        self.assertEqual(first, second)


class CandidatePainterReconstructionTests(unittest.TestCase):
    def test_painter_passes_alpha_gates_on_both_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            source = _soft_ring_source(source_path)
            original = _covering_candidate(svg_path)
            self.assertIn(b'M23 19H105V77H23Z', original)

            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )

            self.assertTrue(report["applied"])
            self.assertEqual(
                report["mask_encoding"], "candidate_painter_luminance_mask"
            )
            self.assertEqual(
                report["schema"], "rfv3d2-candidate-painter-reconstruction-v1"
            )
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)
            self.assertLessEqual(report["painter_native_alpha_mae"], 0.005)
            self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.995)
            self.assertLessEqual(report["source_truth_alpha_mae"], 0.005)
            self.assertGreaterEqual(report["final_evaluator_alpha_iou"], 0.995)
            self.assertEqual(report["final_evaluator_alpha_plane_status"], "passed")
            self.assertLessEqual(
                report["after_byte_size"], report["preflight_byte_limit"]
            )
            self.assertEqual(
                report["preflight_parent_path_count"],
                report["preserved_path_count"],
            )
            self.assertEqual(
                report["preflight_parent_node_count"],
                report["preserved_node_count"],
            )

            text = svg_path.read_bytes()
            self.assertIn(b'M23 19H105V77H23Z', text)
            self.assertIn(b"painter-luminance-v1", text)
            # Luminance mask iki render-eşdeğer vektör kodlamasından biriyle yazılır
            # (döngü başına <polygon> ya da seviye başına gruplanmış <rect>); ikisi de
            # gri rgb() fill kullanır ve raster içermez.
            self.assertIn(report["reconstruction_mask_encoding"], ("polygon", "rect"))
            self.assertIn(b'fill="rgb(', text)
            self.assertNotIn(b"<image", text.lower())
            self.assertNotIn(b"data:image", text.lower())

            rendered = render_svg_to_rgba(svg_path, 128, 96)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            native = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
            self.assertGreaterEqual(native["alpha_iou"], 0.995, native)
            self.assertLessEqual(native["alpha_mae"], 0.005, native)

            # Küçültülmüş görünüm: iki taraf da aynı INTER_AREA filtresinden geçer.
            import cv2

            half_alpha = cv2.resize(
                rendered[:, :, 3], (64, 48), interpolation=cv2.INTER_AREA
            )
            source_half = resize_rgba(source, 64, 48)
            downscaled = alpha_plane_metrics(source_half[:, :, 3], half_alpha)
            self.assertGreaterEqual(downscaled["alpha_iou"], 0.995, downscaled)

    def test_painter_preserves_native_topology(self) -> None:
        from app.final_artifact_evaluator import (
            _classify,
            _derive_palette,
            _topology_signature,
        )
        from app.fidelity import render_svg_to_rgb

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            _covering_candidate(svg_path)
            apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )

            with Image.open(source_path) as image:
                source_rgb = _journal_source_rgb(image)
            rendered = render_svg_to_rgb(svg_path, 128, 96)
            self.assertIsNotNone(rendered)
            palette = _derive_palette(source_rgb)
            min_area = 6
            source_topology = _topology_signature(
                _classify(source_rgb, palette), len(palette), min_area
            )
            render_topology = _topology_signature(
                _classify(rendered, palette), len(palette), min_area
            )
            # Sentetik min_area=6 sınıflandırması mid-ton bantlarda ±1 delik
            # oynatabilir; gerçek parent-delta kapısı journal kabul testinde
            # birebir uygulanıyor. Burada natif ölçekte parçalanma OLMADIĞI
            # (512-grid bloklaşmasının yüzlerce sahte bileşen/delik imzası)
            # kanıtlanır.
            self.assertLessEqual(
                abs(render_topology["holes"] - source_topology["holes"]),
                1,
                (source_topology, render_topology),
            )
            self.assertLessEqual(
                abs(render_topology["components"] - source_topology["components"]),
                2,
                (source_topology, render_topology),
            )

    def test_real_transform_journal_accepts_painter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            parent_path = root / "parent.svg"
            final_path = root / "final.svg"
            _soft_ring_source(source_path)
            original = _covering_candidate(parent_path)
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

    def test_painter_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            first = root / "first.svg"
            second = root / "second.svg"
            _soft_ring_source(source_path)
            _covering_candidate(first)
            _covering_candidate(second)
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

    def test_painter_fails_closed_without_paint_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            # Kaynak alfa desteği boyalı gövdeden uzakta: hiçbir stroke merdiveni
            # oradaki alfayı gerçek boya ile dolduramaz; kapı fail-closed kalmalı
            # ve seçili SVG baytları birebir korunmalı.
            rgba = np.zeros((96, 128, 4), dtype=np.uint8)
            rgba[:, :, :3] = (180, 20, 30)
            rgba[8:28, 4:20, 3] = 255
            Image.fromarray(rgba, mode="RGBA").save(source_path)
            original = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="96" '
                'viewBox="0 0 128 96">'
                '<path d="M0 0H128V96H0Z" fill="#FFFFFF"/>'
                '<path d="M90 60H120V90H90Z" fill="#B4141E"/>'
                '</svg>'
            ).encode("utf-8")
            svg_path.write_bytes(original)

            with self.assertRaisesRegex(
                RuntimeError, "source_alpha_candidate_painter"
            ):
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            self.assertEqual(svg_path.read_bytes(), original)

    def test_painter_rejects_opaque_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(source_path)
            original = _covering_candidate(svg_path)
            with self.assertRaisesRegex(
                RuntimeError, "source_alpha_candidate_painter_opaque_source"
            ):
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            self.assertEqual(svg_path.read_bytes(), original)


class PainterMaskEncodingTests(unittest.TestCase):
    """Polygon ve rect luminance-mask kodlamaları render-EŞDEĞER; karmaşık
    maskede rect daha kompakt olmalı (byte bütçesi için) ve ikisi de path_count
    invaryantını bozmaz (<polygon>/<rect> path olarak sayılmaz)."""

    def _render_mask(self, children, width: int, height: int):
        import xml.etree.ElementTree as ET

        ns = "http://www.w3.org/2000/svg"
        qname = lambda name: f"{{{ns}}}{name}"
        with tempfile.TemporaryDirectory() as directory:
            svg_path = Path(directory) / "mask.svg"
            root = ET.Element(
                qname("svg"),
                {
                    "xmlns": ns,
                    "width": str(width),
                    "height": str(height),
                    "viewBox": f"0 0 {width} {height}",
                },
            )
            defs = ET.SubElement(root, qname("defs"))
            mask = ET.SubElement(
                defs,
                qname("mask"),
                {
                    "id": "m",
                    "maskUnits": "userSpaceOnUse",
                    "x": "0",
                    "y": "0",
                    "width": str(width),
                    "height": str(height),
                },
            )
            content = ET.SubElement(mask, qname("g"))
            ET.SubElement(
                content,
                qname("rect"),
                {
                    "x": "0",
                    "y": "0",
                    "width": str(width),
                    "height": str(height),
                    "fill": "rgb(0,0,0)",
                },
            )
            for child in children:
                content.append(child)
            ET.SubElement(
                root,
                qname("rect"),
                {
                    "x": "0",
                    "y": "0",
                    "width": str(width),
                    "height": str(height),
                    "fill": "#ffffff",
                    "mask": "url(#m)",
                },
            )
            svg_path.write_bytes(ET.tostring(root))
            return render_svg_to_rgba(svg_path, width, height)

    def test_rect_and_polygon_encodings_render_identically(self) -> None:
        from app.alpha_candidate_painter import (
            _painter_loops,
            _painter_polygon_children,
            _painter_rect_children,
            _serialized_children_size,
        )
        from app.alpha_svg_mask import _quantize_alpha

        ns = "http://www.w3.org/2000/svg"
        qname = lambda name: f"{{{ns}}}{name}"
        # Çok seviyeli, çentikli yarı-saydam alfa → çok sayıda kontur döngüsü.
        height, width = 40, 48
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        alpha = (
            128.0 + 96.0 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
        ).clip(0, 255).astype(np.uint8)
        alpha[((xx + yy).astype(int) % 7) == 0] = 0  # kapalı şeffaf lakeler
        quantized, opacity = _quantize_alpha(alpha)

        loops = _painter_loops(quantized, opacity)
        polygon_children = _painter_polygon_children(loops, qname)
        rect_children, rect_count = _painter_rect_children(
            quantized, opacity, qname
        )
        self.assertGreater(rect_count, 0)
        self.assertGreater(len(loops), 8)
        # Karmaşık maskede rect kodlaması daha kompakt (döngü başına tag yerine
        # seviye başına gruplu rect).
        self.assertLess(
            _serialized_children_size(rect_children),
            _serialized_children_size(polygon_children),
        )

        rendered_polygon = self._render_mask(polygon_children, width, height)
        rendered_rect = self._render_mask(rect_children, width, height)
        self.assertIsNotNone(rendered_polygon)
        self.assertIsNotNone(rendered_rect)
        assert rendered_polygon is not None and rendered_rect is not None
        # İki kodlama piksel-özdeş render eder (kayıpsız kompaktlaştırma).
        self.assertTrue(
            np.array_equal(rendered_polygon, rendered_rect),
            int(
                np.abs(
                    rendered_polygon.astype(np.int32)
                    - rendered_rect.astype(np.int32)
                ).max()
            ),
        )

    def test_complex_mask_selects_rect_and_preserves_identity(self) -> None:
        # Karmaşık yarı-saydam kaynak: painter maskede rect kodlamasını seçmeli,
        # byte bütçesine sığmalı, aday path/node kimliğini korumalı, raster yok.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            height, width = 72, 96
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            # Her yerde yarı-saydam, çok seviyeli (alpha hiç 0 değil → tam saydam
            # bölge yok → classify "absent" → Case B: parent paint korunur ve
            # luminance mask ile maskelenir). Karmaşık maske → çok döngü/seviye,
            # ama byte bütçesine sığacak kadar (rect kodlaması seçilir).
            alpha = (
                120.0 + 55.0 * np.sin(xx / 9.0) * np.cos(yy / 10.0)
            ).clip(25, 250)
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = (40, 120, 200)
            rgba[:, :, 3] = alpha.astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(source_path)
            original = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="72" '
                'viewBox="0 0 96 72">'
                '<path d="M0 0H96V72H0Z" fill="#2878C8"/>'
                '</svg>'
            ).encode("utf-8")
            svg_path.write_bytes(original)
            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )
            self.assertEqual(report["reconstruction_mask_encoding"], "rect")
            self.assertLessEqual(
                report["after_byte_size"], report["preflight_byte_limit"]
            )
            # Aday kimliği: mask <rect>/<g> path_count'a sayılmaz.
            self.assertEqual(
                report["preflight_parent_path_count"],
                report["preserved_path_count"],
            )
            self.assertEqual(
                report["preflight_parent_node_count"],
                report["preserved_node_count"],
            )
            self.assertTrue(report["candidate_identity_preserved"])
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)
            text = svg_path.read_bytes()
            self.assertNotIn(b"<image", text.lower())
            self.assertNotIn(b"data:image", text.lower())


if __name__ == "__main__":
    unittest.main()
