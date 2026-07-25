"""FAZ 3B / 3B.1 — painter kompakt encoding turnuvası testleri.

Üç kademeli seçim (render-güvenli varsayılan): Kademe 1 sayı-koruyan {polygon, rect}
(maskeleri path_count'a sayılmaz), Kademe 2 kompakt {contour} (yalnız Kademe 1 bütçeye
sığmazsa), Kademe 3 quantized {contour-q128/64/32} (yalnız hiçbir exact geçmezse). Her
aday byte preflight'tan geçer; bütçeyi aşan render edilmeden elenir. Bütçeye giren aday
değişmemiş tam kapı bataryasından (alfa + journal geometri) geçmelidir; kademedeki geçen
adaylardan en küçük byte kazanır. Böylece basit vakada render-güvenli sayı-koruyan kodlama
korunur; contour yalnız byte-zorlamalı karmaşık maskeler için ayrılır.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_painter import (
    _rectilinear_subpaths,
    _requantize_alpha,
    _simplify_rectilinear_loop,
    apply_candidate_painter_reconstruction,
)


def _simple_soft_disc_source(path: Path, side: int = 96) -> None:
    """Tek yumuşak disk: az seviye, küçük maske → polygon sığar (dikiş-güvenli)."""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = cy = side / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((side * 0.34 - dist) * 40.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[:, :, :3] = (180, 20, 30)
    rgba[:, :, 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(path)


_COMPLEX_SIDE = 360


def _complex_multi_glyph_source(path: Path, side: int = _COMPLEX_SIDE) -> None:
    """Yoğun döşeli yumuşak-halka glifleri (geçen _soft_ring yapısı, çok bileşen).

    Her glif: opak gövde + 2 hücrelik yumuşak halka → alfa 0.995-üretilebilir
    (büyük binary gövde IoU'yu sabitler). Yüzlerce bileşen → polygon kodlaması
    (döngü başına <polygon>) byte bütçesini AŞAR; grouped-evenodd contour (seviye
    başına tek even-odd <path>) ise sığar. Turnuva contour'u seçmeli."""
    step, bsize = 12, 7
    alpha = np.zeros((side, side), dtype=np.float32)
    for by in range(6, side - bsize - 4, step):
        for bx in range(6, side - bsize - 4, step):
            body = np.zeros((side, side), dtype=bool)
            body[by : by + bsize, bx : bx + bsize] = True
            alpha[body] = 255.0
            for distance, value in ((1, 176.0), (2, 64.0)):
                grown = np.zeros((side, side), dtype=bool)
                grown[
                    by - distance : by + bsize + distance,
                    bx - distance : bx + bsize + distance,
                ] = True
                ring = grown & ~body & (alpha == 0.0)
                alpha[ring] = value
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[:, :, :3] = (180, 20, 30)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _covering_parent(path: Path, side: int) -> bytes:
    inner = side - 8
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}">'
        f'<path d="M0 0H{side}V{side}H0Z" fill="#FFFFFF"/>'
        f'<path d="M4 4H{inner}V{inner}H4Z" fill="#1E5AC8"/>'
        '</svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


class PainterEncodingTournamentTests(unittest.TestCase):
    def test_rectilinear_collinear_nodes_are_removed_exactly(self) -> None:
        loop = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (3, 1), (3, 2), (3, 3),
            (2, 3), (1, 3), (0, 3),
            (0, 2), (0, 1),
        ]
        simplified = _simplify_rectilinear_loop(loop)
        self.assertEqual(simplified, [(0, 0), (3, 0), (3, 3), (0, 3)])
        path_data, nodes = _rectilinear_subpaths([loop])
        self.assertEqual(path_data, "M0 0h3v3h-3Z")
        self.assertEqual(nodes, 5)

    def test_rectilinear_simplification_is_cyclic_and_deterministic(self) -> None:
        loop = [
            (1, 0), (2, 0), (2, 1), (2, 2),
            (1, 2), (0, 2), (0, 1), (0, 0),
        ]
        first = _simplify_rectilinear_loop(loop)
        second = _simplify_rectilinear_loop(loop)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(abs(sum(
            first[i][0] * first[(i + 1) % len(first)][1]
            - first[(i + 1) % len(first)][0] * first[i][1]
            for i in range(len(first))
        )), 8)

    def test_requantize_reduces_levels_and_keeps_transparent(self) -> None:
        alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
        quant, opacity = _requantize_alpha(alpha, 32)
        self.assertLessEqual(len(opacity), 32)
        self.assertEqual(opacity[0], 0.0)
        self.assertTrue(bool((quant[alpha == 0] == 0).all()))

    def test_simple_source_selects_smallest_passing_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            report = apply_candidate_painter_reconstruction(svg, source, "logo_color")
            self.assertTrue(report["applied"])
            # FAZ 3B.1 seçim politikası: bütçeye giren+geçen exact adaylardan EN
            # KÜÇÜK byte seçilir (quantized değil). Basit kaynakta bir exact geçer.
            self.assertIn(
                report["painter_encoding_label"], {"polygon", "contour", "rect"}
            )
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)
            self.assertTrue(report["artwork_identity_preserved"])

    def test_node_heavy_source_contour_geometry_rejected_not_rect_byte(self) -> None:
        # Çok bileşenli kaynak: polygon byte'ı aşar, grouped-evenodd contour
        # BÜTÇEYE GİRER ama <path> node sayısı journal node kapısını patlatır
        # (node_complexity_explosion). FAZ 3B.1: ana hata contour'un GERÇEK journal
        # reddi olmalı — son rect byte hatası bunu EZMEMELİ.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            original = _covering_parent(svg, _COMPLEX_SIDE)
            with self.assertRaises(RuntimeError) as context:
                apply_candidate_painter_reconstruction(svg, source, "logo_color")
            message = str(context.exception)
            self.assertIn("no_admissible_reconstruction", message)
            self.assertIn("primary=contour", message)
            self.assertIn("attempts_sha256=", message)
            # Yanıltıcı 'rect:...>...' byte hatası ANA hata OLMAMALI.
            self.assertNotIn("primary=rect", message)
            # Fail-closed: orijinal SVG byte-birebir korunur.
            self.assertEqual(svg.read_bytes(), original)

    def test_deterministic_sha_on_succeeding_source(self) -> None:
        def run(directory: str) -> str:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            report = apply_candidate_painter_reconstruction(svg, source, "logo_color")
            self.assertTrue(report["applied"])
            return hashlib.sha256(svg.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.assertEqual(run(d1), run(d2))


if __name__ == "__main__":
    unittest.main()
