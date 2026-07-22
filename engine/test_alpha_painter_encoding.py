"""FAZ 3B — painter kompakt encoding turnuvası testleri.

Sabit öncelik zinciri: polygon → contour (grouped-evenodd) → contour-q128/64/32
→ rect. Her aday byte preflight'tan geçer; bütçeyi aşan render edilmeden elenir.
Bütçeye giren aday değişmemiş tam kapı bataryasından geçmelidir; kazanan zincirdeki
İLK geçen (polygon sığarsa dikiş-güvenli polygon korunur, byte-zorlandığında daha
kompakt VE dikiş-güvenli contour seçilir). Niceleme YALNIZ alfa kapıları geçerse.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_painter import (
    _requantize_alpha,
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
    def test_requantize_reduces_levels_and_keeps_transparent(self) -> None:
        alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
        quant, opacity = _requantize_alpha(alpha, 32)
        self.assertLessEqual(len(opacity), 32)
        self.assertEqual(opacity[0], 0.0)
        self.assertTrue(bool((quant[alpha == 0] == 0).all()))

    def test_simple_source_keeps_seam_safe_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            report = apply_candidate_painter_reconstruction(svg, source, "logo_color")
            self.assertTrue(report["applied"])
            # Polygon bütçeye sığar → zincirin ilk (dikiş-güvenli) adayı seçilir.
            self.assertEqual(report["painter_encoding_label"], "polygon")
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)

    def test_complex_source_selects_compact_contour_under_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            before = _covering_parent(svg, _COMPLEX_SIDE)
            report = apply_candidate_painter_reconstruction(svg, source, "logo_color")
            self.assertTrue(report["applied"])
            # Polygon bütçeyi aşar → kompakt grouped-evenodd contour seçilir.
            self.assertIn(
                report["painter_encoding_label"],
                {"contour", "contour-q128", "contour-q64", "contour-q32"},
            )
            self.assertEqual(report["reconstruction_mask_encoding"], "contour")
            # Final byte, journal byte bütçesinin altında.
            self.assertLessEqual(
                report["after_byte_size"], report["preflight_byte_limit"]
            )
            # Alfa kapıları geçti, sanat kimliği korundu.
            self.assertGreaterEqual(report["painter_native_alpha_iou"], 0.995)
            self.assertLessEqual(report["painter_native_alpha_mae"], 0.005)
            self.assertTrue(report["artwork_identity_preserved"])
            self.assertEqual(
                report["artwork_identity_authority"], "provenance_fingerprint"
            )
            # <path> maske geometrisi eklendi ama sanat kimliği değişmedi.
            data = svg.read_bytes()
            self.assertIn(b'fill-rule="evenodd"', data)

    def test_complex_source_is_deterministic_sha(self) -> None:
        def run(directory: str) -> str:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            _covering_parent(svg, _COMPLEX_SIDE)
            apply_candidate_painter_reconstruction(svg, source, "logo_color")
            return hashlib.sha256(svg.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.assertEqual(run(d1), run(d2))

    def test_contour_mask_paths_excluded_from_artwork_identity(self) -> None:
        # contour <path> maskeleri path_count'u artırır ama sanat parmak izi
        # değişmez (maske alt-ağacı FAZ 3A ile hariç). preserved != parent path
        # sayısı olabilir; kimlik yine de korunur.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            _covering_parent(svg, _COMPLEX_SIDE)
            report = apply_candidate_painter_reconstruction(svg, source, "logo_color")
            self.assertTrue(report["artwork_identity_preserved"])
            # Maske path'leri toplam sayıya girer (parent'tan fazla) ama kimlik aynı.
            self.assertGreater(
                report["preserved_path_count"], report["parent_path_count"]
            )


if __name__ == "__main__":
    unittest.main()
