"""FAZ 3A — provenance-farkında sanat kimliği testleri.

Kör toplam path/node eşitliği yerine, sanat eserinin geometri+renk kimliği
kanonik parmak izi ile korunur; transform-owned maske/destek geometrisi (bu
işleme etiketli) parmak izinden dışlanır. Kimlik değişirse painter reddedilir;
toplam karmaşıklık ve görünüm değişmemiş journal kapılarına aittir.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_artwork_identity import (
    PROVENANCE_OWNER,
    PROVENANCE_OWNER_ATTR,
    PROVENANCE_ROLE_ATTR,
    PROVENANCE_TXN_ATTR,
    ROLE_ARTWORK_CONTAINER,
    ROLE_MASK_GEOMETRY,
    alpha_transaction_id,
    artwork_fingerprint,
)
from app.alpha_candidate_painter import apply_candidate_painter_reconstruction

_TXN = "test-transaction-0001"


def _root(body: str) -> ET.Element:
    return ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'width="128" height="96" viewBox="0 0 128 96">' + body + "</svg>"
    )


def _fp(body: str, txn: str = _TXN, excluded=()) -> str:
    return artwork_fingerprint(_root(body), txn, excluded)


# --- painter'ı tetikleyen sentetik kaynak + parent (mevcut testlerden) ---
def _soft_ring_source(path: Path) -> None:
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


def _covering_candidate(path: Path) -> bytes:
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


_ARTWORK = (
    '<path d="M10 10H60V60H10Z" fill="#123456"/>'
    '<path d="M20 20H40V40H20Z" fill="#abcdef"/>'
)


class ArtworkFingerprintTests(unittest.TestCase):
    def test_1_deterministic_same_fingerprint(self) -> None:
        self.assertEqual(_fp(_ARTWORK), _fp(_ARTWORK))

    def test_2_path_d_change_rejects(self) -> None:
        changed = _ARTWORK.replace("M10 10H60V60H10Z", "M10 10H61V60H10Z")
        self.assertNotEqual(_fp(_ARTWORK), _fp(changed))

    def test_3_fill_change_rejects(self) -> None:
        changed = _ARTWORK.replace('fill="#123456"', 'fill="#654321"')
        self.assertNotEqual(_fp(_ARTWORK), _fp(changed))

    def test_4_stroke_only_stroke_change_rejects(self) -> None:
        base = '<path d="M5 5H50V50H5Z" fill="none" stroke="#111111" stroke-width="2"/>'
        changed = base.replace('stroke="#111111"', 'stroke="#222222"')
        self.assertNotEqual(_fp(base), _fp(changed))

    def test_5_filled_support_stroke_is_excluded(self) -> None:
        # Dolgulu geometride painter destek çizgisi (stroke=fill) transform-owned.
        plain = '<path d="M5 5H50V50H5Z" fill="#123456"/>'
        supported = (
            '<path d="M5 5H50V50H5Z" fill="#123456" stroke="#123456" '
            'stroke-width="1.5" paint-order="stroke fill markers"/>'
        )
        self.assertEqual(_fp(plain), _fp(supported))

    def test_6_transform_change_rejects(self) -> None:
        base = '<path d="M5 5H50V50H5Z" fill="#123456" transform="translate(1 2)"/>'
        changed = base.replace("translate(1 2)", "translate(1 3)")
        self.assertNotEqual(_fp(base), _fp(changed))

    def test_7_render_order_change_rejects(self) -> None:
        swapped = (
            '<path d="M20 20H40V40H20Z" fill="#abcdef"/>'
            '<path d="M10 10H60V60H10Z" fill="#123456"/>'
        )
        self.assertNotEqual(_fp(_ARTWORK), _fp(swapped))

    def test_8_gradient_stop_change_rejects(self) -> None:
        base = (
            '<defs><linearGradient id="g"><stop offset="0" stop-color="#ff0000"/>'
            '<stop offset="1" stop-color="#0000ff"/></linearGradient></defs>'
            '<path d="M5 5H50V50H5Z" fill="url(#g)"/>'
        )
        changed = base.replace('stop-color="#0000ff"', 'stop-color="#00ff00"')
        self.assertNotEqual(_fp(base), _fp(changed))

    def test_9_gradient_transform_change_rejects(self) -> None:
        base = (
            '<defs><linearGradient id="g" gradientTransform="rotate(10)">'
            '<stop offset="0" stop-color="#ff0000"/></linearGradient></defs>'
            '<path d="M5 5H50V50H5Z" fill="url(#g)"/>'
        )
        changed = base.replace("rotate(10)", "rotate(20)")
        self.assertNotEqual(_fp(base), _fp(changed))

    def test_10_polygon_points_change_rejects(self) -> None:
        base = '<polygon points="0,0 10,0 10,10" fill="#123456"/>'
        changed = '<polygon points="0,0 11,0 10,10" fill="#123456"/>'
        self.assertNotEqual(_fp(base), _fp(changed))

    def test_11_alpha_surface_excluded(self) -> None:
        plain = '<path d="M5 5H50V50H5Z" fill="#123456"/>'
        with_alpha = (
            '<path d="M5 5H50V50H5Z" fill="#123456" opacity="0.5" '
            'fill-opacity="0.3"/>'
        )
        self.assertEqual(_fp(plain), _fp(with_alpha))

    def test_12_tagged_transform_geometry_excluded(self) -> None:
        tagged = (
            _ARTWORK
            + f'<path d="M0 0H1V1H0Z" fill="#fff" '
            f'{PROVENANCE_OWNER_ATTR}="{PROVENANCE_OWNER}" '
            f'{PROVENANCE_ROLE_ATTR}="{ROLE_MASK_GEOMETRY}" '
            f'{PROVENANCE_TXN_ATTR}="{_TXN}"/>'
        )
        self.assertEqual(_fp(_ARTWORK), _fp(tagged))

    def test_13_untagged_extra_path_rejects(self) -> None:
        extra = _ARTWORK + '<path d="M0 0H1V1H0Z" fill="#fff"/>'
        self.assertNotEqual(_fp(_ARTWORK), _fp(extra))

    def test_14_fake_owner_wrong_transaction_not_excluded(self) -> None:
        # Farklı (sahte/eski) transaction id'li owner etiketi HARİÇ TUTULMAZ.
        fake = (
            _ARTWORK
            + f'<path d="M0 0H1V1H0Z" fill="#fff" '
            f'{PROVENANCE_OWNER_ATTR}="{PROVENANCE_OWNER}" '
            f'{PROVENANCE_ROLE_ATTR}="{ROLE_MASK_GEOMETRY}" '
            f'{PROVENANCE_TXN_ATTR}="a-different-transaction"/>'
        )
        self.assertNotEqual(_fp(_ARTWORK), _fp(fake))

    def test_15_artwork_container_is_position_transparent(self) -> None:
        # Sanat eseri, transform-owned artwork-container İÇİNDE olsa da kök
        # seviyesindeki ile aynı parmak izini vermeli (konum-bağımsız).
        contained = (
            "<defs>"
            f'<g id="p" {PROVENANCE_OWNER_ATTR}="{PROVENANCE_OWNER}" '
            f'{PROVENANCE_ROLE_ATTR}="{ROLE_ARTWORK_CONTAINER}" '
            f'{PROVENANCE_TXN_ATTR}="{_TXN}">' + _ARTWORK + "</g>"
            "</defs>"
        )
        self.assertEqual(_fp(_ARTWORK), _fp(contained))

    def test_16_excluded_canvas_matches_removed_canvas(self) -> None:
        with_canvas = '<path d="M0 0H128V96H0Z" fill="#fff"/>' + _ARTWORK
        canvas_root = _root(with_canvas)
        canvas_element = list(canvas_root)[0]
        parent_fp = artwork_fingerprint(canvas_root, _TXN, (canvas_element,))
        self.assertEqual(parent_fp, _fp(_ARTWORK))

    def test_17_transaction_id_deterministic_and_input_sensitive(self) -> None:
        a = alpha_transaction_id("psha", "asha", "logo_color", "polygon")
        b = alpha_transaction_id("psha", "asha", "logo_color", "polygon")
        self.assertEqual(a, b)
        self.assertNotEqual(
            a, alpha_transaction_id("psha", "asha", "logo_color", "rect")
        )
        self.assertNotEqual(
            a, alpha_transaction_id("psha", "asha", "geometric_logo", "polygon")
        )
        self.assertNotEqual(
            a, alpha_transaction_id("OTHER", "asha", "logo_color", "polygon")
        )

    def test_18_referenced_use_target_included(self) -> None:
        base = (
            '<defs><path id="s" d="M5 5H50V50H5Z" fill="#123456"/></defs>'
            '<use xlink:href="#s"/>'
        )
        changed = base.replace("M5 5H50V50H5Z", "M5 5H51V50H5Z")
        self.assertNotEqual(_fp(base), _fp(changed))


class PainterArtworkIdentityIntegrationTests(unittest.TestCase):
    def _run(self, directory: str):
        root = Path(directory)
        source_path = root / "source.png"
        svg_path = root / "candidate.svg"
        _soft_ring_source(source_path)
        original = _covering_candidate(svg_path)
        report = apply_candidate_painter_reconstruction(
            svg_path, source_path, "logo_color"
        )
        return svg_path, original, report

    def test_painter_reports_provenance_identity_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _svg, _original, report = self._run(directory)
            self.assertTrue(report["applied"])
            self.assertTrue(report["artwork_identity_preserved"])
            self.assertEqual(
                report["artwork_identity_authority"], "provenance_fingerprint"
            )

    def test_painter_output_artwork_matches_parent_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            svg_path, original, _report = self._run(directory)
            # Parent (knockout edilecek tuval hariç) ile aday sanat parmak izi eşit.
            parent_root = ET.fromstring(original)
            canvas = list(parent_root)[0]  # tam-tuval beyaz zemin
            txn_polygon = alpha_transaction_id(
                hashlib.sha256(original).hexdigest(),
                "ignored",
                "logo_color",
                "polygon",
            )
            # Aday tarafında transaction id build sırasında üretildi; parmak izi
            # id'den bağımsız olduğundan (owned node yalnız aday tarafında) aynı
            # sanat forest'ı üzerinden eşleşmeli.
            candidate_root = ET.parse(svg_path).getroot()
            container = next(
                element
                for element in candidate_root.iter()
                if element.get(PROVENANCE_ROLE_ATTR) == ROLE_ARTWORK_CONTAINER
            )
            candidate_txn = container.get(PROVENANCE_TXN_ATTR)
            parent_fp = artwork_fingerprint(parent_root, candidate_txn, (canvas,))
            candidate_fp = artwork_fingerprint(candidate_root, candidate_txn)
            self.assertEqual(parent_fp, candidate_fp)

    def test_painter_is_deterministic_sha(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            svg1, _o1, _r1 = self._run(d1)
            svg2, _o2, _r2 = self._run(d2)
            self.assertEqual(
                hashlib.sha256(svg1.read_bytes()).hexdigest(),
                hashlib.sha256(svg2.read_bytes()).hexdigest(),
            )

    def test_painter_rollback_byte_identical_on_opaque_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            Image.new("RGBA", (128, 96), (10, 20, 30, 255)).save(source_path)
            original = _covering_candidate(svg_path)
            with self.assertRaises(RuntimeError):
                apply_candidate_painter_reconstruction(
                    svg_path, source_path, "logo_color"
                )
            # Başarısızlıkta orijinal SVG byte-birebir korunur.
            self.assertEqual(svg_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
