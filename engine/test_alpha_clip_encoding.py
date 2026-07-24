"""FAZ 2: ikili (tek tam-opak seviye) kaynak alfa için kompakt clipPath kodlaması.

Kaynak alfa sert-kenarlı (kısmî değer yok) ve maske içeriği kimlik dönüşümündeyse
üretici, ayrıntılı `<mask>` yerine render-EŞ bir `<clipPath>` yayınlar. clipPath:
  * hiçbir `<path>` EKLEMEZ (path_count sabit),
  * maske boilerplate'ini düşürerek byte'ı azaltır,
  * gating renderer (resvg) altında sert maske ile PİKSEL-EŞ üretir.
Kısmî alfa veya ölçeklenmiş/kaydırılmış içerik bu yola girmez (maskede kalır).
Dosya adına/moda göre koşul yoktur; karar yalnız ölçülen alfa yapısındandır.
"""
from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_mask_budget import wrap_apply_source_alpha_mask
from app.alpha_mask_adaptive import make_adaptive_apply_source_alpha_mask
from app.alpha_svg_mask import apply_source_alpha_mask
from app.source_truth import render_svg_to_rgba

_MODE = "geometric_logo"
_SIDE = 192


def _production_apply():
    return wrap_apply_source_alpha_mask(
        make_adaptive_apply_source_alpha_mask(apply_source_alpha_mask)
    )


def _binary_source(path: Path) -> None:
    """Sert-kenarlı ikili alfa: merkez opak kare, kalan tam şeffaf."""
    rgba = np.zeros((_SIDE, _SIDE, 4), dtype=np.uint8)
    lo, hi = _SIDE // 4, (_SIDE * 3) // 4
    rgba[lo:hi, lo:hi, :3] = (200, 30, 40)
    rgba[lo:hi, lo:hi, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def _partial_source(path: Path) -> None:
    """Yumuşak (anti-aliased) kenar: kısmî alfa içerir → clip UYGULANAMAZ."""
    yy, xx = np.mgrid[0:_SIDE, 0:_SIDE]
    cx = cy = _SIDE / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((_SIDE * 0.32 - dist) * 32.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((_SIDE, _SIDE, 4), dtype=np.uint8)
    rgba[:, :, :3] = (200, 30, 40)
    rgba[:, :, 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(path)


def _opaque_parent(path: Path) -> None:
    """Tam-tuval opak beyaz zemin + iç kare (opaque-canvas-collapse imzası)."""
    lo, hi = _SIDE // 4, (_SIDE * 3) // 4
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SIDE}" height="{_SIDE}" '
        f'viewBox="0 0 {_SIDE} {_SIDE}">'
        f'<path d="M0 0H{_SIDE}V{_SIDE}H0Z" fill="#FFFFFF"/>'
        f'<path d="M{lo} {lo}H{hi}V{hi}H{lo}Z" fill="#C81E28"/></svg>'
    )
    path.write_bytes(data.encode("utf-8"))


def _count_paths(svg_bytes: bytes) -> int:
    return svg_bytes.count(b"<path")


class AlphaClipEncodingTests(unittest.TestCase):
    def test_binary_source_uses_clip_without_adding_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _binary_source(source)
            _opaque_parent(parent)
            before_paths = _count_paths(parent.read_bytes())

            report = _production_apply()(parent, source, _MODE)
            self.assertTrue(report["applied"])
            self.assertEqual(report["mask_encoding"], "clip")
            data = parent.read_bytes()
            # clipPath var, <mask> yok, path EKLENMEDİ.
            self.assertIn(b"<clipPath", data)
            self.assertNotIn(b"<mask", data)
            self.assertIn(b'clip-path="url(#vektoryum-source-alpha)"', data)
            self.assertEqual(_count_paths(data), before_paths)
            # Alfa kapıları geçti (üretici içi sert gate).
            self.assertGreaterEqual(report["alpha_iou"], 0.995)

    def test_clip_is_pixel_equivalent_to_hard_mask_under_resvg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            clip_parent = root / "clip.svg"
            _binary_source(source)
            _opaque_parent(clip_parent)
            report = _production_apply()(clip_parent, source, _MODE)
            self.assertEqual(report["mask_encoding"], "clip")

            clip_text = clip_parent.read_text()
            self.assertIn("<clipPath", clip_text)
            # clipPath'ten sert <mask> eşdeğeri kur (aynı rect'ler).
            import re

            rects = "".join(re.findall(r"<rect[^>]*/>", clip_text))
            clip_defs = re.search(r"<defs>.*?</defs>", clip_text, re.S).group(0)
            mask_defs = (
                '<defs><mask id="vektoryum-source-alpha" maskUnits="userSpaceOnUse" '
                'maskContentUnits="userSpaceOnUse" x="0" y="0" '
                f'width="{_SIDE}" height="{_SIDE}" style="mask-type:alpha">'
                f'<g fill="#ffffff">{rects}</g></mask></defs>'
            )
            mask_text = clip_text.replace(clip_defs, mask_defs).replace(
                'clip-path="url(#vektoryum-source-alpha)"',
                'mask="url(#vektoryum-source-alpha)"',
            )
            mask_parent = root / "mask.svg"
            mask_parent.write_text(mask_text)

            for size in (_SIDE, 256, 512):
                clip_rgba = render_svg_to_rgba(clip_parent, size, size)
                mask_rgba = render_svg_to_rgba(mask_parent, size, size)
                self.assertIsNotNone(clip_rgba)
                self.assertIsNotNone(mask_rgba)
                diff = np.abs(
                    clip_rgba.astype(np.int16) - mask_rgba.astype(np.int16)
                )
                self.assertEqual(
                    int(diff.max()), 0, f"resvg mask/clip farkı size={size}"
                )

    def test_clip_is_smaller_than_mask_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _binary_source(source)
            _opaque_parent(parent)
            before_size = parent.stat().st_size
            report = _production_apply()(parent, source, _MODE)
            after_size = parent.stat().st_size
            # clipPath eklenmiş byte, sert maske sarmalayıcısından küçük olmalı.
            # Sert maske ~410+ byte eklerken clip ~230 byte ekler.
            self.assertEqual(report["mask_encoding"], "clip")
            self.assertLess(after_size - before_size, 300)

    def test_partial_alpha_stays_on_mask_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _partial_source(source)
            _opaque_parent(parent)
            report = _production_apply()(parent, source, _MODE)
            # Kısmî alfa sert clip ile temsil edilemez → maske yolunda kalır.
            self.assertTrue(report["applied"])
            self.assertNotEqual(report["mask_encoding"], "clip")
            data = parent.read_bytes()
            self.assertIn(b"<mask", data)
            self.assertNotIn(b"<clipPath", data)

    def test_valid_svg_after_clip_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _binary_source(source)
            _opaque_parent(parent)
            _production_apply()(parent, source, _MODE)
            # Ayrıştırılabilir ve tek clipPath id'si tutarlı.
            tree = ET.parse(parent)
            clip_ids = [
                element.get("id")
                for element in tree.getroot().iter()
                if element.tag.endswith("clipPath")
            ]
            self.assertEqual(clip_ids, ["vektoryum-source-alpha"])


if __name__ == "__main__":
    unittest.main()
