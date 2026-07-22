"""FAZ 1: ölçüm-kapılı byte-identical no-op.

Seçili SVG kaynak alfayı zaten doğru üretiyorsa gereksiz vektör maske eklenmez.
No-op yalnız alfa IoU/MAE kapıları geçerken VE opak-tuval-collapse yokken kabul
edilir; aksi halde normal maskeleme yoluna fail-closed düşülür. Dosya adı / moda
göre tahmin yoktur.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_svg_mask import (
    _source_alpha_already_satisfied,
    wrap_run_pipeline_with_alpha_mask,
)

_MODE = "geometric_logo"


def _center_square_source(path: Path, side: int = 256) -> np.ndarray:
    """Şeffaf kenarlıklı, opak merkezli kare (kaynak alfa)."""
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    lo, hi = side // 4, (side * 3) // 4
    rgba[lo:hi, lo:hi, :3] = (200, 30, 40)
    rgba[lo:hi, lo:hi, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)
    return rgba


def _svg(side: int, body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}">{body}</svg>'
    ).encode("utf-8")


def _matching_parent(path: Path, side: int = 256) -> bytes:
    """Kaynakla aynı merkez kareyi çizen, ŞEFFAF arka planlı parent (alfa doğru)."""
    lo, hi = side // 4, (side * 3) // 4
    data = _svg(side, f'<path d="M{lo} {lo}H{hi}V{hi}H{lo}Z" fill="#C81E28"/>')
    path.write_bytes(data)
    return data


def _collapsed_parent(path: Path, side: int = 256) -> bytes:
    """Tam-tuval opak beyaz zemin + merkez kare (opaque-canvas-collapse imzası)."""
    lo, hi = side // 4, (side * 3) // 4
    data = _svg(
        side,
        f'<path d="M0 0H{side}V{side}H0Z" fill="#FFFFFF"/>'
        f'<path d="M{lo} {lo}H{hi}V{hi}H{lo}Z" fill="#C81E28"/>',
    )
    path.write_bytes(data)
    return data


def _wrong_alpha_parent(path: Path, side: int = 256) -> bytes:
    """Kaynakla eşleşmeyen küçük, kaydırılmış kare (alfa yanlış, collapse yok)."""
    data = _svg(side, f'<path d="M8 8H40V40H8Z" fill="#C81E28"/>')
    path.write_bytes(data)
    return data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AlphaNoopPreflightTests(unittest.TestCase):
    def test_alpha_already_correct_is_byte_identical_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _center_square_source(source)
            original = _matching_parent(parent)
            before_sha = _sha(parent)
            before_size = parent.stat().st_size

            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNotNone(report, "alfa zaten doğruyken no-op beklenir")
            assert report is not None
            self.assertEqual(report["status"], "alpha_already_satisfied_noop")
            self.assertFalse(report["applied"])
            # SVG'ye dokunulmaz: byte/SHA birebir aynı.
            self.assertEqual(parent.read_bytes(), original)
            self.assertEqual(_sha(parent), before_sha)
            self.assertEqual(parent.stat().st_size, before_size)
            self.assertEqual(report["before_byte_size"], report["after_byte_size"])
            self.assertEqual(report["before_sha256"], report["after_sha256"])
            self.assertEqual(report["after_sha256"], before_sha)
            # Kapıları gerçekten geçmiş ve collapse yok.
            self.assertGreaterEqual(
                report["noop_min_alpha_iou"], report["noop_alpha_iou_min"]
            )
            self.assertLessEqual(
                report["noop_max_alpha_mae"], report["noop_alpha_mae_max"]
            )
            self.assertLess(report["noop_max_render_coverage"], 0.98)
            self.assertTrue(report["candidate_identity_preserved"])

    def test_wrong_alpha_does_not_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _center_square_source(source)
            _wrong_alpha_parent(parent)
            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNone(report, "alfa yanlışken no-op YAPILMAMALI")

    def test_opaque_canvas_collapse_does_not_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _center_square_source(source)
            _collapsed_parent(parent)
            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNone(
                report, "opaque-canvas-collapse'ta no-op YAPILMAMALI"
            )

    def test_opaque_source_does_not_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            side = 256
            Image.new("RGBA", (side, side), (200, 30, 40, 255)).save(source)
            _matching_parent(parent, side)
            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNone(report, "tam opak kaynakta no-op yolu çalışmaz")

    def test_noop_report_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _center_square_source(source)
            _matching_parent(parent)
            first = _source_alpha_already_satisfied(parent, source, _MODE)
            second = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertEqual(first, second)

    def test_two_render_sizes_when_viewbox_exceeds_eval(self) -> None:
        # viewBox > 512 → hem bounded eval (512) hem renderer-native ölçülür.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            side = 900
            _center_square_source(source, side)
            _matching_parent(parent, side)
            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(
                report["noop_render_sizes"], [[512, 512], [900, 900]]
            )

    def test_gradient_parent_preserved_through_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            side = 256
            _center_square_source(source, side)
            lo, hi = side // 4, (side * 3) // 4
            original = _svg(
                side,
                '<defs><linearGradient id="g" x1="0" x2="1">'
                '<stop offset="0" stop-color="#C81E28"/>'
                '<stop offset="1" stop-color="#7A0E14"/></linearGradient></defs>'
                f'<path d="M{lo} {lo}H{hi}V{hi}H{lo}Z" fill="url(#g)"/>',
            )
            parent.write_bytes(original)
            report = _source_alpha_already_satisfied(parent, source, _MODE)
            self.assertIsNotNone(report)
            # Gradient stop'ları ve fill birebir korunur (dosyaya yazılmaz).
            self.assertEqual(parent.read_bytes(), original)
            text = parent.read_bytes()
            self.assertIn(b'<stop offset="0" stop-color="#C81E28"', text)
            self.assertIn(b'fill="url(#g)"', text)

    def test_wrapper_returns_noop_without_touching_selected_svg(self) -> None:
        # Sarmalayıcı seviyesinde: no-op'ta best değişmez, alpha_mask_report
        # 'alpha_already_satisfied_noop' olur ve SVG byte'ları korunur.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _center_square_source(source)
            original = _matching_parent(parent)

            def fake_pipeline(image, original_path, trace_mode, job_dir,
                              refine=True, edge_cleanup=True):
                return {
                    "best": {"svg_path": str(parent), "name": "geo_standard"},
                    "mode_used": _MODE,
                }

            wrapped = wrap_run_pipeline_with_alpha_mask(fake_pipeline)
            with Image.open(source) as image:
                result = wrapped(image, source, "auto", root)
            self.assertEqual(
                result["alpha_mask_report"]["status"],
                "alpha_already_satisfied_noop",
            )
            self.assertEqual(result["best"]["svg_path"], str(parent))
            self.assertEqual(parent.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
