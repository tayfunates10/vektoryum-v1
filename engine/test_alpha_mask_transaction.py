from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_mask_budget import wrap_apply_source_alpha_mask
from app.alpha_svg_mask import _painter_retry_eligible


class AlphaPainterRetryEligibilityTests(unittest.TestCase):
    """Journal reddi kodlarına göre painter yeniden-inşa tetiklemesi (FAZ 3C).

    Painter DENEME kapsamı doğrudan geometri/ölçek-AA reddleridir: ``topology_*``
    (bileşen/delik) + ``seam_regression`` + ``edge_f1_regression`` +
    ``ssim_regression``. TÜM kodlar bu kapsamda olmalı; kapsam dışı tek bir kod
    (renk/palet/path/node/byte) bile varsa fail-closed kalınır. Bu yalnız DENEME
    uygunluğudur — kabul, çağıranın TAZE journal'da aynı değişmemiş kapıları
    (edge/SSIM/seam/topoloji) yeniden ölçmesine bağlıdır (geçen vakaya dokunulmaz).
    Testler üretim yardımcısını (`_painter_retry_eligible`) doğrudan çağırır.
    """

    def _eligible(self, reasons: list[str]) -> bool:
        return _painter_retry_eligible(reasons)

    def test_pure_seam_regression_triggers_painter(self) -> None:
        self.assertTrue(self._eligible(["seam_regression"]))

    def test_topology_only_still_triggers_painter(self) -> None:
        self.assertTrue(
            self._eligible(
                ["topology_component_regression", "topology_hole_regression"]
            )
        )

    def test_topology_and_seam_triggers_painter(self) -> None:
        self.assertTrue(
            self._eligible(["topology_component_regression", "seam_regression"])
        )

    def test_edge_and_ssim_now_in_painter_scope(self) -> None:
        # FAZ 3C: edge_f1/ssim ölçek-AA reddleri artık painter DENEME kapsamında
        # (kabul yine TAZE journal'ın aynı edge/SSIM kapılarına bağlı).
        self.assertTrue(self._eligible(["edge_f1_regression"]))
        self.assertTrue(self._eligible(["ssim_regression"]))
        self.assertTrue(self._eligible(["ssim_regression", "edge_f1_regression"]))
        self.assertTrue(
            self._eligible(
                [
                    "topology_component_regression",
                    "ssim_regression",
                    "edge_f1_regression",
                    "seam_regression",
                ]
            )
        )

    def test_non_repairable_reason_fails_closed(self) -> None:
        # Painter'ın gideremeyeceği bir kod (renk/palet/path/node/byte) kapsam
        # dışı geometri reddiyle KARIŞSA bile gizlenemez → fail-closed.
        self.assertFalse(self._eligible(["seam_regression", "color_regression"]))
        self.assertFalse(self._eligible(["ssim_regression", "palette_regression"]))
        self.assertFalse(self._eligible(["edge_f1_regression", "node_complexity_explosion"]))
        self.assertFalse(self._eligible(["path_count_regression"]))
        self.assertFalse(self._eligible(["byte_budget_regression"]))
        self.assertFalse(self._eligible([]))


class AlphaMaskTransactionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, bytes]:
        source_path = root / "source.png"
        svg_path = root / "selected.svg"
        source = np.zeros((32, 32, 4), dtype=np.uint8)
        source[8:24, 8:24, :3] = (214, 32, 48)
        source[8:24, 8:24, 3] = 255
        Image.fromarray(source, mode="RGBA").save(source_path)
        original = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
            'viewBox="0 0 32 32"><path fill="#d62030" '
            'd="M8 8h16v16H8Z"/></svg>'
        ).encode("utf-8")
        svg_path.write_bytes(original)
        return source_path, svg_path, original

    def test_postwrite_rejection_restores_original_svg_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, svg_path, original = self._fixture(root)

            def rejected(target: Path, source: Path, mode: str):
                self.assertEqual(source, source_path)
                self.assertEqual(mode, "logo_color")
                target.write_text("<svg>rejected mutation</svg>", encoding="utf-8")
                raise RuntimeError("simulated_postwrite_alpha_gate_failure")

            guarded = wrap_apply_source_alpha_mask(rejected)
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated_postwrite_alpha_gate_failure",
            ):
                guarded(svg_path, source_path, "logo_color")

            self.assertEqual(svg_path.read_bytes(), original)
            self.assertEqual(
                list(root.glob(".*.alpha-rollback.svg")),
                [],
            )

    def test_accepted_write_commits_and_removes_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, svg_path, original = self._fixture(root)
            accepted_bytes = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
                'viewBox="0 0 32 32"><g data-accepted="true"/></svg>'
            ).encode("utf-8")

            def accepted(target: Path, source: Path, mode: str):
                del source, mode
                target.write_bytes(accepted_bytes)
                return {"applied": True, "status": "accepted"}

            guarded = wrap_apply_source_alpha_mask(accepted)
            report = guarded(svg_path, source_path, "logo_color")

            self.assertNotEqual(svg_path.read_bytes(), original)
            self.assertEqual(svg_path.read_bytes(), accepted_bytes)
            self.assertEqual(report["rollback_guard"], "armed_and_committed")
            self.assertGreater(report["preflight_rectangle_limit"], 0)
            self.assertEqual(
                list(root.glob(".*.alpha-rollback.svg")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
