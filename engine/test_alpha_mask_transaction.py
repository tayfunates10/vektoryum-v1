from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_mask_budget import wrap_apply_source_alpha_mask
from app.alpha_svg_mask import _is_painter_geometry_reason


class AlphaPainterRetryEligibilityTests(unittest.TestCase):
    """Journal reddi kodlarına göre painter yeniden-inşa tetiklemesi.

    Painter yalnız doğrudan geometri reddlerinde (topoloji ya da dikiş) denenir:
    TÜM kodlar geometri reddi olmalı. Aksi halde fail-closed kalınır — painter
    renk/karmaşıklık ya da SSIM/edge reddini gideremez ve bunlar gizlenmemeli.
    Değişiklik yalnız zaten reddedilmiş vakalarda çalışır; geçen vakalara dokunmaz.
    """

    def _eligible(self, reasons: list[str]) -> bool:
        return bool(reasons) and all(
            _is_painter_geometry_reason(reason) for reason in reasons
        )

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

    def test_seam_with_induced_ssim_edge_fails_closed(self) -> None:
        # Onaylanan kapsam yalnız seam/topoloji: SSIM/edge karışan reddler painter
        # kapsamına alınmaz, fail-closed kalır (eşik/politika değişmez).
        self.assertFalse(
            self._eligible(
                [
                    "topology_component_regression",
                    "ssim_regression",
                    "edge_f1_regression",
                ]
            )
        )
        self.assertFalse(self._eligible(["seam_regression", "ssim_regression"]))

    def test_pure_ssim_regression_fails_closed(self) -> None:
        # Geometri reddi yok; painter maskesi bir şey değiştirmez → fail-closed.
        self.assertFalse(self._eligible(["ssim_regression"]))
        self.assertFalse(self._eligible(["ssim_regression", "edge_f1_regression"]))

    def test_non_repairable_reason_fails_closed(self) -> None:
        # Painter'ın gideremeyeceği bir kod (ör. renk/karmaşıklık) gizlenemez.
        self.assertFalse(
            self._eligible(["seam_regression", "color_regression"])
        )
        self.assertFalse(self._eligible(["path_count_regression"]))
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
