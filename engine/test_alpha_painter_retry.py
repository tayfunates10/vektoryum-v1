"""FAZ 3C — painter yeniden-inşa DENEME kapsamının journal redlerine genişletilmesi.

Kapsam genişler ama KABUL eşikleri değişmez. Painter yalnız doğrudan geometri/
ölçek-AA reddlerinde (topology_*/seam/edge_f1/ssim) DENENİR; kabul, çağıranın TAZE
bir journal'da aynı DEĞİŞMEMİŞ SSIM/edge/seam/topoloji/byte/path/node kapılarını
yeniden ölçmesine bağlıdır. Kapsam dışı tek bir kod (renk/palet/path/node/byte) bile
varsa fail-closed kalınır. Geçen (kabul edilen) vakada painter çağrılmaz. Painter
onaramazsa ilk orijinal journal reddi korunur ve parent SVG byte-birebir bırakılır.

İki grup test:
1. Uygunluk (saf): ``_painter_retry_eligible`` / ``_is_painter_geometry_reason``.
2. Çağırma-kapısı davranışı: gerçek ``wrap_run_pipeline_with_alpha_mask`` sarmalayıcısı,
   journal reddi ve painter sonucu kontrollü biçimde taklit edilerek sürülür.
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from app.alpha_svg_mask import (
    _is_painter_geometry_reason,
    _painter_retry_eligible,
    wrap_run_pipeline_with_alpha_mask,
)


class PainterRetryEligibilityTests(unittest.TestCase):
    """FAZ 3C uygunluk sözleşmesi (saf karar fonksiyonu)."""

    def test_direct_geometry_scale_aa_codes_are_eligible(self) -> None:
        for reason in (
            "seam_regression",
            "edge_f1_regression",
            "ssim_regression",
            "topology_component_regression",
            "topology_hole_regression",
        ):
            self.assertTrue(
                _is_painter_geometry_reason(reason), f"{reason} kapsamda olmalı"
            )
            self.assertTrue(_painter_retry_eligible([reason]))

    def test_combined_in_scope_codes_are_eligible(self) -> None:
        self.assertTrue(
            _painter_retry_eligible(
                [
                    "topology_component_regression",
                    "topology_hole_regression",
                    "seam_regression",
                    "edge_f1_regression",
                    "ssim_regression",
                ]
            )
        )
        self.assertTrue(
            _painter_retry_eligible(["ssim_regression", "edge_f1_regression"])
        )

    def test_out_of_scope_codes_are_not_eligible(self) -> None:
        for reason in (
            "color_regression",
            "palette_regression",
            "path_count_regression",
            "node_complexity_explosion",
            "byte_budget_regression",
            "unknown",
            "",
        ):
            self.assertFalse(
                _is_painter_geometry_reason(reason), f"{reason} kapsam dışı olmalı"
            )
            self.assertFalse(_painter_retry_eligible([reason]))

    def test_any_out_of_scope_code_fails_closed(self) -> None:
        # Kapsam içi + kapsam dışı karışımı: painter gideremeyeceği hatayı gizlemesin.
        self.assertFalse(
            _painter_retry_eligible(["ssim_regression", "color_regression"])
        )
        self.assertFalse(
            _painter_retry_eligible(["edge_f1_regression", "path_count_regression"])
        )
        self.assertFalse(
            _painter_retry_eligible(["seam_regression", "node_complexity_explosion"])
        )

    def test_empty_reason_set_fails_closed(self) -> None:
        self.assertFalse(_painter_retry_eligible([]))
        self.assertFalse(_painter_retry_eligible(None))


# --------------------------------------------------------------------------- #
# Çağırma-kapısı davranış testleri.                                           #
# --------------------------------------------------------------------------- #
def _partial_alpha_source(path: Path, side: int = 64) -> None:
    """Kısmi alfa kaynağı: ne tümüyle opak ne tümüyle şeffaf → maske akışına girer."""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = cy = side / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((side * 0.4 - dist) * 24.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[:, :, :3] = (200, 30, 40)
    rgba[:, :, 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(path)


def _color_parent(path: Path, side: int = 64) -> bytes:
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}">'
        f'<path d="M0 0H{side}V{side}H0Z" fill="#C81E28"/></svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


def _make_fake_journal(outcomes: list[tuple[str, list[str]]], calls: list) -> type:
    """``TransformJournal`` yerine geçen sahte sınıf: consider_candidate sonuçları
    ``outcomes`` sırasıyla belirlenir ('accept' → aday kabul; 'reject' → parent'a
    dön + reason_codes). Her çağrı ``calls``a kaydedilir."""

    class _FakeJournal:
        def __init__(self, parent, source_rgb, image_class=None, required_metrics=None):
            self._parent = Path(parent)

        def consider_candidate(self, stage_id, parent, candidate, transform_report=None):
            index = len(calls)
            calls.append((stage_id, Path(candidate)))
            kind, reasons = outcomes[index] if index < len(outcomes) else ("accept", [])
            if kind == "accept":
                return Path(candidate), {"status": "accepted", "reason_codes": []}
            return Path(parent), {"status": "rejected", "reason_codes": list(reasons)}

        def to_dict(self):
            return {"chain_valid": True}

    return _FakeJournal


class PainterRetryInvocationGateTests(unittest.TestCase):
    """Gerçek sarmalayıcıyı sürerek painter'ın DENENİP denenmediğini doğrular."""

    def _drive(
        self,
        outcomes: list[tuple[str, list[str]]],
        painter_side_effect,
    ):
        """Sarmalayıcıyı taklit patch'leriyle çalıştır; (exception|None, painter_mock,
        parent_bytes_after, finalized_exists, journal_calls) döner."""
        calls: list = []
        fake_journal = _make_fake_journal(outcomes, calls)
        painter = MagicMock(side_effect=painter_side_effect)

        def fake_apply_source_alpha_mask(target, source, mode):
            Path(target).write_text("<svg><!--primary-mask--></svg>", encoding="utf-8")
            return {"applied": True, "status": "accepted", "schema": "test-primary"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            parent = root / "parent.svg"
            _partial_alpha_source(source)
            parent_original = _color_parent(parent)

            def fake_pipeline(image, original_path, trace_mode, job_dir,
                              refine=True, edge_cleanup=True):
                return {
                    "best": {"svg_path": str(parent), "name": "logo_gradient"},
                    "mode_used": "logo_color",
                    "analysis": {},
                    "transform_journal": {},
                    "scored": [],
                    "selection_reason": "selected",
                }

            wrapped = wrap_run_pipeline_with_alpha_mask(fake_pipeline)
            raised: BaseException | None = None
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch("app.alpha_svg_mask.apply_source_alpha_mask",
                          fake_apply_source_alpha_mask)
                )
                stack.enter_context(
                    patch("app.alpha_svg_mask._source_alpha_already_satisfied",
                          lambda *a, **k: None)
                )
                stack.enter_context(
                    patch("app.transform_journal.TransformJournal", fake_journal)
                )
                stack.enter_context(
                    patch("app.transform_journal.merge_journal_reports",
                          lambda *a, **k: {"chain_valid": True})
                )
                stack.enter_context(
                    patch("app.alpha_candidate_painter."
                          "apply_candidate_painter_reconstruction", painter)
                )
                stack.enter_context(
                    patch("app.pipeline.score_candidate",
                          lambda *a, **k: {"rendered_ok": True, "name": "logo_gradient_alpha"})
                )
                stack.enter_context(
                    patch("app.pipeline.score_structure_integrity", lambda *a, **k: {})
                )
                with Image.open(source) as image:
                    try:
                        wrapped(image, source, "auto", root)
                    except BaseException as exc:  # noqa: BLE001
                        raised = exc
            finalized = root / f"{parent.stem}_alpha.svg"
            return (
                raised,
                painter,
                parent.read_bytes(),
                finalized.exists(),
                calls,
                parent_original,
            )

    @staticmethod
    def _painter_writes_then_report(target, source, mode):
        Path(target).write_text("<svg><!--painter--></svg>", encoding="utf-8")
        return {"applied": True, "status": "accepted", "schema": "painter"}

    def test_seam_rejection_invokes_painter(self) -> None:
        # İlk journal seam reddi → painter DENENİR. Painter çıktısı ikinci journal'da
        # yine reddedilirse fail-closed; ama painter'ın ÇAĞRILDIĞI kanıtlanır.
        raised, painter, parent_bytes, finalized_exists, calls, original = self._drive(
            [("reject", ["seam_regression"]), ("reject", ["seam_regression"])],
            self._painter_writes_then_report,
        )
        self.assertTrue(painter.called)
        self.assertEqual(len(calls), 2)  # ön journal + painter sonrası taze journal
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(parent_bytes, original)  # parent byte-birebir
        self.assertFalse(finalized_exists)  # aday atomik silindi

    def test_topology_rejection_invokes_painter(self) -> None:
        raised, painter, parent_bytes, _fx, calls, original = self._drive(
            [("reject", ["topology_component_regression"]),
             ("reject", ["topology_component_regression"])],
            self._painter_writes_then_report,
        )
        self.assertTrue(painter.called)
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(parent_bytes, original)

    def test_edge_rejection_invokes_painter_but_rejects_if_edge_still_fails(self) -> None:
        # FAZ 3C: edge_f1 reddi artık painter'ı DENER; painter çıktısı edge kapısını
        # geçmezse (ikinci journal edge reddi) KABUL EDİLMEZ, fail-closed kalır.
        raised, painter, parent_bytes, finalized_exists, _c, original = self._drive(
            [("reject", ["edge_f1_regression"]), ("reject", ["edge_f1_regression"])],
            self._painter_writes_then_report,
        )
        self.assertTrue(painter.called)
        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("edge_f1_regression", str(raised))
        self.assertEqual(parent_bytes, original)
        self.assertFalse(finalized_exists)

    def test_ssim_rejection_invokes_painter_but_rejects_if_ssim_still_fails(self) -> None:
        raised, painter, parent_bytes, finalized_exists, _c, original = self._drive(
            [("reject", ["ssim_regression"]), ("reject", ["ssim_regression"])],
            self._painter_writes_then_report,
        )
        self.assertTrue(painter.called)
        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("ssim_regression", str(raised))
        self.assertEqual(parent_bytes, original)
        self.assertFalse(finalized_exists)

    def test_passing_journal_does_not_invoke_painter(self) -> None:
        # İlk journal KABUL → geçen artifact; painter ASLA çağrılmaz.
        raised, painter, parent_bytes, _fx, calls, original = self._drive(
            [("accept", [])],
            self._painter_writes_then_report,
        )
        self.assertFalse(painter.called)
        self.assertIsNone(raised)
        self.assertEqual(len(calls), 1)  # yalnız ön journal
        self.assertEqual(parent_bytes, original)

    def test_out_of_scope_rejection_does_not_invoke_painter(self) -> None:
        # Kapsam dışı red (renk) → painter çağrılmaz; ORİJİNAL journal reddi korunur.
        raised, painter, parent_bytes, _fx, calls, original = self._drive(
            [("reject", ["color_regression"])],
            self._painter_writes_then_report,
        )
        self.assertFalse(painter.called)
        self.assertIsInstance(raised, RuntimeError)
        self.assertIn("source_alpha_mask_transform_gate_rejected", str(raised))
        self.assertIn("color_regression", str(raised))
        self.assertEqual(len(calls), 1)
        self.assertEqual(parent_bytes, original)

    def test_painter_failure_preserves_original_journal_error(self) -> None:
        # Painter (eligible seam reddinde) iç hatayla düşerse: ilk ORİJİNAL journal
        # reddi korunur (painter iç hatası değil), parent byte-birebir.
        def painter_raises(target, source, mode):
            raise RuntimeError(
                "source_alpha_candidate_painter_no_admissible_reconstruction:"
                "primary=contour:native_iou_gate_failed:0.42"
            )

        raised, painter, parent_bytes, finalized_exists, _c, original = self._drive(
            [("reject", ["seam_regression"])],
            painter_raises,
        )
        self.assertTrue(painter.called)
        self.assertIsInstance(raised, RuntimeError)
        message = str(raised)
        self.assertIn("source_alpha_mask_transform_gate_rejected:seam_regression", message)
        # İç painter hatası ANA hata OLMAMALI (orijinal journal reddi korunur).
        self.assertNotIn("no_admissible_reconstruction", message)
        self.assertEqual(parent_bytes, original)  # rollback byte-birebir
        self.assertFalse(finalized_exists)


if __name__ == "__main__":
    unittest.main()
