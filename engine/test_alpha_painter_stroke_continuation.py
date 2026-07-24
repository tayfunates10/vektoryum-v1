from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.alpha_candidate_painter import apply_candidate_painter_reconstruction
from engine.test_alpha_painter_ledger import (
    _capture_attempts,
    _covering_parent,
    _simple_soft_disc_source,
)


class PainterStrokeContinuationTests(unittest.TestCase):
    def test_retry_eligible_journal_rejection_tries_next_stroke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            calls: list[int] = []

            def journal(*_args, **_kwargs):
                calls.append(len(calls))
                if len(calls) == 1:
                    return False, ["topology_hole_regression", "seam_regression"]
                return True, []

            with patch(
                "app.alpha_candidate_painter._run_painter_geometry_journal",
                side_effect=journal,
            ):
                report, ledgers, error = _capture_attempts(
                    lambda: apply_candidate_painter_reconstruction(
                        svg, source, "logo_color"
                    )
                )

            self.assertIsNone(error)
            self.assertTrue(report["applied"])
            polygon = [
                entry for entry in ledgers[0]
                if entry["encoding_family"] == "polygon"
            ]
            self.assertEqual(
                [entry["stroke_width"] for entry in polygon], [1.0, 1.5, 2.0]
            )
            self.assertEqual(polygon[1]["status"], "geometry_rejected")
            self.assertEqual(
                polygon[1]["journal_reason_codes"],
                ["topology_hole_regression", "seam_regression"],
            )
            self.assertEqual(polygon[2]["status"], "accepted")
            self.assertTrue(polygon[2]["journal_passed"])
            self.assertGreaterEqual(len(calls), 2)

    def test_noneligible_journal_rejection_stops_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            original = _covering_parent(svg, 96)

            with patch(
                "app.alpha_candidate_painter._run_painter_geometry_journal",
                return_value=(False, ["node_complexity_explosion"]),
            ):
                _report, ledgers, error = _capture_attempts(
                    lambda: apply_candidate_painter_reconstruction(
                        svg, source, "logo_color"
                    )
                )

            self.assertIsInstance(error, RuntimeError)
            self.assertIn("node_complexity_explosion", str(error))
            self.assertEqual(svg.read_bytes(), original)
            validated = [entry for entry in ledgers[0] if entry["validation_started"]]
            self.assertTrue(validated)
            self.assertTrue(all(entry["stroke_width"] == 1.5 for entry in validated))

    def test_mixed_reason_set_is_not_retry_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            original = _covering_parent(svg, 96)
            mixed = ["topology_hole_regression", "node_complexity_explosion"]

            with patch(
                "app.alpha_candidate_painter._run_painter_geometry_journal",
                return_value=(False, mixed),
            ):
                _report, ledgers, error = _capture_attempts(
                    lambda: apply_candidate_painter_reconstruction(
                        svg, source, "logo_color"
                    )
                )

            self.assertIsInstance(error, RuntimeError)
            self.assertIn("node_complexity_explosion", str(error))
            self.assertEqual(svg.read_bytes(), original)
            validated = [entry for entry in ledgers[0] if entry["validation_started"]]
            self.assertTrue(validated)
            self.assertTrue(all(entry["stroke_width"] == 1.5 for entry in validated))
            self.assertTrue(all(entry["journal_reason_codes"] == mixed for entry in validated))

    def test_retry_path_is_deterministic(self) -> None:
        def run_once(root: Path):
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            calls: list[int] = []

            def journal(*_args, **_kwargs):
                calls.append(len(calls))
                if len(calls) == 1:
                    return False, ["topology_hole_regression", "seam_regression"]
                return True, []

            with patch(
                "app.alpha_candidate_painter._run_painter_geometry_journal",
                side_effect=journal,
            ):
                report, ledgers, error = _capture_attempts(
                    lambda: apply_candidate_painter_reconstruction(
                        svg, source, "logo_color"
                    )
                )
            self.assertIsNone(error)
            return (
                hashlib.sha256(svg.read_bytes()).hexdigest(),
                report["painter_encoding_label"],
                report["candidate_support_stroke_width_pixels"],
                report["painter_attempts_sha256"],
                ledgers[0],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            self.assertEqual(run_once(first), run_once(second))


if __name__ == "__main__":
    unittest.main()
