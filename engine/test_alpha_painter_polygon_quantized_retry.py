from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_painter import (
    _NODE_FREE_QUANTIZED_FAMILY,
    _NODE_FREE_QUANTIZED_LEVELS,
    _node_free_polygon_retry_eligible,
    _painter_loops,
    _painter_polygon_children,
    _requantize_alpha,
)
from app.alpha_svg_mask import _quantize_alpha

_SVG_NS = "http://www.w3.org/2000/svg"


def _qname(name: str) -> str:
    return f"{{{_SVG_NS}}}{name}"


def _dense_alpha_ramp() -> np.ndarray:
    row = np.arange(256, dtype=np.uint8)
    return np.tile(row[None, :], (256, 1))


def _polygon_bytes(quantized: np.ndarray, opacity: dict[int, float]) -> int:
    children = _painter_polygon_children(
        _painter_loops(quantized, opacity), _qname
    )
    return sum(len(ET.tostring(child)) for child in children)


class NodeFreePolygonQuantizationTests(unittest.TestCase):
    def test_family_does_not_pollute_existing_quantized_ledger_contract(self) -> None:
        self.assertNotEqual(_NODE_FREE_QUANTIZED_FAMILY, "quantized")

    def test_coarser_lattice_strictly_reduces_polygon_bytes(self) -> None:
        alpha = _dense_alpha_ramp()
        exact_quantized, exact_opacity = _quantize_alpha(alpha)
        baseline = _polygon_bytes(exact_quantized, exact_opacity)
        previous = baseline
        for levels in _NODE_FREE_QUANTIZED_LEVELS:
            requantized, opacity = _requantize_alpha(alpha, levels)
            current = _polygon_bytes(requantized, opacity)
            self.assertLess(current, previous, f"polygon-q{levels}")
            previous = current
        self.assertLess(previous, baseline * 0.25)

    def test_quantized_polygon_emits_no_path_elements(self) -> None:
        alpha = _dense_alpha_ramp()
        for levels in _NODE_FREE_QUANTIZED_LEVELS:
            requantized, opacity = _requantize_alpha(alpha, levels)
            children = _painter_polygon_children(
                _painter_loops(requantized, opacity), _qname
            )
            self.assertTrue(children)
            self.assertTrue(
                all(str(child.tag).endswith("}polygon") for child in children)
            )


class NodeFreePolygonRetryGateTests(unittest.TestCase):
    @staticmethod
    def _polygon_exact(**updates):
        entry = {
            "encoding_label": "polygon",
            "encoding_family": "polygon",
            "exact_or_quantized": "exact",
            "status": "byte_rejected",
            "journal_gate_started": False,
            "journal_passed": None,
            "journal_reason_codes": [],
        }
        entry.update(updates)
        return entry

    def test_byte_only_exact_polygon_failure_is_eligible(self) -> None:
        self.assertTrue(_node_free_polygon_retry_eligible([self._polygon_exact()]))

    def test_retry_eligible_exact_polygon_journal_failure_is_eligible(self) -> None:
        self.assertTrue(
            _node_free_polygon_retry_eligible(
                [self._polygon_exact(
                    status="geometry_rejected",
                    journal_gate_started=True,
                    journal_passed=False,
                    journal_reason_codes=["topology_hole_regression", "seam_regression"],
                )]
            )
        )

    def test_noneligible_exact_polygon_journal_failure_blocks_retry(self) -> None:
        self.assertFalse(
            _node_free_polygon_retry_eligible(
                [self._polygon_exact(
                    status="geometry_rejected",
                    journal_gate_started=True,
                    journal_passed=False,
                    journal_reason_codes=["node_complexity_explosion"],
                )]
            )
        )

    def test_mixed_exact_polygon_journal_failure_blocks_retry(self) -> None:
        self.assertFalse(
            _node_free_polygon_retry_eligible(
                [self._polygon_exact(
                    status="geometry_rejected",
                    journal_gate_started=True,
                    journal_passed=False,
                    journal_reason_codes=[
                        "topology_hole_regression",
                        "node_complexity_explosion",
                    ],
                )]
            )
        )

    def test_noneligible_contour_does_not_veto_byte_only_polygon_retry(self) -> None:
        attempts = [
            self._polygon_exact(),
            {
                "encoding_label": "contour",
                "encoding_family": "contour",
                "exact_or_quantized": "exact",
                "status": "geometry_rejected",
                "journal_gate_started": True,
                "journal_passed": False,
                "journal_reason_codes": ["node_complexity_explosion"],
            },
        ]
        self.assertTrue(_node_free_polygon_retry_eligible(attempts))


if __name__ == "__main__":
    unittest.main()
