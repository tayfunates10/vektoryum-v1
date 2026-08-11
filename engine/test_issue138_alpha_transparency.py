from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import (
    _path_node_counts,
    apply_candidate_geometry_knockout,
)
from app.alpha_candidate_support import make_candidate_support_reconstruction_fallback
from app.alpha_candidate_validation import validate_alpha_reconstruction_contract
from app.alpha_mask_budget import (
    _JOURNAL_BYTE_GROWTH_ABSOLUTE,
    _JOURNAL_BYTE_GROWTH_FACTOR,
    _JOURNAL_NODE_GROWTH_ABSOLUTE,
    _JOURNAL_NODE_GROWTH_FACTOR,
    _JOURNAL_PATH_GROWTH_ABSOLUTE,
    _JOURNAL_PATH_GROWTH_FACTOR,
)
from app.source_truth import render_svg_to_rgba


# This is deliberately NOT the historical qa-soft-alpha-shadow fixture.  The
# original fixture is absent from the repository and supplied uploads.  This
# canonical reproducer matches the observed production parent-derived byte
# limit (259130) and genuinely reaches the same knockout budget-rejection path.
CANONICAL_PARENT_BYTES = 9_130
CANONICAL_BYTE_LIMIT = 259_130
CANONICAL_CANDIDATE_SHA256 = (
    "7886662bd6d8edc270e1637632d2c965257cdbd86bf2815ffa299ea2aef5624f"
)
CANONICAL_SOURCE_RGBA_SHA256 = (
    "b352eecb5d71792732985bbc62a103ad32f626f663f9364431dc6991e7ff2d4d"
)


def canonical_source_rgba() -> np.ndarray:
    height, width = 64, 96
    yy, xx = np.indices((height, width))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.where(
        xx < 4,
        0,
        np.where(((xx + yy) % 2) == 0, 96, 192),
    ).astype(np.uint8)
    return rgba


def canonical_candidate_svg_bytes() -> bytes:
    prefix = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
        'viewBox="0 0 96 64">'
        '<metadata id="issue138-canonical">'
    )
    suffix = (
        "</metadata>"
        '<path d="M0 0H96V64H0Z" fill="#FFFFFF"/>'
        '<path d="M0 0H96V64H0Z" fill="#000000"/>'
        "</svg>"
    )
    fixed = (prefix + suffix).encode("utf-8")
    padding = CANONICAL_PARENT_BYTES - len(fixed)
    if padding < 0:
        raise AssertionError("canonical SVG template exceeds fixed byte contract")
    result = (prefix + ("x" * padding) + suffix).encode("utf-8")
    if len(result) != CANONICAL_PARENT_BYTES:
        raise AssertionError("canonical parent byte contract drifted")
    return result


def _write_canonical(root: Path) -> tuple[Path, Path]:
    source_path = root / "canonical-alpha.png"
    candidate_path = root / "canonical-candidate.svg"
    Image.fromarray(canonical_source_rgba(), mode="RGBA").save(source_path)
    candidate_path.write_bytes(canonical_candidate_svg_bytes())
    return source_path, candidate_path


def _identity_svg_cases() -> dict[str, str]:
    return {
        "soft_shadow": (
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
            'viewBox="0 0 96 64"><defs><radialGradient id="s">'
            '<stop offset="0" stop-color="#14213d" stop-opacity=".72"/>'
            '<stop offset=".62" stop-color="#14213d" stop-opacity=".35"/>'
            '<stop offset="1" stop-color="#14213d" stop-opacity="0"/>'
            '</radialGradient></defs><circle cx="48" cy="32" r="27" fill="url(#s)"/>'
            "</svg>"
        ),
        "semi_transparent_overlap": (
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
            'viewBox="0 0 96 64">'
            '<rect x="12" y="10" width="46" height="38" rx="6" fill="#ef7c00" opacity=".55"/>'
            '<rect x="38" y="18" width="44" height="34" rx="5" fill="#253349" opacity=".68"/>'
            "</svg>"
        ),
        "nested_opacity_clip": (
            '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
            'viewBox="0 0 96 64"><defs><clipPath id="c">'
            '<circle cx="48" cy="32" r="24"/></clipPath></defs>'
            '<g opacity=".78" clip-path="url(#c)"><g opacity=".62">'
            '<rect x="18" y="8" width="60" height="48" fill="#336699"/>'
            '<circle cx="58" cy="28" r="18" fill="#ef7c00" opacity=".64"/>'
            "</g></g></svg>"
        ),
    }


class Issue138AlphaTransparencyTests(unittest.TestCase):
    def test_canonical_input_hashes_and_existing_budget_contract_are_fixed(self) -> None:
        candidate = canonical_candidate_svg_bytes()
        source = canonical_source_rgba()
        self.assertEqual(len(candidate), CANONICAL_PARENT_BYTES)
        self.assertEqual(hashlib.sha256(candidate).hexdigest(), CANONICAL_CANDIDATE_SHA256)
        self.assertEqual(
            hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
            CANONICAL_SOURCE_RGBA_SHA256,
        )
        self.assertEqual(_JOURNAL_BYTE_GROWTH_FACTOR, 3)
        self.assertEqual(_JOURNAL_BYTE_GROWTH_ABSOLUTE, 250_000)
        self.assertEqual(_JOURNAL_PATH_GROWTH_FACTOR, 4)
        self.assertEqual(_JOURNAL_PATH_GROWTH_ABSOLUTE, 500)
        self.assertEqual(_JOURNAL_NODE_GROWTH_FACTOR, 4)
        self.assertEqual(_JOURNAL_NODE_GROWTH_ABSOLUTE, 2_500)

    def test_real_knockout_budget_rejection_then_p0a_compact_fallback_same_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, candidate_path = _write_canonical(root)
            original = candidate_path.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                r"^source_alpha_candidate_knockout_byte_budget_rejected:",
            ) as baseline:
                apply_candidate_geometry_knockout(
                    candidate_path, source_path, "logo_color"
                )
            baseline_error = str(baseline.exception)
            self.assertEqual(candidate_path.read_bytes(), original)
            sizes = baseline_error.rsplit(":", 1)[1]
            projected, limit = (int(value) for value in sizes.split(">"))
            self.assertGreater(projected, limit)
            self.assertEqual(limit, CANONICAL_BYTE_LIMIT)

            fallback = make_candidate_support_reconstruction_fallback(
                apply_candidate_geometry_knockout
            )
            report = fallback(candidate_path, source_path, "logo_color")
            artifact = candidate_path.read_bytes()
            lower = artifact.lower()

            self.assertEqual(
                report["mask_fallback_reason"],
                "candidate_knockout_byte_budget_rejected",
            )
            self.assertEqual(report["mask_fallback_trigger"], baseline_error)
            self.assertEqual(report["before_byte_size"], CANONICAL_PARENT_BYTES)
            self.assertLessEqual(report["after_byte_size"], CANONICAL_BYTE_LIMIT)
            self.assertEqual(report["preflight_byte_limit"], CANONICAL_BYTE_LIMIT)
            self.assertEqual(
                report["preserved_path_count"], report["preflight_parent_path_count"]
            )
            self.assertEqual(
                report["preserved_node_count"], report["preflight_parent_node_count"]
            )
            self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.999)
            self.assertLessEqual(report["source_truth_alpha_mae"], 0.5 / 255.0)
            self.assertLessEqual(report["visible_composite_residual"], 0.01)
            self.assertLessEqual(report["visible_composite_de00_p95"], 1.0)
            if report["alpha_boundary_p95_px"] is not None:
                self.assertLessEqual(report["alpha_boundary_p95_px"], 0.75)
            self.assertNotIn(b"<image", lower)
            self.assertNotIn(b"data:image", lower)
            self.assertNotIn(b"base64", lower)
            print(
                "ISSUE138_CANONICAL_EVIDENCE="
                + json.dumps(
                    {
                        "historical_fixture": False,
                        "canonical_candidate_sha256": CANONICAL_CANDIDATE_SHA256,
                        "canonical_source_rgba_sha256": CANONICAL_SOURCE_RGBA_SHA256,
                        "baseline_error": baseline_error,
                        "before_bytes": int(report["before_byte_size"]),
                        "after_bytes": int(report["after_byte_size"]),
                        "byte_limit": int(report["preflight_byte_limit"]),
                        "path_count": int(report["preserved_path_count"]),
                        "node_count": int(report["preserved_node_count"]),
                        "alpha_iou": float(report["source_truth_alpha_iou"]),
                        "alpha_mae": float(report["source_truth_alpha_mae"]),
                        "alpha_max": float(report["source_truth_alpha_max"]),
                        "visible_composite_residual": float(
                            report["visible_composite_residual"]
                        ),
                        "visible_composite_de00_p95": float(
                            report["visible_composite_de00_p95"]
                        ),
                        "alpha_boundary_p95_px": report["alpha_boundary_p95_px"],
                        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                        "raster_embed_count": 0,
                    },
                    sort_keys=True,
                )
            )

    def test_compact_fallback_is_byte_deterministic_across_two_runs(self) -> None:
        outputs: list[tuple[str, str, str]] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_path, candidate_path = _write_canonical(root)
                fallback = make_candidate_support_reconstruction_fallback(
                    apply_candidate_geometry_knockout
                )
                report = fallback(candidate_path, source_path, "logo_color")
                outputs.append(
                    (
                        hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                        str(report["mask_encoding"]),
                        str(report["mask_fallback_reason"]),
                    )
                )
        self.assertEqual(outputs[0], outputs[1])
        print(
            "ISSUE138_DETERMINISM_EVIDENCE="
            + json.dumps(
                {
                    "repeat_1": outputs[0],
                    "repeat_2": outputs[1],
                    "same_artifact_sha256": outputs[0][0] == outputs[1][0],
                    "same_winner": outputs[0][1:] == outputs[1][1:],
                },
                sort_keys=True,
            )
        )

    def test_soft_overlap_and_nested_clip_composite_metrics_are_gated(self) -> None:
        for name, svg in _identity_svg_cases().items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{name}.svg"
                path.write_text(svg, encoding="utf-8")
                source = render_svg_to_rgba(path, 96, 64)
                self.assertIsNotNone(source)
                assert source is not None
                parent_counts = _path_node_counts(ET.fromstring(path.read_bytes()))
                report = validate_alpha_reconstruction_contract(
                    path, source, "logo_color", parent_counts
                )
                self.assertGreaterEqual(report["source_truth_alpha_iou"], 0.999)
                self.assertLessEqual(report["source_truth_alpha_mae"], 0.5 / 255.0)
                self.assertLessEqual(report["visible_composite_residual"], 0.01)
                self.assertEqual(report["visible_composite_gate_status"], "passed")
                self.assertLessEqual(report["visible_composite_de00_p95"], 1.0)
                if report["alpha_boundary_p95_px"] is not None:
                    self.assertLessEqual(report["alpha_boundary_p95_px"], 0.75)


if __name__ == "__main__":
    unittest.main()
