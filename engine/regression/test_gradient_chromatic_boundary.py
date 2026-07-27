from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import gradient_vectorize as gradient


class GradientChromaticBoundaryTests(unittest.TestCase):
    @staticmethod
    def _isoluminant_split(size: int = 96) -> np.ndarray:
        # Rec.601 luminance is approximately equal, while chroma is far apart.
        left = np.array([0, 130, 255], dtype=np.uint8)
        right = np.array([255, 50, 0], dtype=np.uint8)
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[:, : size // 2] = left
        rgb[:, size // 2 :] = right
        return rgb

    @staticmethod
    def _smooth_ramp(size: int = 96) -> np.ndarray:
        left = np.array([0, 130, 255], dtype=np.float32)
        right = np.array([255, 50, 0], dtype=np.float32)
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        row = np.clip(left[None, :] * (1.0 - t) + right[None, :] * t, 0, 255).astype(np.uint8)
        return np.repeat(row[None, :, :], size, axis=0)

    def test_isoluminant_hard_boundary_becomes_two_regions(self) -> None:
        rgb = self._isoluminant_split()
        legacy = gradient._legacy_edge_based_segments(rgb)
        chromatic = gradient._edge_based_segments(rgb)

        self.assertEqual(len(np.unique(legacy)), 1)
        self.assertGreaterEqual(len(np.unique(chromatic)), 2)
        self.assertNotEqual(
            int(chromatic[rgb.shape[0] // 2, rgb.shape[1] // 4]),
            int(chromatic[rgb.shape[0] // 2, 3 * rgb.shape[1] // 4]),
        )

    def test_smooth_color_ramp_remains_one_region(self) -> None:
        labels = gradient._edge_based_segments(self._smooth_ramp())
        self.assertEqual(len(np.unique(labels)), 1)

    def test_vectorizer_uses_flat_paths_for_hard_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "hard.png"
            output = root / "hard.svg"
            Image.fromarray(self._isoluminant_split(), mode="RGB").save(source)

            gradient.vectorize_with_gradients(source, output, {"epsilon": 0.3})
            svg = output.read_text(encoding="utf-8")

            self.assertNotIn("<linearGradient", svg)
            self.assertGreaterEqual(svg.count("<path"), 2)
            self.assertIn("#0082ff", svg)
            self.assertIn("#ff3200", svg)

    def test_vectorizer_keeps_linear_gradient_for_smooth_ramp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ramp.png"
            output = root / "ramp.svg"
            Image.fromarray(self._smooth_ramp(), mode="RGB").save(source)

            gradient.vectorize_with_gradients(source, output, {"epsilon": 0.3})
            svg = output.read_text(encoding="utf-8")

            self.assertIn("<linearGradient", svg)
            self.assertEqual(svg.count("<path"), 1)


if __name__ == "__main__":
    unittest.main()
