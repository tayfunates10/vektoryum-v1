from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from engine.regression.residual_multiscale_source import (
    _resize_canonical_rgba,
    measure_multiscale_svg_from_base,
)


class CanonicalSourceMultiscaleTests(unittest.TestCase):
    @staticmethod
    def _base() -> np.ndarray:
        rgba = np.full((64, 64, 4), 255, dtype=np.uint8)
        rgba[8:56, 12:52, :3] = 0
        rgba[24:40, 24:40, :3] = (220, 35, 45)
        return rgba

    def test_builder_is_called_once_at_base_size(self) -> None:
        calls: list[int] = []
        canonical = self._base()

        def builder(size: int) -> Image.Image:
            calls.append(size)
            self.assertEqual(size, 64)
            return Image.fromarray(canonical, mode="RGBA")

        def render(_path, width: int, height: int):
            self.assertEqual(width, height)
            return _resize_canonical_rgba(canonical, width)

        with patch(
            "engine.regression.residual_multiscale_source.render_svg_to_rgba",
            side_effect=render,
        ):
            result = measure_multiscale_svg_from_base(
                "unused.svg",
                builder,
                base_size=64,
                scales=(0.5, 1.0, 2.0),
            )

        self.assertEqual(calls, [64])
        self.assertTrue(result["all_scales_measured"])
        self.assertEqual(result["min_source_component_recall"], 1.0)
        self.assertEqual(result["min_small_component_recall"], 1.0)
        self.assertEqual(result["max_normalized_boundary_excess_px"], 0.0)
        self.assertEqual(result["source_scale_policy"], "resize_canonical_base_raster_v1")

    def test_downscale_uses_area_and_upscale_uses_lanczos_shape_contract(self) -> None:
        canonical = self._base()
        down = _resize_canonical_rgba(canonical, 32)
        up = _resize_canonical_rgba(canonical, 128)
        self.assertEqual(down.shape, (32, 32, 4))
        self.assertEqual(up.shape, (128, 128, 4))
        self.assertEqual(down.dtype, np.uint8)
        self.assertEqual(up.dtype, np.uint8)

        expected_down = cv2.resize(canonical, (32, 32), interpolation=cv2.INTER_AREA)
        expected_up = cv2.resize(canonical, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        self.assertTrue(np.array_equal(down, expected_down))
        self.assertTrue(np.array_equal(up, expected_up))

    def test_missing_render_remains_fail_closed(self) -> None:
        canonical = self._base()

        def builder(_size: int) -> Image.Image:
            return Image.fromarray(canonical, mode="RGBA")

        def render(_path, width: int, _height: int):
            if width == 32:
                return None
            return _resize_canonical_rgba(canonical, width)

        with patch(
            "engine.regression.residual_multiscale_source.render_svg_to_rgba",
            side_effect=render,
        ):
            result = measure_multiscale_svg_from_base(
                "unused.svg",
                builder,
                base_size=64,
                scales=(0.5, 1.0),
            )

        self.assertFalse(result["all_scales_measured"])
        self.assertEqual(result["missing_scales"], ["0.5x"])


if __name__ == "__main__":
    unittest.main()
