from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from app import preprocess


class BroadIntentionalNeutralRegionGuardTests(unittest.TestCase):
    @staticmethod
    def _badge_rgba(size: int = 128) -> np.ndarray:
        rgb = np.full((size, size, 3), 255, dtype=np.uint8)
        cv2.rectangle(rgb, (24, 24), (103, 103), (25, 25, 25), thickness=-1)
        cv2.rectangle(rgb, (40, 40), (87, 87), (235, 235, 235), thickness=-1)
        cv2.rectangle(rgb, (58, 32), (69, 95), (190, 25, 35), thickness=-1)
        cv2.rectangle(rgb, (32, 58), (95, 69), (190, 25, 35), thickness=-1)
        return np.dstack([rgb, np.full((size, size), 255, dtype=np.uint8)])

    def test_geometric_preprocess_restores_broad_dark_and_light_neutral_fills(self) -> None:
        source = Image.fromarray(self._badge_rgba(), mode="RGBA").resize(
            (256, 256), Image.Resampling.LANCZOS
        )
        report: dict[str, object] = {
            "steps": [],
            "supersampled": {"from": 128, "scale": 2},
        }

        processed = preprocess.preprocess_geometric_logo(np.asarray(source), report)
        colors, counts = np.unique(processed.reshape(-1, 3), axis=0, return_counts=True)
        palette = {
            tuple(int(value) for value in color): int(count)
            for color, count in zip(colors, counts, strict=True)
        }

        self.assertEqual(len(palette), 4)
        self.assertIn((255, 255, 255), palette)
        self.assertTrue(any(max(color) - min(color) <= 8 and 15 <= sum(color) / 3 <= 40 for color in palette))
        self.assertTrue(any(r >= 175 and r > g + 80 and r > b + 80 for r, g, b in palette))
        light_neutral = [
            (color, count)
            for color, count in palette.items()
            if max(color) - min(color) <= 8 and 205 <= sum(color) / 3 <= 245
        ]
        self.assertEqual(len(light_neutral), 1)
        self.assertGreater(light_neutral[0][1], 4000)
        self.assertIn("restore_broad_intentional_neutral_regions", report["steps"])
        guard = report.get("neutral_region_guard")
        self.assertIsInstance(guard, dict)
        self.assertEqual(guard["accepted_count"], 5)
        tone_classes = [component["tone_class"] for component in guard["components"]]
        self.assertEqual(tone_classes.count("dark"), 1)
        self.assertEqual(tone_classes.count("light"), 4)
        self.assertIs(
            preprocess._base._DISPATCH["geometric_logo"],
            preprocess.preprocess_geometric_logo,
        )

    def test_thin_light_neutral_film_is_not_restored(self) -> None:
        source = np.full((128, 128, 3), 255, dtype=np.uint8)
        cv2.rectangle(source, (24, 24), (103, 103), (235, 235, 235), thickness=1)
        processed = np.full_like(source, 255)

        restored, accepted = preprocess._restore_broad_intentional_neutral_regions(
            source,
            processed,
            scale=1,
        )

        self.assertEqual(accepted, [])
        self.assertTrue(np.array_equal(restored, processed))

    def test_thin_dark_neutral_film_is_not_restored(self) -> None:
        source = np.full((128, 128, 3), 255, dtype=np.uint8)
        cv2.rectangle(source, (24, 24), (103, 103), (25, 25, 25), thickness=1)
        processed = np.full_like(source, 255)

        restored, accepted = preprocess._restore_broad_intentional_neutral_regions(
            source,
            processed,
            scale=1,
        )

        self.assertEqual(accepted, [])
        self.assertTrue(np.array_equal(restored, processed))

    def test_nonwhite_canvas_is_fail_closed(self) -> None:
        source = np.full((128, 128, 3), 230, dtype=np.uint8)
        cv2.rectangle(source, (32, 32), (95, 95), (190, 190, 190), thickness=-1)
        processed = np.full_like(source, 230)

        restored, accepted = preprocess._restore_broad_intentional_neutral_regions(
            source,
            processed,
            scale=1,
        )

        self.assertEqual(accepted, [])
        self.assertTrue(np.array_equal(restored, processed))

    def test_chromatic_fill_is_not_treated_as_neutral(self) -> None:
        source = np.full((128, 128, 3), 255, dtype=np.uint8)
        cv2.rectangle(source, (32, 32), (95, 95), (205, 225, 245), thickness=-1)
        processed = np.full_like(source, 255)

        restored, accepted = preprocess._restore_broad_intentional_neutral_regions(
            source,
            processed,
            scale=1,
        )

        self.assertEqual(accepted, [])
        self.assertTrue(np.array_equal(restored, processed))


if __name__ == "__main__":
    unittest.main()
