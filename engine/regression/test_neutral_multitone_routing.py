from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from app.analyzer import analyze_image_from_mem, meaningful_neutral_midtones


class NeutralMultitoneRoutingTests(unittest.TestCase):
    @staticmethod
    def _gray_border_logo(size: int = 256) -> Image.Image:
        image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((24, 36, 232, 220), radius=38, fill=(150, 150, 150, 255))
        draw.rounded_rectangle((38, 50, 218, 206), radius=30, fill=(15, 15, 15, 255))
        draw.ellipse((86, 78, 170, 174), fill=(255, 255, 255, 255))
        draw.rectangle((122, 168, 144, 220), fill=(255, 255, 255, 255))
        return image

    @staticmethod
    def _binary_silhouette(size: int = 256) -> Image.Image:
        image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((36, 36, 220, 220), fill=(0, 0, 0, 255))
        draw.ellipse((88, 88, 168, 168), fill=(255, 255, 255, 255))
        return image

    @staticmethod
    def _antialiased_binary_circle(size: int = 256) -> Image.Image:
        scale = 4
        large = Image.new("RGBA", (size * scale, size * scale), (255, 255, 255, 255))
        draw = ImageDraw.Draw(large)
        draw.ellipse((32 * scale, 32 * scale, 224 * scale, 224 * scale), fill=(0, 0, 0, 255))
        return large.resize((size, size), Image.Resampling.LANCZOS)

    def test_real_gray_band_routes_away_from_binary_modes(self) -> None:
        image = self._gray_border_logo()
        report = analyze_image_from_mem(image)

        self.assertTrue(report["has_meaningful_neutral_midtone"])
        self.assertGreaterEqual(len(report["meaningful_neutral_midtones"]), 1)
        self.assertEqual(report["recommended_mode"], "geometric_logo")
        self.assertEqual(report["detected_type"], "geometric_logo")
        self.assertTrue(report["neutral_multitone_routing_guard"]["applied"])
        self.assertIn(
            report["neutral_multitone_routing_guard"]["previous_mode"],
            {"single_color", "lineart"},
        )

    def test_true_binary_silhouette_keeps_binary_routing(self) -> None:
        report = analyze_image_from_mem(self._binary_silhouette())

        self.assertFalse(report["has_meaningful_neutral_midtone"])
        self.assertFalse(report["neutral_multitone_routing_guard"]["applied"])
        self.assertIn(report["recommended_mode"], {"single_color", "lineart"})

    def test_antialias_film_is_not_treated_as_real_gray_layer(self) -> None:
        image = self._antialiased_binary_circle()
        tones = meaningful_neutral_midtones(image)
        report = analyze_image_from_mem(image)

        self.assertEqual(tones, [])
        self.assertFalse(report["neutral_multitone_routing_guard"]["applied"])
        self.assertNotEqual(report["recommended_mode"], "geometric_logo")


if __name__ == "__main__":
    unittest.main()
