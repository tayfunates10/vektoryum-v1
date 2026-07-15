import copy
import io
import json
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from engine.regression.rfv2_public_source_acquire import (
    MANIFEST_PATH,
    PublicSourceError,
    canonicalize_image,
    load_json,
    select_loc_asset_url,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"


def encoded_image(mode, size, *, transparent=False, image_format="PNG"):
    image = Image.new(mode, size, (40, 80, 120, 0 if transparent else 255) if "A" in mode else (40, 80, 120))
    if transparent and "A" in mode:
        for x in range(size[0] // 4, size[0] * 3 // 4):
            for y in range(size[1] // 4, size[1] * 3 // 4):
                image.putpixel((x, y), (220, 100, 50, 255))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class RFV2PublicSourceAcquireTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_json(MANIFEST_PATH)

    def test_selected_manifest_is_exact_finite_and_unique(self):
        cases = validate_manifest(self.manifest)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["case_id"] for case in cases}), 24)
        self.assertEqual(len({(case["provider"], case["provider_asset_id"]) for case in cases}), 24)
        self.assertEqual(Counter(case["category"] for case in cases), Counter(self.manifest["category_targets"]))
        self.assertEqual(Counter(case["provider"] for case in cases), Counter({"openclipart": 19, "library_of_congress": 5}))
        self.assertEqual(Counter(case["license"] for case in cases), Counter({"cc0": 19, "public-domain": 5}))

    def test_manifest_rejects_license_host_duplicate_and_progress_fabrication(self):
        mutations = []
        bad_license = copy.deepcopy(self.manifest)
        bad_license["cases"][0]["license"] = "copyright-unverified"
        mutations.append(bad_license)

        bad_host = copy.deepcopy(self.manifest)
        bad_host["cases"][0]["asset_url"] = "https://example.com/unreviewed.png"
        mutations.append(bad_host)

        duplicate = copy.deepcopy(self.manifest)
        duplicate["cases"][1]["provider_asset_id"] = duplicate["cases"][0]["provider_asset_id"]
        mutations.append(duplicate)

        fabricated = copy.deepcopy(self.manifest)
        fabricated["status"] = "acquired"
        mutations.append(fabricated)

        wrong_category = copy.deepcopy(self.manifest)
        wrong_category["cases"][0]["category"] = "complex_illustration"
        mutations.append(wrong_category)

        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(PublicSourceError):
                    validate_manifest(payload)

    def test_loc_asset_resolver_prefers_allowlisted_original_raster(self):
        metadata = {
            "resources": [
                {"files": [
                    {"url": "https://tile.loc.gov/storage-services/service/pnp/test-thumb.jpg"},
                    {"url": "https://tile.loc.gov/storage-services/master/pnp/test-original.tif"},
                    {"url": "https://evil.example/test-original.tif"},
                ]}
            ]
        }
        selected = select_loc_asset_url(metadata, {"www.loc.gov", "tile.loc.gov", "cdn.loc.gov"})
        self.assertEqual(selected, "https://tile.loc.gov/storage-services/master/pnp/test-original.tif")
        with self.assertRaises(PublicSourceError):
            select_loc_asset_url({"resources": [{"url": "https://evil.example/image.tif"}]}, {"tile.loc.gov"})

    def test_graphic_profiles_preserve_png_and_enforce_transparency(self):
        opaque = encoded_image("RGBA", (256, 128), transparent=False)
        result, extension, evidence = canonicalize_image(opaque, "openclipart_png")
        self.assertEqual(extension, "png")
        self.assertEqual(evidence["transform"], "decode_and_lossless_png")
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (256, 128))

        transparent = encoded_image("RGBA", (128, 128), transparent=True)
        result, extension, _ = canonicalize_image(transparent, "openclipart_transparent_png")
        self.assertEqual(extension, "png")
        with Image.open(io.BytesIO(result)) as image:
            self.assertLess(image.getchannel("A").getextrema()[0], 255)

        with self.assertRaises(PublicSourceError):
            canonicalize_image(opaque, "openclipart_transparent_png")

    def test_photo_profiles_are_bounded_and_non_upscaled(self):
        low_source = encoded_image("RGB", (1600, 1200), image_format="JPEG")
        result, extension, evidence = canonicalize_image(low_source, "loc_low_resolution_signage_photo")
        self.assertEqual(extension, "jpeg")
        self.assertLessEqual(max(evidence["width"], evidence["height"]), 640)
        with Image.open(io.BytesIO(result)) as image:
            self.assertLessEqual(image.width * image.height, 640 * 640)

        high_source = encoded_image("RGB", (4200, 2800), image_format="JPEG")
        result, extension, evidence = canonicalize_image(high_source, "loc_public_domain_4k_crop")
        self.assertEqual(extension, "jpeg")
        self.assertEqual((evidence["width"], evidence["height"]), (3840, 2160))
        self.assertEqual(evidence["transform"], "center_crop_16x9_non_upscaled_4k")
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(image.size, (3840, 2160))

        too_small = encoded_image("RGB", (1920, 1080), image_format="JPEG")
        with self.assertRaises(PublicSourceError):
            canonicalize_image(too_small, "loc_public_domain_4k_crop")

    def test_roadmap_remains_honest_until_assets_are_acquired(self):
        roadmap = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(phases[0]["status"], "merged")
        self.assertEqual(phases[1]["status"], "pending")
        self.assertEqual(phases[2]["status"], "pending")
        self.assertEqual(phases[3]["status"], "pending")
        self.assertEqual(phases[1]["public_source_evidence"], "docs/real_world_fidelity/rfv-2d.md")
        self.assertEqual(phases[1]["selected_public_source_count"], 24)


if __name__ == "__main__":
    unittest.main()
