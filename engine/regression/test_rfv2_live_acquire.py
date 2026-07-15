import io
import unittest

from PIL import Image

from engine.regression.rfv2_live_acquire import (
    extract_openclipart_candidates,
    prepare_live_provider_case,
    resolve_openclipart_asset_url,
)
from engine.regression.rfv2_public_source_acquire import PublicSourceError


def png_bytes(*, transparent=False):
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 0 if transparent else 255))
    if transparent:
        image.putpixel((0, 0), (10, 20, 30, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class RFV2LiveAcquireTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "allowed_source_hosts": ["openclipart.org", "www.loc.gov", "tile.loc.gov", "cdn.loc.gov"]
        }
        self.case = {
            "case_id": "qualification-public-01",
            "provider": "openclipart",
            "provider_asset_id": "345253",
            "source_page_url": "https://openclipart.org/detail/345253",
            "asset_url": "https://openclipart.org/image/2000px/345253",
            "acquisition_profile": "openclipart_png",
        }

    def test_extracts_current_png_from_reviewed_source_page(self):
        page = b'''<html><head>
        <meta property="og:image" content="https://openclipart.org/image/800px/svg_to_png/345253/orange-square.png">
        </head><body>
        <a href="https://evil.example/image/345253.png">bad</a>
        <a href="/detail/345253">detail</a>
        </body></html>'''
        candidates = extract_openclipart_candidates(
            page,
            source_page_url=self.case["source_page_url"],
            provider_asset_id="345253",
            allowed_hosts={"openclipart.org"},
        )
        self.assertEqual(
            candidates,
            ["https://openclipart.org/image/800px/svg_to_png/345253/orange-square.png"],
        )

    def test_resolves_first_decodable_allowlisted_candidate(self):
        current = "https://openclipart.org/image/800px/svg_to_png/345253/orange-square.png"
        page = f'<meta property="og:image" content="{current}">'.encode()
        calls = []

        def fetcher(url, allowed_hosts, max_bytes):
            calls.append(url)
            self.assertEqual(allowed_hosts, {"openclipart.org", "www.loc.gov", "tile.loc.gov", "cdn.loc.gov"})
            if url == self.case["source_page_url"]:
                return page, url, "text/html"
            if url == current:
                return png_bytes(), url, "image/png"
            raise PublicSourceError("simulated stale URL")

        resolved = resolve_openclipart_asset_url(self.case, self.manifest, fetcher=fetcher)
        self.assertEqual(resolved, current)
        self.assertEqual(calls[:2], [self.case["source_page_url"], current])

    def test_transparent_profile_requires_actual_alpha(self):
        case = dict(self.case, acquisition_profile="openclipart_transparent_png")
        opaque = "https://openclipart.org/image/800px/svg_to_png/345253/opaque.png"
        transparent = "https://openclipart.org/image/400px/svg_to_png/345253/transparent.png"
        page = f'<img src="{opaque}"><img data-src="{transparent}">'.encode()

        def fetcher(url, allowed_hosts, max_bytes):
            if url == case["source_page_url"]:
                return page, url, "text/html"
            if url == opaque:
                return png_bytes(transparent=False), url, "image/png"
            if url == transparent:
                return png_bytes(transparent=True), url, "image/png"
            raise PublicSourceError("stale")

        resolved = resolve_openclipart_asset_url(case, self.manifest, fetcher=fetcher)
        self.assertEqual(resolved, transparent)

    def test_fails_closed_when_page_has_no_decodable_allowlisted_asset(self):
        page = b'<img src="https://evil.example/image/345253.png">'

        def fetcher(url, allowed_hosts, max_bytes):
            if url == self.case["source_page_url"]:
                return page, url, "text/html"
            raise PublicSourceError("HTTP 404")

        with self.assertRaisesRegex(PublicSourceError, "asset resolution failed"):
            resolve_openclipart_asset_url(self.case, self.manifest, fetcher=fetcher)

    def test_loc_uses_official_json_metadata_as_machine_readable_proof(self):
        original = {
            "case_id": "qualification-public-20",
            "provider": "library_of_congress",
            "source_page_url": "https://www.loc.gov/item/2016812028/",
            "license_proof_url": "https://www.loc.gov/item/2016812028/",
            "metadata_url": "https://www.loc.gov/item/2016812028/?fo=json",
            "rights_statement": "No known restrictions on publication.",
        }
        prepared = prepare_live_provider_case(original, self.manifest)
        self.assertEqual(prepared["source_page_url"], original["metadata_url"])
        self.assertEqual(prepared["license_proof_url"], original["metadata_url"])
        self.assertEqual(original["source_page_url"], "https://www.loc.gov/item/2016812028/")

    def test_loc_metadata_adapter_rejects_host_query_or_rights_drift(self):
        base = {
            "case_id": "qualification-public-20",
            "provider": "library_of_congress",
            "metadata_url": "https://www.loc.gov/item/2016812028/?fo=json",
            "rights_statement": "No known restrictions on publication.",
        }
        for patch in (
            {"metadata_url": "https://openclipart.org/item/2016812028/?fo=json"},
            {"metadata_url": "https://www.loc.gov/item/2016812028/"},
            {"rights_statement": "unknown"},
        ):
            with self.subTest(patch=patch), self.assertRaises(PublicSourceError):
                prepare_live_provider_case({**base, **patch}, self.manifest)


if __name__ == "__main__":
    unittest.main()
