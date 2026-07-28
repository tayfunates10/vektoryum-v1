from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from engine.regression.rfv3_corpus_identity import (
    CorpusIdentityError,
    canonical_sha256,
    stable_cases_sha256,
    validate_reviewed_corpus_identity,
)


class StableCorpusIdentityTests(unittest.TestCase):
    def _case(self, case_id: str, source_sha256: str) -> dict[str, object]:
        return {
            "case_id": case_id,
            "category": "flat_logo",
            "source_sha256": source_sha256,
            "consent_sha256": "1" * 64,
            "inspection_sha256": "2" * 64,
            "license": "cc0",
            "source_format": "png",
            "width": 64,
            "height": 64,
            "file_bytes": 128,
            "storage_object_id": f"rfv/qualification/{case_id}/source.png",
            "privacy_review": "approved",
            "contains_public_pii": False,
            "decode_verified": True,
            "consent_verified": True,
            "object_immutable": True,
            "split": "qualification",
        }

    def _manifest(self) -> dict[str, object]:
        cases = [
            self._case("qualification-public-02", "b" * 64),
            self._case("qualification-public-01", "a" * 64),
        ]
        return {
            "schema": "vektoryum-rfv2-qualification-manifest-v1",
            "status": "qualified",
            "expected_case_count": 2,
            "qualified_case_count": 2,
            "public_repo_contains_raw_assets": False,
            "cases_sha256": canonical_sha256(cases),
            "cases": cases,
        }

    def test_volatile_proof_digests_do_not_change_stable_identity(self) -> None:
        first = self._manifest()
        second = copy.deepcopy(first)
        for case in second["cases"]:
            case["consent_sha256"] = "3" * 64
            case["inspection_sha256"] = "4" * 64
        self.assertEqual(
            stable_cases_sha256(first, expected_case_count=2),
            stable_cases_sha256(second, expected_case_count=2),
        )

    def test_source_digest_change_changes_stable_identity(self) -> None:
        first = self._manifest()
        second = copy.deepcopy(first)
        second["cases"][0]["source_sha256"] = "c" * 64
        self.assertNotEqual(
            stable_cases_sha256(first, expected_case_count=2),
            stable_cases_sha256(second, expected_case_count=2),
        )

    def test_case_order_does_not_change_stable_identity(self) -> None:
        first = self._manifest()
        second = copy.deepcopy(first)
        second["cases"].reverse()
        self.assertEqual(
            stable_cases_sha256(first, expected_case_count=2),
            stable_cases_sha256(second, expected_case_count=2),
        )

    def test_duplicate_case_id_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["cases"][1]["case_id"] = manifest["cases"][0]["case_id"]
        with self.assertRaisesRegex(CorpusIdentityError, "duplicate"):
            stable_cases_sha256(manifest, expected_case_count=2)

    def test_missing_stable_field_is_rejected(self) -> None:
        manifest = self._manifest()
        del manifest["cases"][0]["source_sha256"]
        with self.assertRaisesRegex(CorpusIdentityError, "fields missing"):
            stable_cases_sha256(manifest, expected_case_count=2)

    def test_bundle_and_manifest_validation_is_fail_closed(self) -> None:
        manifest = self._manifest()
        cases_sha256 = manifest["cases_sha256"]
        audit = {
            "schema": "vektoryum-rfv2-assembly-audit-v1",
            "complete": True,
            "qualified_case_count": 2,
            "missing_categories": [],
            "cases_sha256": cases_sha256,
        }
        bundle_bytes = b"stable-test-bundle"
        checksums = {
            "schema": "vektoryum-rfv2-live-bundle-checksums-v1",
            "qualified_case_count": 2,
            "storage_mode": "github_actions_immutable_artifact",
            "retention_days": 90,
            "cases_sha256": cases_sha256,
            "bundle_bytes": len(bundle_bytes),
            "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        }
        expected = stable_cases_sha256(manifest, expected_case_count=2)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "corpus.tar.gz"
            bundle.write_bytes(bundle_bytes)
            self.assertEqual(
                validate_reviewed_corpus_identity(
                    manifest=manifest,
                    audit=audit,
                    checksums=checksums,
                    bundle=bundle,
                    expected_case_count=2,
                    expected_stable_digest=expected,
                ),
                expected,
            )
            checksums["bundle_sha256"] = "0" * 64
            with self.assertRaisesRegex(CorpusIdentityError, "bundle digest"):
                validate_reviewed_corpus_identity(
                    manifest=manifest,
                    audit=audit,
                    checksums=checksums,
                    bundle=bundle,
                    expected_case_count=2,
                    expected_stable_digest=expected,
                )


if __name__ == "__main__":
    unittest.main()
