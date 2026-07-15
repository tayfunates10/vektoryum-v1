import copy
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from engine.regression.rfv2_live_bundle import BundleError, build_bundle, canonical_bytes

CATEGORIES = [
    "flat_logo",
    "badge_seal",
    "small_text",
    "monoline",
    "multicolor",
    "low_resolution_signage_photo",
    "gradient_artwork",
    "native_4k",
    "transparent_dark_background",
    "complex_illustration",
]


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_fixture(root):
    storage = root / "storage"
    downloads = root / "downloads"
    records = root / "records"
    out = root / "out"
    for directory in (storage, downloads, records, out):
        directory.mkdir(parents=True, exist_ok=True)

    cases = []
    for index in range(24):
        case_id = f"qualification-public-{index + 1:02d}"
        category = CATEGORIES[index % len(CATEGORIES)]
        payload = f"public-asset-{index}".encode("utf-8")
        source_sha256 = sha256_bytes(payload)
        consent_sha256 = f"{index + 1001:064x}"
        inspection_sha256 = f"{index + 2001:064x}"
        object_id = f"rfv/qualification/{source_sha256[:2]}/{source_sha256}.png"
        object_path = storage / object_id
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(payload)

        case = {
            "case_id": case_id,
            "category": category,
            "split": "qualification",
            "source_sha256": source_sha256,
            "consent_sha256": consent_sha256,
            "inspection_sha256": inspection_sha256,
            "license": "cc0",
            "source_format": "png",
            "width": 16,
            "height": 16,
            "file_bytes": len(payload),
            "storage_object_id": object_id,
            "privacy_review": "approved",
            "contains_public_pii": False,
            "source_verified": True,
            "consent_verified": True,
            "object_immutable": True,
            "decode_verified": True,
        }
        cases.append(case)
        record = {"schema": "vektoryum-rfv2-qualified-case-v1", **case, "inspection": {}}
        write_json(records / f"{case_id}.json", record)
        proof = {
            "schema": "vektoryum-rfv2-public-license-proof-v1",
            "case_id": case_id,
            "canonical_source_sha256": source_sha256,
            "source_page_sha256": f"{index + 3001:064x}",
            "license_proof_sha256": f"{index + 4001:064x}",
            "downloaded_asset_sha256": f"{index + 5001:064x}",
        }
        write_json(downloads / case_id / "license-proof.json", proof)

    cases_sha256 = sha256_bytes(canonical_bytes(cases))
    manifest = {
        "schema": "vektoryum-rfv2-qualification-manifest-v1",
        "status": "qualified",
        "expected_case_count": 24,
        "qualified_case_count": 24,
        "public_repo_contains_raw_assets": False,
        "cases_sha256": cases_sha256,
        "cases": cases,
    }
    audit = {
        "schema": "vektoryum-rfv2-assembly-audit-v1",
        "complete": True,
        "required_case_count": 24,
        "qualified_case_count": 24,
        "category_counts": {category: sum(case["category"] == category for case in cases) for category in CATEGORIES},
        "missing_categories": [],
        "duplicate_case_ids": 0,
        "duplicate_source_digests": 0,
        "duplicate_storage_objects": 0,
        "duplicate_inspection_digests": 0,
        "cases_sha256": cases_sha256,
    }
    manifest_path = root / "qualification-manifest.json"
    audit_path = root / "qualification-audit.json"
    write_json(manifest_path, manifest)
    write_json(audit_path, audit)
    return storage, downloads, records, out, manifest_path, audit_path


class RFV2LiveBundleTests(unittest.TestCase):
    def test_builds_deterministic_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage, downloads, records, out, manifest, audit = build_fixture(root)
            first_bundle = out / "first.tar.gz"
            first_checksums = out / "first.json"
            second_bundle = out / "second.tar.gz"
            second_checksums = out / "second.json"
            first = build_bundle(
                storage_root=storage,
                download_root=downloads,
                records_dir=records,
                manifest_path=manifest,
                audit_path=audit,
                bundle_out=first_bundle,
                checksums_out=first_checksums,
            )
            second = build_bundle(
                storage_root=storage,
                download_root=downloads,
                records_dir=records,
                manifest_path=manifest,
                audit_path=audit,
                bundle_out=second_bundle,
                checksums_out=second_checksums,
            )
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertEqual(first_bundle.read_bytes(), second_bundle.read_bytes())
            self.assertEqual(first["qualified_case_count"], 24)
            self.assertEqual(first["storage_mode"], "github_actions_immutable_artifact")
            with tarfile.open(first_bundle, "r:gz") as archive:
                names = archive.getnames()
            self.assertIn("bundle-index.json", names)
            self.assertIn("qualification-manifest.json", names)
            self.assertIn("qualification-audit.json", names)
            self.assertIn("source-selection-manifest.json", names)
            self.assertEqual(sum(name.startswith("objects/") for name in names), 24)
            self.assertEqual(sum(name.startswith("proofs/") for name in names), 24)
            self.assertEqual(sum(name.startswith("records/") for name in names), 24)

    def test_missing_or_tampered_object_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage, downloads, records, out, manifest_path, audit = build_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            object_path = storage / manifest["cases"][0]["storage_object_id"]
            object_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(BundleError, "stored object digest mismatch"):
                build_bundle(
                    storage_root=storage,
                    download_root=downloads,
                    records_dir=records,
                    manifest_path=manifest_path,
                    audit_path=audit,
                    bundle_out=out / "bundle.tar.gz",
                    checksums_out=out / "checksums.json",
                )

    def test_incomplete_manifest_and_audit_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage, downloads, records, out, manifest_path, audit_path = build_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "collecting"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(BundleError, "not qualified"):
                build_bundle(
                    storage_root=storage,
                    download_root=downloads,
                    records_dir=records,
                    manifest_path=manifest_path,
                    audit_path=audit_path,
                    bundle_out=out / "bundle.tar.gz",
                    checksums_out=out / "checksums.json",
                )

            manifest["status"] = "qualified"
            write_json(manifest_path, manifest)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["duplicate_source_digests"] = 1
            write_json(audit_path, audit)
            with self.assertRaises(BundleError):
                build_bundle(
                    storage_root=storage,
                    download_root=downloads,
                    records_dir=records,
                    manifest_path=manifest_path,
                    audit_path=audit_path,
                    bundle_out=out / "bundle.tar.gz",
                    checksums_out=out / "checksums.json",
                )

    def test_symlink_proof_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage, downloads, records, out, manifest_path, audit_path = build_fixture(root)
            proof = downloads / "qualification-public-01" / "license-proof.json"
            target = root / "proof-target.json"
            target.write_bytes(proof.read_bytes())
            proof.unlink()
            proof.symlink_to(target)
            with self.assertRaisesRegex(BundleError, "license proof"):
                build_bundle(
                    storage_root=storage,
                    download_root=downloads,
                    records_dir=records,
                    manifest_path=manifest_path,
                    audit_path=audit_path,
                    bundle_out=out / "bundle.tar.gz",
                    checksums_out=out / "checksums.json",
                )


if __name__ == "__main__":
    unittest.main()
