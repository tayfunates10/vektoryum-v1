import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from engine.regression.rfv2_secure_intake import (
    ROOT,
    IntakeError,
    build_qualification_record,
    load_json,
    sha256_file,
    write_json_atomic,
)
from engine.regression.test_rfv2_qualification_adapter import validate_case, validate_manifest

INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
QUALIFICATION_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_policy.json"
MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"
TOOL_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_secure_intake_policy.json"


class RFV2SecureIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.consent = self.root / "consent.txt"
        Image.new("RGBA", (64, 48), (20, 40, 60, 255)).save(self.source)
        self.consent.write_text("reviewed consent evidence", encoding="utf-8")
        self.storage = self.root / "objects"
        self.record_out = self.root / "records" / "case.json"
        self.intake_policy = load_json(INTAKE_POLICY_PATH)
        self.qualification_policy = load_json(QUALIFICATION_POLICY_PATH)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **overrides):
        values = {
            "source": self.source,
            "consent": self.consent,
            "case_id": "qualification-real-001",
            "category": "flat_logo",
            "license_name": "owned-original",
            "storage_root": self.storage,
            "record_out": self.record_out,
            "privacy_review": "approved",
            "confirm_no_public_pii": True,
            "existing_records": None,
        }
        values.update(overrides)
        return build_qualification_record(**values)

    def test_registers_verified_content_addressed_record(self):
        record = self.build()
        validate_case(record, self.qualification_policy, self.intake_policy)
        self.assertEqual(record["source_sha256"], sha256_file(self.source))
        self.assertEqual(record["split"], "qualification")
        self.assertTrue(record["storage_object_id"].startswith("rfv/qualification/"))
        stored = self.storage / record["storage_object_id"]
        self.assertTrue(stored.is_file())
        self.assertEqual(sha256_file(stored), record["source_sha256"])
        self.assertEqual(stored.stat().st_mode & 0o777, 0o440)

        write_json_atomic(self.record_out, record)
        self.assertEqual(json.loads(self.record_out.read_text(encoding="utf-8"))["case_id"], record["case_id"])

    def test_duplicate_case_source_and_object_fail_closed(self):
        record = self.build()
        existing = self.root / "existing.json"
        existing.write_text(json.dumps({"cases": [record]}), encoding="utf-8")
        with self.assertRaises(IntakeError):
            self.build(case_id="qualification-real-002", existing_records=existing)

    def test_privacy_license_category_and_repository_paths_fail_closed(self):
        bad_calls = [
            {"confirm_no_public_pii": False},
            {"privacy_review": "pending"},
            {"license_name": "unknown"},
            {"category": "unknown"},
            {"storage_root": ROOT / "private-assets"},
            {"record_out": ROOT / "private-record.json"},
        ]
        for overrides in bad_calls:
            with self.subTest(overrides=overrides):
                with self.assertRaises(IntakeError):
                    self.build(**overrides)

    def test_symlink_inputs_and_decode_failure_fail_closed(self):
        source_link = self.root / "source-link.png"
        consent_link = self.root / "consent-link.txt"
        source_link.symlink_to(self.source)
        consent_link.symlink_to(self.consent)
        with self.assertRaises(IntakeError):
            self.build(source=source_link)
        with self.assertRaises(IntakeError):
            self.build(consent=consent_link)

        broken = self.root / "broken.png"
        broken.write_bytes(b"not-an-image")
        with self.assertRaises(IntakeError):
            self.build(source=broken)

    def test_policy_and_repository_evidence_follow_finite_lifecycle(self):
        policy = load_json(TOOL_POLICY_PATH)
        self.assertEqual(policy["schema"], "vektoryum-rfv2-secure-intake-v1")
        self.assertFalse(policy["network_access_required"])
        self.assertFalse(policy["raw_assets_in_repository"])
        self.assertTrue(policy["storage_root_external"])
        self.assertTrue(policy["atomic_copy"])

        manifest = load_json(MANIFEST_PATH)
        validate_manifest(manifest, self.qualification_policy, self.intake_policy)
        phases = load_json(ROADMAP_PATH)["phases"]
        self.assertEqual(phases[0]["status"], "merged")
        self.assertEqual([phase["status"] for phase in phases[2:]], ["pending", "pending"])
        if manifest["status"] == "awaiting_real_assets":
            self.assertEqual(manifest["qualified_case_count"], 0)
            self.assertEqual(manifest["cases"], [])
            self.assertEqual(phases[1]["status"], "pending")
        else:
            validate_manifest(manifest, self.qualification_policy, self.intake_policy, require_complete=True)
            self.assertEqual(manifest["qualified_case_count"], 24)
            self.assertIn(phases[1]["status"], {"implemented", "merged"})
            self.assertTrue((ROOT / phases[1]["evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()
