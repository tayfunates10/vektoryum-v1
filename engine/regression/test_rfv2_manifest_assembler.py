import copy
import json
import tempfile
import unittest
from pathlib import Path

from engine.regression.rfv2_manifest_assembler import (
    AssemblyError,
    ROOT,
    assemble_manifest,
    canonical_sha256,
    load_record_directory,
    require_external_output,
)

INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_record(index, category):
    source_sha256 = f"{index + 1:064x}"
    consent_sha256 = f"{index + 1001:064x}"
    inspection = {
        "schema": "vektoryum-rfv2-inspection-v1",
        "case_id": f"qualification-real-{index:02d}",
        "category": category,
        "source_sha256": source_sha256,
        "consent_sha256": consent_sha256,
        "source_format": "png",
        "width": 1024,
        "height": 1024,
        "file_bytes": 4096 + index,
    }
    return {
        "schema": "vektoryum-rfv2-qualified-case-v1",
        "case_id": inspection["case_id"],
        "category": category,
        "split": "qualification",
        "source_sha256": source_sha256,
        "consent_sha256": consent_sha256,
        "inspection_sha256": canonical_sha256(inspection),
        "license": "owned-original",
        "source_format": inspection["source_format"],
        "width": inspection["width"],
        "height": inspection["height"],
        "file_bytes": inspection["file_bytes"],
        "storage_object_id": f"rfv/qualification/{source_sha256[:2]}/{source_sha256}.png",
        "privacy_review": "approved",
        "contains_public_pii": False,
        "source_verified": True,
        "consent_verified": True,
        "object_immutable": True,
        "decode_verified": True,
        "inspection": inspection,
    }


def make_complete_records():
    categories = list(load_json(INTAKE_POLICY_PATH)["categories"])
    return [make_record(index, categories[index % len(categories)]) for index in range(24)]


class RFV2ManifestAssemblerTests(unittest.TestCase):
    def test_complete_manifest_is_qualified_and_deterministic(self):
        records = make_complete_records()
        manifest_a, audit_a = assemble_manifest(records, require_complete=True)
        manifest_b, audit_b = assemble_manifest(list(reversed(records)), require_complete=True)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(manifest_a["status"], "qualified")
        self.assertEqual(manifest_a["qualified_case_count"], 24)
        self.assertTrue(audit_a["complete"])
        self.assertEqual(audit_a["missing_categories"], [])
        self.assertEqual(manifest_a["cases_sha256"], canonical_sha256(manifest_a["cases"]))
        self.assertFalse(manifest_a["public_repo_contains_raw_assets"])
        self.assertNotIn("inspection", manifest_a["cases"][0])
        self.assertNotIn("schema", manifest_a["cases"][0])

    def test_partial_collection_never_claims_qualification(self):
        records = make_complete_records()[:5]
        manifest, audit = assemble_manifest(records, require_complete=False)
        self.assertEqual(manifest["status"], "collecting")
        self.assertEqual(manifest["qualified_case_count"], 5)
        self.assertFalse(audit["complete"])
        with self.assertRaises(AssemblyError):
            assemble_manifest(records, require_complete=True)

    def test_missing_category_fails_complete_gate(self):
        records = [make_record(index, "flat_logo") for index in range(24)]
        with self.assertRaises(AssemblyError):
            assemble_manifest(records, require_complete=True)
        manifest, audit = assemble_manifest(records, require_complete=False)
        self.assertEqual(manifest["status"], "collecting")
        self.assertIn("badge_seal", audit["missing_categories"])

    def test_duplicate_and_tampered_evidence_fails_closed(self):
        records = make_complete_records()
        duplicate = copy.deepcopy(records)
        duplicate[1]["source_sha256"] = duplicate[0]["source_sha256"]
        duplicate[1]["inspection"]["source_sha256"] = duplicate[0]["source_sha256"]
        duplicate[1]["inspection_sha256"] = canonical_sha256(duplicate[1]["inspection"])
        with self.assertRaisesRegex(AssemblyError, "duplicate source_sha256"):
            assemble_manifest(duplicate, require_complete=True)

        tampered = copy.deepcopy(records)
        tampered[0]["inspection"]["width"] = 2048
        with self.assertRaises(AssemblyError):
            assemble_manifest(tampered, require_complete=True)

        leaked = copy.deepcopy(records)
        leaked[0]["source_path"] = "/private/customer/logo.png"
        with self.assertRaises(AssemblyError):
            assemble_manifest(leaked, require_complete=True)

        unreviewed = copy.deepcopy(records)
        unreviewed[0]["license"] = "copyright-unverified"
        with self.assertRaises(AssemblyError):
            assemble_manifest(unreviewed, require_complete=True)

    def test_external_directory_loader_and_output_boundary(self):
        records = make_complete_records()[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_dir = root / "records"
            records_dir.mkdir()
            for index, record in enumerate(records):
                (records_dir / f"record-{index}.json").write_text(json.dumps(record), encoding="utf-8")
            loaded = load_record_directory(records_dir)
            self.assertEqual(len(loaded), 2)
            output = require_external_output(root / "out" / "manifest.json", "manifest output")
            self.assertFalse(output.exists())

        with self.assertRaises(AssemblyError):
            require_external_output(ROOT / "engine" / "regression" / "forbidden.json", "manifest output")

    def test_roadmap_remains_honest_while_rfv2_is_blocked(self):
        roadmap = load_json(ROADMAP_PATH)
        phases = roadmap["phases"]
        self.assertEqual([phase["id"] for phase in phases], ["RFV-1", "RFV-2", "RFV-3", "RFV-4"])
        self.assertEqual(phases[0]["status"], "merged")
        self.assertEqual(phases[1]["status"], "pending")
        self.assertEqual(phases[2]["status"], "pending")
        self.assertEqual(phases[3]["status"], "pending")
        self.assertTrue((ROOT / phases[1]["assembly_evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()
