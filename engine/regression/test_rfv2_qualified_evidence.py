import hashlib
import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_policy.json"
INTAKE_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
SELECTION_PATH = ROOT / "engine" / "regression" / "rfv2_public_source_manifest.json"
AUDIT_PATH = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv2_qualification_audit.json"
CHECKSUMS_PATH = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv2_bundle_checksums.json"
ENVELOPE_PATH = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv2_publication_envelope.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"
EVIDENCE_DOC = ROOT / "docs" / "real_world_fidelity" / "rfv-2.md"

EXPECTED_CASES_SHA256 = "5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89"
EXPECTED_BUNDLE_SHA256 = "1da641ad27b58985e4e1cf8d9972af5a15f5105e85e2c319151a5e373b6afc46"
EXPECTED_ARTIFACT_DIGEST = "a8be8c0782a8aeb037a2736de8adbd357c5074ad0da6355562e2543092d6af76"
EXPECTED_PRODUCER_HEAD = "d55f812f492e8e93c1956fe79bceaf7e3754d7e9"
EXPECTED_ARTIFACT_ID = 8354853386
EXPECTED_RUN_ID = 29444449427
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
CASE_ID_RE = re.compile(r"^qualification-public-[0-9]{2}$")
OBJECT_RE = re.compile(r"^rfv/qualification/[0-9a-f]{2}/([0-9a-f]{64})\.(png|jpeg|webp|tiff)$")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RFV2QualifiedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_json(MANIFEST_PATH)
        cls.policy = load_json(POLICY_PATH)
        cls.intake = load_json(INTAKE_PATH)
        cls.selection = load_json(SELECTION_PATH)
        cls.audit = load_json(AUDIT_PATH)
        cls.checksums = load_json(CHECKSUMS_PATH)
        cls.envelope = load_json(ENVELOPE_PATH)
        cls.roadmap = load_json(ROADMAP_PATH)

    def test_manifest_is_exact_finite_qualified_corpus(self) -> None:
        manifest = self.manifest
        cases = manifest["cases"]
        self.assertEqual(manifest["schema"], "vektoryum-rfv2-qualification-manifest-v1")
        self.assertEqual(manifest["status"], "qualified")
        self.assertEqual(manifest["expected_case_count"], 24)
        self.assertEqual(manifest["qualified_case_count"], 24)
        self.assertFalse(manifest["public_repo_contains_raw_assets"])
        self.assertEqual(len(cases), 24)
        self.assertEqual(manifest["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(canonical_sha256(cases), EXPECTED_CASES_SHA256)

    def test_each_case_passes_policy_privacy_and_digest_contract(self) -> None:
        required = set(self.policy["required_case_fields"])
        allowed_licenses = set(self.intake["allowed_licenses"])
        allowed_formats = set(self.intake["allowed_source_formats"])
        allowed_categories = set(self.intake["categories"])
        seen = {key: set() for key in ("case_id", "source_sha256", "consent_sha256", "inspection_sha256", "storage_object_id")}

        for case in self.manifest["cases"]:
            self.assertEqual(set(case), required)
            self.assertRegex(case["case_id"], CASE_ID_RE)
            self.assertIn(case["category"], allowed_categories)
            self.assertEqual(case["split"], "qualification")
            self.assertIn(case["license"], allowed_licenses)
            self.assertIn(case["source_format"], allowed_formats)
            for key in ("source_sha256", "consent_sha256", "inspection_sha256"):
                self.assertRegex(case[key], SHA256_RE)
            object_match = OBJECT_RE.fullmatch(case["storage_object_id"])
            self.assertIsNotNone(object_match)
            self.assertEqual(object_match.group(1), case["source_sha256"])
            self.assertEqual(object_match.group(2), case["source_format"])
            self.assertEqual(case["privacy_review"], "approved")
            self.assertIs(case["contains_public_pii"], False)
            for key in self.policy["required_boolean_true_fields"]:
                self.assertIs(case[key], True)
            self.assertIsInstance(case["width"], int)
            self.assertIsInstance(case["height"], int)
            self.assertIsInstance(case["file_bytes"], int)
            self.assertGreater(case["width"], 0)
            self.assertGreater(case["height"], 0)
            self.assertGreater(case["file_bytes"], 0)
            self.assertLessEqual(case["width"] * case["height"], self.intake["max_pixels"])
            self.assertLessEqual(case["file_bytes"], self.intake["max_file_bytes"])
            for key in seen:
                self.assertNotIn(case[key], seen[key], f"duplicate {key}")
                seen[key].add(case[key])
            for value in case.values():
                if isinstance(value, str):
                    self.assertFalse(value.startswith(("http://", "https://", "/")))

    def test_category_and_selected_source_identity_are_exact(self) -> None:
        cases = self.manifest["cases"]
        selected = self.selection["cases"]
        expected_counts = self.selection["category_targets"]
        self.assertEqual(self.selection["schema"], "vektoryum-rfv2-public-source-manifest-v1")
        self.assertEqual(self.selection["expected_case_count"], 24)
        self.assertFalse(self.selection["public_repo_contains_raw_assets"])
        self.assertEqual(len(selected), 24)
        self.assertEqual(Counter(case["category"] for case in cases), Counter(expected_counts))
        self.assertEqual(
            [(case["case_id"], case["category"], case["license"]) for case in cases],
            [(case["case_id"], case["category"], case["license"]) for case in selected],
        )

    def test_audit_is_complete_and_cross_bound(self) -> None:
        audit = self.audit
        self.assertEqual(audit["schema"], "vektoryum-rfv2-assembly-audit-v1")
        self.assertIs(audit["complete"], True)
        self.assertEqual(audit["required_case_count"], 24)
        self.assertEqual(audit["qualified_case_count"], 24)
        self.assertEqual(audit["category_counts"], self.selection["category_targets"])
        self.assertEqual(audit["missing_categories"], [])
        for key in (
            "duplicate_case_ids",
            "duplicate_source_digests",
            "duplicate_storage_objects",
            "duplicate_inspection_digests",
        ):
            self.assertEqual(audit[key], 0)
        self.assertEqual(audit["cases_sha256"], EXPECTED_CASES_SHA256)

    def test_bundle_and_artifact_publication_are_immutable_and_consistent(self) -> None:
        checksums = self.checksums
        envelope = self.envelope
        self.assertEqual(checksums["schema"], "vektoryum-rfv2-live-bundle-checksums-v1")
        self.assertEqual(checksums["bundle_sha256"], EXPECTED_BUNDLE_SHA256)
        self.assertEqual(checksums["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(checksums["qualified_case_count"], 24)
        self.assertGreater(checksums["bundle_bytes"], 0)
        self.assertFalse(checksums["raw_assets_in_repository"])
        self.assertEqual(checksums["storage_mode"], "github_actions_immutable_artifact")
        self.assertEqual(checksums["retention_days"], 90)
        self.assertRegex(checksums["qualification_manifest_sha256"], SHA256_RE)
        self.assertRegex(checksums["qualification_audit_sha256"], SHA256_RE)

        self.assertEqual(envelope["schema"], "vektoryum-rfv2-actions-publication-envelope-v1")
        self.assertRegex(envelope["head_sha"], SHA1_RE)
        self.assertEqual(envelope["head_sha"], EXPECTED_PRODUCER_HEAD)
        self.assertEqual(envelope["artifact_id"], EXPECTED_ARTIFACT_ID)
        self.assertEqual(envelope["artifact_digest"], EXPECTED_ARTIFACT_DIGEST)
        self.assertEqual(envelope["bundle_sha256"], EXPECTED_BUNDLE_SHA256)
        self.assertEqual(envelope["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(envelope["qualified_case_count"], 24)
        self.assertFalse(envelope["raw_assets_in_repository"])
        self.assertEqual(envelope["retention_days"], 90)
        parsed = urlsplit(envelope["artifact_url"])
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "github.com")
        self.assertEqual(
            parsed.path,
            f"/tayfunates10/vektoryum-v1/actions/runs/{EXPECTED_RUN_ID}/artifacts/{EXPECTED_ARTIFACT_ID}",
        )

    def test_roadmap_has_monotonic_terminal_state_and_evidence_paths(self) -> None:
        phases = self.roadmap["phases"]
        self.assertEqual(self.roadmap["schema"], "vektoryum-real-world-fidelity-v1")
        self.assertEqual(self.roadmap["phase_count"], 4)
        self.assertEqual([phase["id"] for phase in phases], ["RFV-1", "RFV-2", "RFV-3", "RFV-4"])
        self.assertEqual(phases[0]["status"], "merged")
        self.assertIn(phases[1]["status"], {"implemented", "merged"})
        self.assertEqual([phase["status"] for phase in phases[2:]], ["pending", "pending"])
        phase = phases[1]
        self.assertEqual(phase["qualified_case_count"], 24)
        for key in (
            "evidence",
            "qualification_manifest",
            "qualification_audit",
            "bundle_checksums",
            "publication_envelope",
            "preparation_evidence",
            "intake_evidence",
            "assembly_evidence",
            "public_source_evidence",
            "live_acquisition_evidence",
        ):
            self.assertTrue((ROOT / phase[key]).is_file(), key)
        self.assertTrue(EVIDENCE_DOC.is_file())

    def test_json_evidence_contains_only_finite_values(self) -> None:
        for payload in (self.manifest, self.audit, self.checksums, self.envelope):
            stack = [payload]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
                elif isinstance(value, float):
                    self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
