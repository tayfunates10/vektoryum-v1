import copy
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_policy.json"
MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_policy(policy, intake_policy):
    if policy.get("schema") != "vektoryum-rfv2-qualification-policy-v1":
        raise AssertionError("invalid RFV-2 qualification schema")
    if policy.get("required_case_count") != intake_policy["splits"]["qualification"]:
        raise AssertionError("qualification quota drift")
    if policy.get("required_split") != "qualification":
        raise AssertionError("qualification split drift")
    if policy.get("storage_mode") != "external_immutable_object_store":
        raise AssertionError("mutable storage is forbidden")
    if policy.get("public_repo_contains_raw_assets") is not False:
        raise AssertionError("raw private assets must not be public")
    if policy.get("minimum_category_coverage") != len(intake_policy["categories"]):
        raise AssertionError("category coverage drift")

    fields = policy.get("required_case_fields")
    true_fields = policy.get("required_boolean_true_fields")
    if not isinstance(fields, list) or len(fields) != len(set(fields)):
        raise AssertionError("invalid required case fields")
    if not isinstance(true_fields, list) or not true_fields or not set(true_fields).issubset(fields):
        raise AssertionError("invalid verification fields")
    failures = policy.get("fail_closed_on")
    if not isinstance(failures, list) or len(failures) != len(set(failures)) or len(failures) < 10:
        raise AssertionError("incomplete fail-closed policy")


def validate_case(record, policy, intake_policy):
    required_fields = set(policy["required_case_fields"])
    if not isinstance(record, dict) or not required_fields.issubset(record):
        raise AssertionError("missing qualification fields")

    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
        raise AssertionError("invalid case_id")
    if record.get("category") not in intake_policy["categories"]:
        raise AssertionError("unknown category")
    if record.get("split") != policy["required_split"]:
        raise AssertionError("wrong corpus split")

    for key in ("source_sha256", "consent_sha256", "inspection_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise AssertionError(f"invalid {key}")

    if record.get("license") not in intake_policy["allowed_licenses"]:
        raise AssertionError("unreviewed license")
    if record.get("source_format") not in intake_policy["allowed_source_formats"]:
        raise AssertionError("unsupported source format")

    width, height = record.get("width"), record.get("height")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (width, height)):
        raise AssertionError("invalid dimensions")
    if width * height > intake_policy["max_pixels"]:
        raise AssertionError("pixel budget exceeded")

    file_bytes = record.get("file_bytes")
    if not isinstance(file_bytes, int) or isinstance(file_bytes, bool) or not 0 < file_bytes <= intake_policy["max_file_bytes"]:
        raise AssertionError("file budget exceeded")

    object_id = record.get("storage_object_id")
    if not isinstance(object_id, str) or not object_id.startswith("rfv/qualification/") or "://" in object_id:
        raise AssertionError("storage identifier must be opaque and qualification-scoped")
    if record.get("privacy_review") != "approved":
        raise AssertionError("privacy review missing")
    if record.get("contains_public_pii") is not False:
        raise AssertionError("public PII is forbidden")

    for field in policy["required_boolean_true_fields"]:
        if record.get(field) is not True:
            raise AssertionError(f"failed verification: {field}")


def validate_qualified_cases(records, policy, intake_policy, require_complete=True):
    validate_policy(policy, intake_policy)
    if not isinstance(records, list) or not records:
        raise AssertionError("empty qualification evidence")
    for record in records:
        validate_case(record, policy, intake_policy)

    for key in ("case_id", "source_sha256", "storage_object_id", "inspection_sha256"):
        values = [record[key] for record in records]
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate {key}")

    if require_complete:
        if len(records) != policy["required_case_count"]:
            raise AssertionError("qualification corpus target not met")
        coverage = Counter(record["category"] for record in records)
        if len(coverage) < policy["minimum_category_coverage"] or any(value < 1 for value in coverage.values()):
            raise AssertionError("missing required category")


def validate_manifest(manifest, policy, intake_policy, require_complete=False):
    if manifest.get("schema") != "vektoryum-rfv2-qualification-manifest-v1":
        raise AssertionError("invalid manifest schema")
    if manifest.get("expected_case_count") != policy["required_case_count"]:
        raise AssertionError("manifest quota drift")
    if manifest.get("public_repo_contains_raw_assets") is not False:
        raise AssertionError("raw assets cannot be published")

    cases = manifest.get("cases")
    status = manifest.get("status")
    if status == "awaiting_real_assets":
        if cases != [] or manifest.get("qualified_case_count") != 0:
            raise AssertionError("placeholder cannot contain or claim qualified cases")
        if require_complete:
            raise AssertionError("real qualification evidence is still missing")
        return
    if status != "qualified":
        raise AssertionError("unknown manifest status")
    if manifest.get("qualified_case_count") != len(cases):
        raise AssertionError("qualified count mismatch")
    validate_qualified_cases(cases, policy, intake_policy, require_complete=True)


def make_qualified_cases(policy, intake_policy):
    categories = list(intake_policy["categories"])
    records = []
    for index in range(policy["required_case_count"]):
        category = categories[index % len(categories)]
        records.append(
            {
                "case_id": f"qualification-case-{index:02d}",
                "category": category,
                "split": "qualification",
                "source_sha256": f"{index + 1:064x}",
                "consent_sha256": f"{index + 1001:064x}",
                "inspection_sha256": f"{index + 2001:064x}",
                "license": "owned-original",
                "source_format": "png",
                "width": 1024,
                "height": 1024,
                "file_bytes": 4096,
                "storage_object_id": f"rfv/qualification/case-{index:02d}",
                "privacy_review": "approved",
                "contains_public_pii": False,
                "source_verified": True,
                "consent_verified": True,
                "object_immutable": True,
                "decode_verified": True,
            }
        )
    return records


class RFV2QualificationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.intake_policy = load_json(INTAKE_POLICY_PATH)
        self.policy = load_json(POLICY_PATH)
        self.manifest = load_json(MANIFEST_PATH)

    def test_policy_aligns_with_immutable_intake_contract(self):
        validate_policy(self.policy, self.intake_policy)

    def test_adapter_accepts_complete_qualified_metadata(self):
        records = make_qualified_cases(self.policy, self.intake_policy)
        validate_qualified_cases(records, self.policy, self.intake_policy)
        manifest = {
            "schema": "vektoryum-rfv2-qualification-manifest-v1",
            "status": "qualified",
            "expected_case_count": 24,
            "qualified_case_count": 24,
            "public_repo_contains_raw_assets": False,
            "cases": records,
        }
        validate_manifest(manifest, self.policy, self.intake_policy, require_complete=True)

    def test_repository_manifest_matches_its_finite_lifecycle_state(self):
        validate_manifest(self.manifest, self.policy, self.intake_policy)
        if self.manifest["status"] == "qualified":
            validate_manifest(self.manifest, self.policy, self.intake_policy, require_complete=True)
            self.assertEqual(self.manifest["qualified_case_count"], 24)
        else:
            with self.assertRaises(AssertionError):
                validate_manifest(self.manifest, self.policy, self.intake_policy, require_complete=True)

    def test_invalid_or_fabricated_evidence_fails_closed(self):
        base = make_qualified_cases(self.policy, self.intake_policy)[0]
        bad_records = [
            dict(base, source_verified=False),
            dict(base, consent_sha256="bad"),
            dict(base, privacy_review="pending"),
            dict(base, contains_public_pii=True),
            dict(base, storage_object_id="https://example.com/private.png"),
            dict(base, license="unknown"),
            dict(base, split="calibration"),
            dict(base, decode_verified=False),
        ]
        for record in bad_records:
            with self.subTest(record=record):
                with self.assertRaises(AssertionError):
                    validate_qualified_cases([record], self.policy, self.intake_policy, require_complete=False)

        duplicate = copy.deepcopy(base)
        duplicate["case_id"] = "qualification-duplicate"
        with self.assertRaises(AssertionError):
            validate_qualified_cases([base, duplicate], self.policy, self.intake_policy, require_complete=False)

    def test_roadmap_allows_only_monotonic_rfv2_progression(self):
        roadmap = load_json(ROADMAP_PATH)
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 4)
        self.assertEqual([phase["id"] for phase in phases], ["RFV-1", "RFV-2", "RFV-3", "RFV-4"])
        self.assertEqual(phases[0]["status"], "merged")
        self.assertIn(phases[1]["status"], {"pending", "implemented", "merged"})
        self.assertEqual([phase["status"] for phase in phases[2:]], ["pending", "pending"])
        if self.manifest["status"] == "awaiting_real_assets":
            self.assertEqual(phases[1]["status"], "pending")
        else:
            self.assertIn(phases[1]["status"], {"implemented", "merged"})
            self.assertTrue((ROOT / phases[1]["evidence"]).is_file())
        self.assertTrue((ROOT / phases[1]["preparation_evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()
