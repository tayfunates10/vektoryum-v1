import copy
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
ROADMAP_PATH = ROOT / "docs" / "real_world_fidelity_roadmap.json"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
REQUIRED_CATEGORIES = {
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
}
REQUIRED_SPLITS = {"calibration", "qualification", "holdout"}
REQUIRED_FIELDS = {
    "case_id",
    "category",
    "source_sha256",
    "consent_sha256",
    "license",
    "width",
    "height",
    "file_bytes",
    "source_format",
    "storage_object_id",
    "privacy_review",
    "contains_public_pii",
    "split",
}


def load_policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def validate_policy(policy):
    if policy.get("schema") != "vektoryum-rfv1-intake-v1":
        raise AssertionError("invalid intake schema")

    target = policy.get("target_cases")
    if not isinstance(target, int) or isinstance(target, bool) or not 100 <= target <= 300:
        raise AssertionError("target must remain finite and inside 100-300")

    if policy.get("storage_mode") != "external_immutable_object_store":
        raise AssertionError("mutable or public asset storage is forbidden")
    if policy.get("public_repo_contains_raw_assets") is not False:
        raise AssertionError("raw private assets must not be published")

    formats = policy.get("allowed_source_formats")
    if formats != ["png", "jpeg", "webp", "tiff"]:
        raise AssertionError("source format policy drift")

    for key in ("max_pixels", "max_file_bytes"):
        value = policy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AssertionError(f"invalid {key}")

    if set(policy.get("required_record_fields", [])) != REQUIRED_FIELDS:
        raise AssertionError("required record field drift")

    licenses = policy.get("allowed_licenses")
    if not isinstance(licenses, list) or len(licenses) != len(set(licenses)) or not licenses:
        raise AssertionError("invalid license allowlist")
    forbidden = policy.get("forbidden_sources")
    if not isinstance(forbidden, list) or set(licenses) & set(forbidden):
        raise AssertionError("source policy overlap")

    privacy = policy.get("privacy")
    if not isinstance(privacy, dict) or not all(value is True for value in privacy.values()):
        raise AssertionError("privacy controls must fail closed")

    splits = policy.get("splits")
    if not isinstance(splits, dict) or set(splits) != REQUIRED_SPLITS:
        raise AssertionError("invalid split set")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in splits.values()):
        raise AssertionError("invalid split quota")
    if sum(splits.values()) != target:
        raise AssertionError("split quotas do not match target")

    categories = policy.get("categories")
    if not isinstance(categories, dict) or set(categories) != REQUIRED_CATEGORIES:
        raise AssertionError("required category drift")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in categories.values()):
        raise AssertionError("invalid category quota")
    if sum(categories.values()) != target:
        raise AssertionError("category quotas do not match target")


def validate_record(record, policy):
    if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
        raise AssertionError("missing intake record fields")

    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
        raise AssertionError("invalid case_id")

    for key in ("source_sha256", "consent_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise AssertionError(f"invalid {key}")

    if record.get("category") not in policy["categories"]:
        raise AssertionError("unknown category")
    if record.get("license") not in policy["allowed_licenses"]:
        raise AssertionError("unreviewed license")
    if record.get("source_format") not in policy["allowed_source_formats"]:
        raise AssertionError("unsupported source format")
    if record.get("split") not in policy["splits"]:
        raise AssertionError("unknown split")

    width, height = record.get("width"), record.get("height")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (width, height)):
        raise AssertionError("invalid dimensions")
    if width * height > policy["max_pixels"]:
        raise AssertionError("pixel budget exceeded")

    file_bytes = record.get("file_bytes")
    if not isinstance(file_bytes, int) or isinstance(file_bytes, bool) or not 0 < file_bytes <= policy["max_file_bytes"]:
        raise AssertionError("file budget exceeded")

    object_id = record.get("storage_object_id")
    if not isinstance(object_id, str) or not object_id.startswith("rfv/") or "://" in object_id:
        raise AssertionError("storage identifier must be opaque")
    if record.get("privacy_review") != "approved":
        raise AssertionError("privacy review missing")
    if record.get("contains_public_pii") is not False:
        raise AssertionError("public PII is forbidden")

    redaction_required = record.get("redaction_required", False)
    if not isinstance(redaction_required, bool):
        raise AssertionError("invalid redaction flag")
    if redaction_required:
        redacted = record.get("redacted_source_sha256")
        if not isinstance(redacted, str) or not SHA256_RE.fullmatch(redacted):
            raise AssertionError("missing redacted digest")
        if redacted == record["source_sha256"]:
            raise AssertionError("redacted digest must identify a new artifact")


def validate_records(records, policy, require_complete=False):
    validate_policy(policy)
    if not isinstance(records, list) or not records:
        raise AssertionError("empty intake records")

    for record in records:
        validate_record(record, policy)

    for key in ("case_id", "source_sha256", "storage_object_id"):
        values = [record[key] for record in records]
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate {key}")

    if require_complete:
        if len(records) != policy["target_cases"]:
            raise AssertionError("corpus target not met")
        if Counter(record["category"] for record in records) != Counter(policy["categories"]):
            raise AssertionError("category quota mismatch")
        if Counter(record["split"] for record in records) != Counter(policy["splits"]):
            raise AssertionError("split quota mismatch")


def make_complete_metadata(policy):
    categories = []
    for category, quota in policy["categories"].items():
        categories.extend([category] * quota)
    splits = []
    for split, quota in policy["splits"].items():
        splits.extend([split] * quota)

    records = []
    for index, (category, split) in enumerate(zip(categories, splits)):
        records.append(
            {
                "case_id": f"real-case-{index:03d}",
                "category": category,
                "source_sha256": f"{index + 1:064x}",
                "consent_sha256": f"{index + 1001:064x}",
                "license": "owned-original",
                "width": 1024,
                "height": 1024,
                "file_bytes": 4096,
                "source_format": "png",
                "storage_object_id": f"rfv/real-case-{index:03d}",
                "privacy_review": "approved",
                "contains_public_pii": False,
                "split": split,
            }
        )
    return records


class RFV1IntakeContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_policy_is_finite_private_and_balanced(self):
        validate_policy(self.policy)

    def test_roadmap_is_finite_and_forward_compatible(self):
        roadmap = json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 4)
        self.assertEqual([phase["id"] for phase in phases], [f"RFV-{index}" for index in range(1, 5)])

        seen_non_merged = False
        pending_started = False
        implemented_count = 0
        for phase in phases:
            status = phase["status"]
            self.assertIn(status, {"merged", "implemented", "pending"})
            if status == "merged":
                self.assertFalse(seen_non_merged)
            elif status == "implemented":
                seen_non_merged = True
                self.assertFalse(pending_started)
                implemented_count += 1
                self.assertLessEqual(implemented_count, 1)
            else:
                seen_non_merged = True
                pending_started = True

        self.assertIn(phases[0]["status"], {"implemented", "merged"})
        self.assertTrue((ROOT / phases[0]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[0]["acceptance"]), 5)

    def test_complete_metadata_meets_declared_quotas(self):
        validate_records(make_complete_metadata(self.policy), self.policy, require_complete=True)

    def test_fails_closed_on_invalid_or_private_records(self):
        base = make_complete_metadata(self.policy)[0]
        bad_records = [
            dict(base, source_sha256="bad"),
            dict(base, license="unknown"),
            dict(base, contains_public_pii=True),
            dict(base, privacy_review="pending"),
            dict(base, width=self.policy["max_pixels"] + 1, height=2),
            dict(base, storage_object_id="https://example.com/private.png"),
            dict(base, redaction_required=True),
        ]
        for record in bad_records:
            with self.subTest(record=record):
                with self.assertRaises(AssertionError):
                    validate_records([record], self.policy)

    def test_duplicate_identity_and_corpus_shrinkage_fail_closed(self):
        base = make_complete_metadata(self.policy)[0]
        duplicate = copy.deepcopy(base)
        duplicate["case_id"] = "real-case-duplicate"
        with self.assertRaises(AssertionError):
            validate_records([base, duplicate], self.policy)

        incomplete = make_complete_metadata(self.policy)[:-1]
        with self.assertRaises(AssertionError):
            validate_records(incomplete, self.policy, require_complete=True)


if __name__ == "__main__":
    unittest.main()
