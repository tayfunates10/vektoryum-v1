import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.regression.rfv3_live_prepare import (
    EXPECTED_CASE_COUNT,
    LivePrepareError,
    prepare_live_bundle,
    stable_identity,
)
from engine.regression.rfv3_measurement_runner import (
    EXPECTED_CASES_SHA256,
    QUALIFICATION_MANIFEST_PATH,
    canonical_sha256,
    load_json,
)


def _tar_info(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info


def _build_live_bundle(root: Path, *, mutate_stable=False):
    committed = load_json(QUALIFICATION_MANIFEST_PATH)
    live = copy.deepcopy(committed)
    for index, case in enumerate(live["cases"]):
        case["consent_sha256"] = hashlib.sha256(f"live-consent-{index}".encode()).hexdigest()
        case["inspection_sha256"] = hashlib.sha256(f"live-inspection-{index}".encode()).hexdigest()
    if mutate_stable:
        live["cases"][0]["category"] = "badge_seal"
    live_cases_sha = canonical_sha256(live["cases"])
    live["cases_sha256"] = live_cases_sha
    index = {
        "schema": "vektoryum-rfv2-live-bundle-index-v1",
        "qualified_case_count": EXPECTED_CASE_COUNT,
        "cases_sha256": live_cases_sha,
        "raw_assets_in_repository": False,
        "files": [],
    }
    bundle = root / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, value in (
            ("qualification-manifest.json", live),
            ("bundle-index.json", index),
        ):
            payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            archive.addfile(_tar_info(name, payload), io.BytesIO(payload))
    checksums = root / "checksums.json"
    checksums.write_text(
        json.dumps(
            {
                "schema": "vektoryum-rfv2-live-bundle-checksums-v1",
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "cases_sha256": live_cases_sha,
                "qualified_case_count": EXPECTED_CASE_COUNT,
                "raw_assets_in_repository": False,
            }
        ),
        encoding="utf-8",
    )
    return bundle, checksums, live_cases_sha


class RFV3LivePrepareTests(unittest.TestCase):
    def test_stable_identity_ignores_only_dynamic_evidence_digests(self):
        committed = load_json(QUALIFICATION_MANIFEST_PATH)
        live = copy.deepcopy(committed["cases"])
        live[0]["consent_sha256"] = "a" * 64
        live[0]["inspection_sha256"] = "b" * 64
        self.assertEqual(stable_identity(live), stable_identity(committed["cases"]))
        live[0]["source_sha256"] = "c" * 64
        self.assertNotEqual(stable_identity(live), stable_identity(committed["cases"]))

    def test_prepare_preserves_live_evidence_and_normalizes_measurement_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle, checksums, live_cases_sha = _build_live_bundle(root)
            destination = root / "prepared"
            with patch("engine.regression.rfv3_live_prepare._validate_object", return_value=None):
                evidence = prepare_live_bundle(bundle=bundle, checksums=checksums, destination=destination)
            normalized = json.loads((destination / "qualification-manifest.json").read_text())
            preserved = json.loads((destination / "live-qualification-manifest.json").read_text())
            index = json.loads((destination / "bundle-index.json").read_text())
        self.assertEqual(evidence["live_cases_sha256"], live_cases_sha)
        self.assertEqual(evidence["measurement_cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(normalized["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(preserved["cases_sha256"], live_cases_sha)
        self.assertEqual(index["cases_sha256"], EXPECTED_CASES_SHA256)
        self.assertEqual(index["live_cases_sha256"], live_cases_sha)
        self.assertTrue(index["measurement_identity_normalized"])

    def test_prepare_rejects_stable_source_identity_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle, checksums, _ = _build_live_bundle(root, mutate_stable=True)
            with patch("engine.regression.rfv3_live_prepare._validate_object", return_value=None):
                with self.assertRaisesRegex(LivePrepareError, "stable source identity drift"):
                    prepare_live_bundle(bundle=bundle, checksums=checksums, destination=root / "prepared")

    def test_prepare_rejects_bundle_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle, checksums, _ = _build_live_bundle(root)
            payload = json.loads(checksums.read_text())
            payload["bundle_sha256"] = "0" * 64
            checksums.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LivePrepareError, "bundle digest mismatch"):
                prepare_live_bundle(bundle=bundle, checksums=checksums, destination=root / "prepared")


if __name__ == "__main__":
    unittest.main()
