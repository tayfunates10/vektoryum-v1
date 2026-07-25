from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from engine.regression.rfv2_immutable_fallback import (
    ImmutableCorpusError,
    extract_and_verify_artifact,
)


IDENTITY_KEYS = (
    "case_id",
    "category",
    "license",
    "source_sha256",
    "source_format",
    "file_bytes",
    "width",
    "height",
    "storage_object_id",
)


def _identities() -> dict[str, dict]:
    result = {}
    for index in range(1, 25):
        case = {
            "case_id": f"qualification-public-{index:02d}",
            "category": "flat_logo",
            "license": "cc0",
            "source_sha256": f"{index:064x}",
            "source_format": "png",
            "file_bytes": index,
            "width": 32,
            "height": 32,
            "storage_object_id": f"sha256/{index:064x}",
        }
        result[case["case_id"]] = {key: case[key] for key in IDENTITY_KEYS}
    return result


def _archive(expected: dict[str, dict], *, unsafe: bool = False) -> tuple[bytes, dict]:
    cases = list(expected.values())
    bundle = b"deterministic-corpus"
    audit = {"complete": True, "missing_categories": []}
    manifest = {
        "status": "qualified",
        "qualified_case_count": 24,
        "cases_sha256": "c" * 64,
        "cases": cases,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    audit_bytes = (json.dumps(audit, sort_keys=True) + "\n").encode()
    checksums = {
        "cases_sha256": manifest["cases_sha256"],
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "qualification_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "qualification_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "storage_mode": "github_actions_immutable_artifact",
        "qualified_case_count": 24,
    }
    checksums_bytes = (json.dumps(checksums, sort_keys=True) + "\n").encode()

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("rfv2-public-corpus.tar.gz", bundle)
        archive.writestr("qualification-manifest.json", manifest_bytes)
        archive.writestr("qualification-audit.json", audit_bytes)
        archive.writestr("bundle-checksums.json", checksums_bytes)
        if unsafe:
            archive.writestr("../escape.txt", b"forbidden")
    payload = output.getvalue()
    artifact = {
        "id": 123,
        "name": "rfv2-public-qualification-corpus-test",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    return payload, artifact


class ImmutableCorpusFallbackTests(unittest.TestCase):
    def test_accepts_exact_committed_identity_and_internal_digests(self) -> None:
        expected = _identities()
        payload, artifact = _archive(expected)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            verified = extract_and_verify_artifact(payload, artifact, out, expected)
            self.assertEqual(verified["manifest"]["qualified_case_count"], 24)
            self.assertTrue((out / "rfv2-public-corpus.tar.gz").is_file())
            self.assertEqual(
                verified["checksums"]["bundle_sha256"],
                hashlib.sha256(b"deterministic-corpus").hexdigest(),
            )

    def test_rejects_archive_digest_mismatch(self) -> None:
        expected = _identities()
        payload, artifact = _archive(expected)
        artifact["digest"] = "sha256:" + "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ImmutableCorpusError, "archive digest mismatch"):
                extract_and_verify_artifact(payload, artifact, Path(directory), expected)

    def test_rejects_path_traversal_layout(self) -> None:
        expected = _identities()
        payload, artifact = _archive(expected, unsafe=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ImmutableCorpusError, "layout rejected"):
                extract_and_verify_artifact(payload, artifact, Path(directory), expected)
            self.assertFalse((Path(directory).parent / "escape.txt").exists())

    def test_rejects_committed_identity_drift(self) -> None:
        expected = _identities()
        payload, artifact = _archive(expected)
        drifted = dict(expected)
        first = next(iter(drifted))
        drifted[first] = {**drifted[first], "source_sha256": "f" * 64}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ImmutableCorpusError, "source identity drift"):
                extract_and_verify_artifact(payload, artifact, Path(directory), drifted)


if __name__ == "__main__":
    unittest.main()
