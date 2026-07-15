from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from engine.regression.rfv2_secure_intake import ROOT

SOURCE_SELECTION_PATH = ROOT / "engine" / "regression" / "rfv2_public_source_manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CASE_COUNT = 24
RETENTION_DAYS = 90


class BundleError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise BundleError(f"{label} must resolve outside the repository")
    if not resolved.exists() or not resolved.is_dir() or resolved.is_symlink():
        raise BundleError(f"{label} must be an existing non-symlink directory")
    return resolved


def require_external_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise BundleError(f"{label} must resolve outside the repository")
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise BundleError(f"{label} must be an existing non-symlink file")
    return resolved


def require_external_output(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise BundleError(f"{label} must resolve outside the repository")
    if resolved.exists() and resolved.is_symlink():
        raise BundleError(f"{label} symlinks are forbidden")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.parent.is_symlink():
        raise BundleError(f"{label} parent symlinks are forbidden")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise BundleError(f"invalid {label}")
    return value


def _safe_storage_path(storage_root: Path, object_id: Any) -> Path:
    if not isinstance(object_id, str) or not object_id.startswith("rfv/qualification/"):
        raise BundleError("invalid storage object id")
    relative = Path(object_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise BundleError("unsafe storage object id")
    resolved = (storage_root / relative).resolve()
    if not _is_inside(resolved, storage_root):
        raise BundleError("storage object escapes the storage root")
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise BundleError("storage object is missing or not a regular file")
    return resolved


def validate_evidence(
    *,
    storage_root: Path,
    download_root: Path,
    records_dir: Path,
    manifest_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, Path]]]:
    storage_root = require_external_directory(storage_root, "storage root")
    download_root = require_external_directory(download_root, "download root")
    records_dir = require_external_directory(records_dir, "records directory")
    manifest_path = require_external_file(manifest_path, "qualification manifest")
    audit_path = require_external_file(audit_path, "qualification audit")

    manifest = load_json(manifest_path)
    audit = load_json(audit_path)
    if manifest.get("schema") != "vektoryum-rfv2-qualification-manifest-v1":
        raise BundleError("qualification manifest schema mismatch")
    if manifest.get("status") != "qualified":
        raise BundleError("qualification manifest is not qualified")
    if manifest.get("expected_case_count") != EXPECTED_CASE_COUNT:
        raise BundleError("qualification target drift")
    if manifest.get("qualified_case_count") != EXPECTED_CASE_COUNT:
        raise BundleError("qualification case count is incomplete")
    if manifest.get("public_repo_contains_raw_assets") is not False:
        raise BundleError("raw asset repository boundary failed")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise BundleError("qualification cases are incomplete")
    if _require_sha256(manifest.get("cases_sha256"), "cases_sha256") != sha256_bytes(canonical_bytes(cases)):
        raise BundleError("qualification cases digest mismatch")

    if audit.get("schema") != "vektoryum-rfv2-assembly-audit-v1":
        raise BundleError("qualification audit schema mismatch")
    if audit.get("complete") is not True:
        raise BundleError("qualification audit is incomplete")
    if audit.get("required_case_count") != EXPECTED_CASE_COUNT or audit.get("qualified_case_count") != EXPECTED_CASE_COUNT:
        raise BundleError("qualification audit count mismatch")
    if audit.get("missing_categories") != []:
        raise BundleError("qualification category coverage is incomplete")
    for key in (
        "duplicate_case_ids",
        "duplicate_source_digests",
        "duplicate_storage_objects",
        "duplicate_inspection_digests",
    ):
        if audit.get(key) != 0:
            raise BundleError(f"qualification audit duplicate failure: {key}")
    if audit.get("cases_sha256") != manifest.get("cases_sha256"):
        raise BundleError("qualification audit digest mismatch")

    record_files = sorted(path for path in records_dir.glob("*.json") if path.name != ".existing-records.json")
    if len(record_files) != EXPECTED_CASE_COUNT:
        raise BundleError("exactly 24 individual records are required")
    records_by_case: dict[str, Path] = {}
    for path in record_files:
        if path.is_symlink() or not path.is_file():
            raise BundleError("record symlinks and non-files are forbidden")
        record = load_json(path)
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id in records_by_case:
            raise BundleError("invalid or duplicate record case id")
        if record.get("schema") != "vektoryum-rfv2-qualified-case-v1":
            raise BundleError("individual record schema mismatch")
        records_by_case[case_id] = path

    archive_files: list[tuple[str, Path]] = []
    manifest_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise BundleError("qualification case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in manifest_case_ids:
            raise BundleError("invalid or duplicate qualification case id")
        manifest_case_ids.add(case_id)
        record_path = records_by_case.get(case_id)
        if record_path is None:
            raise BundleError("qualification case is missing its individual record")
        source_sha256 = _require_sha256(case.get("source_sha256"), "source_sha256")
        object_path = _safe_storage_path(storage_root, case.get("storage_object_id"))
        if sha256_file(object_path) != source_sha256:
            raise BundleError("stored object digest mismatch")

        proof_path = (download_root / case_id / "license-proof.json").resolve()
        if not _is_inside(proof_path, download_root):
            raise BundleError("license proof escapes the download root")
        if not proof_path.exists() or not proof_path.is_file() or proof_path.is_symlink():
            raise BundleError("license proof is missing or not a regular file")
        proof = load_json(proof_path)
        if proof.get("schema") != "vektoryum-rfv2-public-license-proof-v1" or proof.get("case_id") != case_id:
            raise BundleError("license proof identity mismatch")
        if proof.get("canonical_source_sha256") != source_sha256:
            raise BundleError("license proof source digest mismatch")
        for key in (
            "source_page_sha256",
            "license_proof_sha256",
            "downloaded_asset_sha256",
            "canonical_source_sha256",
        ):
            _require_sha256(proof.get(key), key)

        archive_files.extend(
            [
                (f"objects/{case['storage_object_id']}", object_path),
                (f"proofs/{case_id}.json", proof_path),
                (f"records/{case_id}.json", record_path),
            ]
        )

    if set(records_by_case) != manifest_case_ids:
        raise BundleError("individual record set does not match the qualification manifest")
    return manifest, audit, archive_files


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o444
    return info


def _write_deterministic_tar_gz(output: Path, entries: list[tuple[str, bytes]]) -> None:
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, prefix=".rfv2-bundle-", delete=False) as raw:
            temp_name = raw.name
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for name, payload in sorted(entries, key=lambda item: item[0]):
                        archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp_name, output)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".rfv2-json-", delete=False) as temp:
            temp_name = temp.name
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def build_bundle(
    *,
    storage_root: Path,
    download_root: Path,
    records_dir: Path,
    manifest_path: Path,
    audit_path: Path,
    bundle_out: Path,
    checksums_out: Path,
) -> dict[str, Any]:
    bundle_out = require_external_output(bundle_out, "bundle output")
    checksums_out = require_external_output(checksums_out, "checksums output")
    if bundle_out == checksums_out:
        raise BundleError("bundle and checksums outputs must be different files")

    manifest, audit, archive_files = validate_evidence(
        storage_root=storage_root,
        download_root=download_root,
        records_dir=records_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )
    source_selection = load_json(SOURCE_SELECTION_PATH)
    if source_selection.get("schema") != "vektoryum-rfv2-public-source-manifest-v1":
        raise BundleError("source selection schema mismatch")

    entries: list[tuple[str, bytes]] = []
    file_index: list[dict[str, Any]] = []
    for archive_name, path in archive_files:
        payload = path.read_bytes()
        entries.append((archive_name, payload))
        file_index.append({"path": archive_name, "sha256": sha256_bytes(payload), "bytes": len(payload)})
    fixed_payloads = {
        "qualification-manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "qualification-audit.json": json.dumps(audit, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        "source-selection-manifest.json": json.dumps(source_selection, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    }
    for archive_name, payload in fixed_payloads.items():
        entries.append((archive_name, payload))
        file_index.append({"path": archive_name, "sha256": sha256_bytes(payload), "bytes": len(payload)})

    index = {
        "schema": "vektoryum-rfv2-live-bundle-index-v1",
        "qualified_case_count": EXPECTED_CASE_COUNT,
        "cases_sha256": manifest["cases_sha256"],
        "raw_assets_in_repository": False,
        "files": sorted(file_index, key=lambda item: item["path"]),
    }
    index_payload = json.dumps(index, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    entries.append(("bundle-index.json", index_payload))
    _write_deterministic_tar_gz(bundle_out, entries)

    checksums = {
        "schema": "vektoryum-rfv2-live-bundle-checksums-v1",
        "bundle_sha256": sha256_file(bundle_out),
        "bundle_bytes": bundle_out.stat().st_size,
        "qualification_manifest_sha256": sha256_file(require_external_file(manifest_path, "qualification manifest")),
        "qualification_audit_sha256": sha256_file(require_external_file(audit_path, "qualification audit")),
        "cases_sha256": manifest["cases_sha256"],
        "qualified_case_count": EXPECTED_CASE_COUNT,
        "raw_assets_in_repository": False,
        "storage_mode": "github_actions_immutable_artifact",
        "retention_days": RETENTION_DAYS,
    }
    write_json_atomic(checksums_out, checksums)
    return checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic RFV-2 live qualification artifact bundle.")
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--bundle-out", type=Path, required=True)
    parser.add_argument("--checksums-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_bundle(
            storage_root=args.storage_root,
            download_root=args.download_root,
            records_dir=args.records_dir,
            manifest_path=args.manifest,
            audit_path=args.audit,
            bundle_out=args.bundle_out,
            checksums_out=args.checksums_out,
        )
    except BundleError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "bundled", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
