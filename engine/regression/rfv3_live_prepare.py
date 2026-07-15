from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from engine.regression.rfv3_measurement_runner import (
    EXPECTED_CASES_SHA256,
    QUALIFICATION_MANIFEST_PATH,
    ROOT,
    canonical_sha256,
    load_json,
    require_external_output,
    sha256_file,
)

EXPECTED_CASE_COUNT = 24
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STABLE_IDENTITY_FIELDS = (
    "case_id",
    "category",
    "source_sha256",
    "license",
    "source_format",
    "storage_object_id",
    "file_bytes",
    "width",
    "height",
    "split",
)


class LivePrepareError(RuntimeError):
    pass


def _external_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise LivePrepareError(f"{label} must resolve outside the repository")
    if not resolved.is_file() or resolved.is_symlink():
        raise LivePrepareError(f"{label} must be an existing non-symlink file")
    return resolved


def _safe_member_path(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise LivePrepareError("unsafe RFV-2 bundle path")
    resolved = (destination / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(destination.resolve())
    except ValueError as exc:
        raise LivePrepareError("RFV-2 bundle path escapes extraction root") from exc
    return resolved


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".rfv3-live-", delete=False) as temp:
            temp_name = temp.name
            json.dump(value, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def stable_identity(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or any(field not in case for field in STABLE_IDENTITY_FIELDS):
            raise LivePrepareError("RFV-2 case is missing stable measurement identity fields")
        projection.append({field: case[field] for field in STABLE_IDENTITY_FIELDS})
    return sorted(projection, key=lambda item: item["case_id"])


def _validate_evidence_case(case: dict[str, Any]) -> None:
    for field in ("consent_sha256", "inspection_sha256", "source_sha256"):
        value = case.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise LivePrepareError(f"invalid live evidence digest: {field}")
    for field in ("source_verified", "consent_verified", "object_immutable", "decode_verified"):
        if case.get(field) is not True:
            raise LivePrepareError(f"live evidence verification failed: {field}")
    if case.get("privacy_review") != "approved" or case.get("contains_public_pii") is not False:
        raise LivePrepareError("live evidence privacy review failed")


def _validate_object(destination: Path, case: dict[str, Any]) -> None:
    object_id = case.get("storage_object_id")
    if not isinstance(object_id, str) or not object_id.startswith("rfv/qualification/"):
        raise LivePrepareError("invalid RFV-2 storage object id")
    relative = Path("objects") / object_id
    resolved = (destination / relative).resolve()
    try:
        resolved.relative_to(destination.resolve())
    except ValueError as exc:
        raise LivePrepareError("RFV-2 object escapes extraction root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise LivePrepareError("RFV-2 source object is missing")
    if sha256_file(resolved) != case["source_sha256"]:
        raise LivePrepareError("RFV-2 source object digest mismatch")


def prepare_live_bundle(*, bundle: Path, checksums: Path, destination: Path) -> dict[str, Any]:
    bundle = _external_file(bundle, "RFV-2 corpus bundle")
    checksums = _external_file(checksums, "RFV-2 bundle checksums")
    destination = require_external_output(destination, "RFV-3 extracted corpus")
    if any(destination.iterdir()):
        raise LivePrepareError("RFV-3 extraction destination must be empty")

    checksum_payload = load_json(checksums)
    if checksum_payload.get("schema") != "vektoryum-rfv2-live-bundle-checksums-v1":
        raise LivePrepareError("RFV-2 bundle checksum schema mismatch")
    if checksum_payload.get("qualified_case_count") != EXPECTED_CASE_COUNT:
        raise LivePrepareError("RFV-2 bundle case count mismatch")
    if checksum_payload.get("bundle_sha256") != sha256_file(bundle):
        raise LivePrepareError("RFV-2 bundle digest mismatch")
    if checksum_payload.get("raw_assets_in_repository") is not False:
        raise LivePrepareError("RFV-2 raw-asset boundary mismatch")
    live_cases_sha = checksum_payload.get("cases_sha256")
    if not isinstance(live_cases_sha, str) or not SHA256_RE.fullmatch(live_cases_sha):
        raise LivePrepareError("invalid RFV-2 live case-set digest")

    seen: set[str] = set()
    try:
        with tarfile.open(bundle, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name in seen:
                    raise LivePrepareError("duplicate RFV-2 bundle member")
                seen.add(member.name)
                output = _safe_member_path(destination, member.name)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise LivePrepareError("non-regular RFV-2 bundle member")
                source = archive.extractfile(member)
                if source is None:
                    raise LivePrepareError("RFV-2 bundle member cannot be read")
                output.parent.mkdir(parents=True, exist_ok=True)
                with source, output.open("wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
    except (tarfile.TarError, OSError) as exc:
        raise LivePrepareError("RFV-2 bundle extraction failed") from exc

    live_manifest_path = destination / "qualification-manifest.json"
    live_index_path = destination / "bundle-index.json"
    if not live_manifest_path.is_file() or not live_index_path.is_file():
        raise LivePrepareError("RFV-2 bundle metadata is missing")
    live_manifest = load_json(live_manifest_path)
    live_index = load_json(live_index_path)
    committed_manifest = load_json(QUALIFICATION_MANIFEST_PATH)

    if live_manifest.get("schema") != "vektoryum-rfv2-qualification-manifest-v1":
        raise LivePrepareError("RFV-2 live manifest schema mismatch")
    if live_manifest.get("status") != "qualified" or live_manifest.get("qualified_case_count") != EXPECTED_CASE_COUNT:
        raise LivePrepareError("RFV-2 live manifest is incomplete")
    live_cases = live_manifest.get("cases")
    committed_cases = committed_manifest.get("cases")
    if not isinstance(live_cases, list) or len(live_cases) != EXPECTED_CASE_COUNT:
        raise LivePrepareError("RFV-2 live case set is incomplete")
    if not isinstance(committed_cases, list) or len(committed_cases) != EXPECTED_CASE_COUNT:
        raise LivePrepareError("committed RFV-2 case set is incomplete")
    if canonical_sha256(live_cases) != live_cases_sha or live_manifest.get("cases_sha256") != live_cases_sha:
        raise LivePrepareError("RFV-2 live case-set digest mismatch")
    if committed_manifest.get("cases_sha256") != EXPECTED_CASES_SHA256 or canonical_sha256(committed_cases) != EXPECTED_CASES_SHA256:
        raise LivePrepareError("committed RFV-2 case-set digest mismatch")
    if stable_identity(live_cases) != stable_identity(committed_cases):
        raise LivePrepareError("RFV-2 stable source identity drift")

    if live_index.get("schema") != "vektoryum-rfv2-live-bundle-index-v1":
        raise LivePrepareError("RFV-2 live bundle index schema mismatch")
    if live_index.get("qualified_case_count") != EXPECTED_CASE_COUNT or live_index.get("cases_sha256") != live_cases_sha:
        raise LivePrepareError("RFV-2 live bundle index identity mismatch")
    if live_index.get("raw_assets_in_repository") is not False:
        raise LivePrepareError("RFV-2 live bundle repository boundary mismatch")

    for case in live_cases:
        _validate_evidence_case(case)
        _validate_object(destination, case)

    # Preserve the exact live evidence, then expose the frozen committed manifest to
    # the production measurement runner. Only run-specific proof/inspection digests
    # differ; source identity and source bytes must match exactly.
    _write_json_atomic(destination / "live-qualification-manifest.json", live_manifest)
    _write_json_atomic(destination / "live-bundle-index.json", live_index)
    _write_json_atomic(live_manifest_path, committed_manifest)
    normalized_index = dict(live_index)
    normalized_index["cases_sha256"] = EXPECTED_CASES_SHA256
    normalized_index["live_cases_sha256"] = live_cases_sha
    normalized_index["measurement_identity_normalized"] = True
    _write_json_atomic(live_index_path, normalized_index)

    return {
        "schema": "vektoryum-rfv3-live-prepare-v1",
        "case_count": EXPECTED_CASE_COUNT,
        "live_cases_sha256": live_cases_sha,
        "measurement_cases_sha256": EXPECTED_CASES_SHA256,
        "stable_source_identity_sha256": hashlib.sha256(_canonical_bytes(stable_identity(committed_cases))).hexdigest(),
        "raw_assets_in_repository": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely prepare a dynamic RFV-2 live bundle for frozen RFV-3 measurement identity.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = prepare_live_bundle(bundle=args.bundle, checksums=args.checksums, destination=args.destination)
        evidence_out = require_external_output(args.evidence_out.parent, "RFV-3 preparation evidence directory") / args.evidence_out.name
        _write_json_atomic(evidence_out, evidence)
    except LivePrepareError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "prepared", **evidence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
