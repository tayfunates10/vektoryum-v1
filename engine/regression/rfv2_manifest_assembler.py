from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
QUALIFICATION_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_policy.json"
TOOL_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_secure_intake_policy.json"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AssemblyError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"invalid JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise AssemblyError(f"JSON root must be an object: {path}")
    return data


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_external_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise AssemblyError(f"{label} must resolve outside the repository")
    if not resolved.exists() or not resolved.is_dir() or resolved.is_symlink():
        raise AssemblyError(f"{label} must be an existing non-symlink directory")
    return resolved


def require_external_output(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise AssemblyError(f"{label} must resolve outside the repository")
    if resolved.exists() and resolved.is_symlink():
        raise AssemblyError(f"{label} symlinks are forbidden")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.parent.is_symlink():
        raise AssemblyError(f"{label} parent symlinks are forbidden")
    return resolved


def _validate_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AssemblyError(f"invalid {label}")


def _load_policies() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intake = load_json(INTAKE_POLICY_PATH)
    qualification = load_json(QUALIFICATION_POLICY_PATH)
    tool = load_json(TOOL_POLICY_PATH)
    if tool.get("schema") != "vektoryum-rfv2-secure-intake-v1":
        raise AssemblyError("secure intake policy drift")
    if qualification.get("schema") != "vektoryum-rfv2-qualification-policy-v1":
        raise AssemblyError("qualification policy drift")
    if qualification.get("required_case_count") != intake.get("splits", {}).get("qualification"):
        raise AssemblyError("qualification count drift")
    return intake, qualification, tool


def validate_record(
    record: dict[str, Any],
    intake: dict[str, Any],
    qualification: dict[str, Any],
    tool: dict[str, Any],
) -> dict[str, Any]:
    required = list(qualification["required_case_fields"])
    allowed_top_level = {"schema", "inspection", *required}
    if set(record) != allowed_top_level:
        raise AssemblyError("record contains missing or unapproved fields")
    if record.get("schema") != tool["record_schema"]:
        raise AssemblyError("record schema mismatch")

    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
        raise AssemblyError("invalid case_id")
    if record.get("category") not in intake["categories"]:
        raise AssemblyError("unknown category")
    if record.get("split") != qualification["required_split"]:
        raise AssemblyError("wrong split")

    for key in ("source_sha256", "consent_sha256", "inspection_sha256"):
        _validate_digest(record.get(key), key)
    if record.get("license") not in intake["allowed_licenses"]:
        raise AssemblyError("unreviewed license")
    if record.get("source_format") not in intake["allowed_source_formats"]:
        raise AssemblyError("unsupported source format")

    width = record.get("width")
    height = record.get("height")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (width, height)):
        raise AssemblyError("invalid dimensions")
    if width * height > intake["max_pixels"]:
        raise AssemblyError("pixel budget exceeded")
    file_bytes = record.get("file_bytes")
    if not isinstance(file_bytes, int) or isinstance(file_bytes, bool) or not 0 < file_bytes <= intake["max_file_bytes"]:
        raise AssemblyError("file budget exceeded")

    object_id = record.get("storage_object_id")
    if not isinstance(object_id, str) or not object_id.startswith("rfv/qualification/") or "://" in object_id:
        raise AssemblyError("invalid storage object identifier")
    if record.get("privacy_review") != "approved" or record.get("contains_public_pii") is not False:
        raise AssemblyError("privacy gate failed")
    for field in qualification["required_boolean_true_fields"]:
        if record.get(field) is not True:
            raise AssemblyError(f"verification failed: {field}")

    inspection = record.get("inspection")
    expected_inspection = {
        "schema": tool["inspection_schema"],
        "case_id": record["case_id"],
        "category": record["category"],
        "source_sha256": record["source_sha256"],
        "consent_sha256": record["consent_sha256"],
        "source_format": record["source_format"],
        "width": record["width"],
        "height": record["height"],
        "file_bytes": record["file_bytes"],
    }
    if inspection != expected_inspection:
        raise AssemblyError("inspection evidence does not match the record")
    if canonical_sha256(inspection) != record["inspection_sha256"]:
        raise AssemblyError("inspection digest mismatch")

    return {key: record[key] for key in required}


def load_record_directory(records_dir: Path) -> list[dict[str, Any]]:
    records_dir = require_external_directory(records_dir, "records directory")
    files = sorted(records_dir.glob("*.json"))
    if not files:
        raise AssemblyError("records directory contains no JSON records")
    records: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise AssemblyError("record symlinks and non-files are forbidden")
        records.append(load_json(path))
    return records


def assemble_manifest(records: list[dict[str, Any]], *, require_complete: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    intake, qualification, tool = _load_policies()
    if not isinstance(records, list) or not records or any(not isinstance(item, dict) for item in records):
        raise AssemblyError("at least one record is required")
    if len(records) > qualification["required_case_count"]:
        raise AssemblyError("qualification corpus exceeds the finite target")

    normalized = [validate_record(item, intake, qualification, tool) for item in records]
    normalized.sort(key=lambda item: item["case_id"])
    for key in ("case_id", "source_sha256", "storage_object_id", "inspection_sha256"):
        values = [item[key] for item in normalized]
        if len(values) != len(set(values)):
            raise AssemblyError(f"duplicate {key}")

    category_counts = Counter(item["category"] for item in normalized)
    missing_categories = sorted(set(intake["categories"]) - set(category_counts))
    complete = len(normalized) == qualification["required_case_count"] and not missing_categories
    if require_complete and not complete:
        raise AssemblyError("complete qualification gate not met")

    cases_sha256 = canonical_sha256(normalized)
    manifest = {
        "schema": "vektoryum-rfv2-qualification-manifest-v1",
        "status": "qualified" if complete else "collecting",
        "expected_case_count": qualification["required_case_count"],
        "qualified_case_count": len(normalized),
        "public_repo_contains_raw_assets": False,
        "cases_sha256": cases_sha256,
        "cases": normalized,
    }
    audit = {
        "schema": "vektoryum-rfv2-assembly-audit-v1",
        "complete": complete,
        "required_case_count": qualification["required_case_count"],
        "qualified_case_count": len(normalized),
        "category_counts": {key: category_counts.get(key, 0) for key in sorted(intake["categories"])},
        "missing_categories": missing_categories,
        "duplicate_case_ids": 0,
        "duplicate_source_digests": 0,
        "duplicate_storage_objects": 0,
        "duplicate_inspection_digests": 0,
        "cases_sha256": cases_sha256,
    }
    return manifest, audit


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".rfv-assembly-", delete=False) as temp:
            temp_name = temp.name
            json.dump(payload, temp, indent=2, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble reviewed RFV-2 records into a deterministic external manifest.")
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_out = require_external_output(args.manifest_out, "manifest output")
        audit_out = require_external_output(args.audit_out, "audit output")
        if manifest_out == audit_out:
            raise AssemblyError("manifest and audit outputs must be different files")
        records = load_record_directory(args.records_dir)
        manifest, audit = assemble_manifest(records, require_complete=args.require_complete)
        write_json_atomic(manifest_out, manifest)
        write_json_atomic(audit_out, audit)
    except AssemblyError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": manifest["status"], "qualified_case_count": manifest["qualified_case_count"], "cases_sha256": manifest["cases_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
