from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
QUALIFICATION_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_qualification_policy.json"
TOOL_POLICY_PATH = ROOT / "engine" / "regression" / "rfv2_secure_intake_policy.json"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
FORMAT_MAP = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "TIFF": "tiff"}


class IntakeError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise IntakeError(f"invalid JSON object: {path}")
    return data


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


def require_regular_input(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise IntakeError(f"{label} must be an existing regular file")
    if path.is_symlink():
        raise IntakeError(f"{label} symlinks are forbidden")
    if path.stat().st_size <= 0:
        raise IntakeError(f"{label} must not be empty")
    return path.resolve()


def require_external_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _is_inside(resolved, ROOT):
        raise IntakeError(f"{label} must resolve outside the repository")
    return resolved


def inspect_image(source: Path, intake_policy: dict[str, Any]) -> dict[str, Any]:
    file_bytes = source.stat().st_size
    if file_bytes > intake_policy["max_file_bytes"]:
        raise IntakeError("source exceeds the file-size budget")
    try:
        with Image.open(source) as image:
            detected = FORMAT_MAP.get(image.format or "")
            width, height = image.size
            image.verify()
        with Image.open(source) as image:
            image.load()
    except Exception as exc:
        raise IntakeError("source image could not be decoded") from exc
    if detected not in intake_policy["allowed_source_formats"]:
        raise IntakeError("source image format is not allowed")
    if width <= 0 or height <= 0 or width * height > intake_policy["max_pixels"]:
        raise IntakeError("source dimensions exceed the pixel budget")
    return {
        "source_format": detected,
        "width": width,
        "height": height,
        "file_bytes": file_bytes,
    }


def _load_existing_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(require_regular_input(path, "existing records").read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise IntakeError("existing records must be a JSON list or manifest with a cases list")
    return data


def _reject_duplicates(record: dict[str, Any], existing: list[dict[str, Any]]) -> None:
    for key in ("case_id", "source_sha256", "storage_object_id", "inspection_sha256"):
        values = {item.get(key) for item in existing}
        if record[key] in values:
            raise IntakeError(f"duplicate {key}")


def _copy_content_addressed(source: Path, storage_root: Path, source_sha256: str, extension: str) -> tuple[Path, str]:
    object_id = f"rfv/qualification/{source_sha256[:2]}/{source_sha256}.{extension}"
    target = storage_root / object_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or sha256_file(target) != source_sha256:
            raise IntakeError("existing storage object failed immutable digest verification")
        target.chmod(0o440)
        return target, object_id

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".rfv-intake-", delete=False) as temp:
            temp_name = temp.name
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, temp, length=1024 * 1024)
            temp.flush()
            os.fsync(temp.fileno())
        temp_path = Path(temp_name)
        if sha256_file(temp_path) != source_sha256:
            raise IntakeError("copied object digest mismatch")
        temp_path.chmod(0o440)
        os.replace(temp_path, target)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    if sha256_file(target) != source_sha256:
        raise IntakeError("stored object digest mismatch")
    return target, object_id


def build_qualification_record(
    *,
    source: Path,
    consent: Path,
    case_id: str,
    category: str,
    license_name: str,
    storage_root: Path,
    record_out: Path,
    privacy_review: str,
    confirm_no_public_pii: bool,
    existing_records: Path | None = None,
) -> dict[str, Any]:
    intake_policy = load_json(INTAKE_POLICY_PATH)
    qualification_policy = load_json(QUALIFICATION_POLICY_PATH)
    tool_policy = load_json(TOOL_POLICY_PATH)
    if tool_policy.get("schema") != "vektoryum-rfv2-secure-intake-v1":
        raise IntakeError("invalid secure-intake policy")
    if qualification_policy.get("required_split") != "qualification":
        raise IntakeError("qualification policy drift")
    if not CASE_ID_RE.fullmatch(case_id):
        raise IntakeError("invalid case_id")
    if category not in intake_policy["categories"]:
        raise IntakeError("unknown category")
    if license_name not in intake_policy["allowed_licenses"]:
        raise IntakeError("unreviewed license")
    if privacy_review != "approved":
        raise IntakeError("privacy review must be approved")
    if not confirm_no_public_pii:
        raise IntakeError("explicit no-public-PII confirmation is required")

    source = require_regular_input(source, "source")
    consent = require_regular_input(consent, "consent or license proof")
    storage_root = require_external_path(storage_root, "storage root")
    record_out = require_external_path(record_out, "record output")
    if record_out.exists() and record_out.is_symlink():
        raise IntakeError("record output symlinks are forbidden")

    inspection = inspect_image(source, intake_policy)
    source_sha256 = sha256_file(source)
    consent_sha256 = sha256_file(consent)
    _, object_id = _copy_content_addressed(source, storage_root, source_sha256, inspection["source_format"])

    inspection_payload = {
        "schema": tool_policy["inspection_schema"],
        "case_id": case_id,
        "category": category,
        "source_sha256": source_sha256,
        "consent_sha256": consent_sha256,
        **inspection,
    }
    inspection_bytes = json.dumps(inspection_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record = {
        "schema": tool_policy["record_schema"],
        "case_id": case_id,
        "category": category,
        "split": "qualification",
        "source_sha256": source_sha256,
        "consent_sha256": consent_sha256,
        "inspection_sha256": hashlib.sha256(inspection_bytes).hexdigest(),
        "license": license_name,
        **inspection,
        "storage_object_id": object_id,
        "privacy_review": "approved",
        "contains_public_pii": False,
        "source_verified": True,
        "consent_verified": True,
        "object_immutable": True,
        "decode_verified": True,
        "inspection": inspection_payload,
    }
    _reject_duplicates(record, _load_existing_records(existing_records))
    return record


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".rfv-record-", delete=False) as temp:
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
    parser = argparse.ArgumentParser(description="Register one reviewed RFV-2 qualification image offline.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--consent", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--license", dest="license_name", required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--record-out", type=Path, required=True)
    parser.add_argument("--privacy-review", choices=["approved"], required=True)
    parser.add_argument("--confirm-no-public-pii", action="store_true")
    parser.add_argument("--existing-records", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = build_qualification_record(
            source=args.source,
            consent=args.consent,
            case_id=args.case_id,
            category=args.category,
            license_name=args.license_name,
            storage_root=args.storage_root,
            record_out=args.record_out,
            privacy_review=args.privacy_review,
            confirm_no_public_pii=args.confirm_no_public_pii,
            existing_records=args.existing_records,
        )
        write_json_atomic(require_external_path(args.record_out, "record output"), record)
    except IntakeError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "registered", "case_id": record["case_id"], "storage_object_id": record["storage_object_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
