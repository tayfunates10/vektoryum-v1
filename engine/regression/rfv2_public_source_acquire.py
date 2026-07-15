from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from engine.regression.rfv2_secure_intake import (
    ROOT,
    build_qualification_record,
    require_external_path,
    write_json_atomic,
)

MANIFEST_PATH = ROOT / "engine" / "regression" / "rfv2_public_source_manifest.json"
INTAKE_POLICY_PATH = ROOT / "engine" / "regression" / "rfv1_intake_policy.json"
MAX_HTTP_BYTES = 200 * 1024 * 1024
USER_AGENT = "Vektoryum-RFV2-Public-Acquirer/1.0"


class PublicSourceError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSourceError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise PublicSourceError(f"JSON root must be an object: {path}")
    return payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_https_url(url: Any, allowed_hosts: set[str], label: str) -> str:
    if not isinstance(url, str) or not url:
        raise PublicSourceError(f"missing {label}")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PublicSourceError(f"{label} must use HTTPS")
    if parsed.username or parsed.password or parsed.fragment:
        raise PublicSourceError(f"{label} contains forbidden URL components")
    if parsed.hostname.lower() not in allowed_hosts:
        raise PublicSourceError(f"{label} host is not allowlisted")
    return url


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    intake_policy = load_json(INTAKE_POLICY_PATH)
    if manifest.get("schema") != "vektoryum-rfv2-public-source-manifest-v1":
        raise PublicSourceError("public-source schema drift")
    if manifest.get("status") != "selected_not_acquired":
        raise PublicSourceError("source selection must not claim acquisition")
    if manifest.get("public_repo_contains_raw_assets") is not False:
        raise PublicSourceError("raw public assets must remain outside the repository")

    expected_count = manifest.get("expected_case_count")
    cases = manifest.get("cases")
    if expected_count != intake_policy["splits"]["qualification"] or not isinstance(cases, list):
        raise PublicSourceError("qualification source count drift")
    if len(cases) != expected_count:
        raise PublicSourceError("exactly 24 selected sources are required")

    allowed_hosts = manifest.get("allowed_source_hosts")
    allowed_licenses = manifest.get("allowed_license_classes")
    targets = manifest.get("category_targets")
    if not isinstance(allowed_hosts, list) or len(allowed_hosts) != len(set(allowed_hosts)):
        raise PublicSourceError("invalid source host allowlist")
    if not isinstance(allowed_licenses, list) or set(allowed_licenses) != {"cc0", "public-domain"}:
        raise PublicSourceError("license allowlist drift")
    if not isinstance(targets, dict) or targets != {
        "flat_logo": 3,
        "badge_seal": 2,
        "small_text": 3,
        "monoline": 2,
        "multicolor": 2,
        "low_resolution_signage_photo": 3,
        "gradient_artwork": 2,
        "native_4k": 2,
        "transparent_dark_background": 2,
        "complex_illustration": 3,
    }:
        raise PublicSourceError("finite category target drift")

    allowed_host_set = {host.lower() for host in allowed_hosts}
    identities: dict[str, list[str]] = {
        "case_id": [],
        "provider_identity": [],
        "source_page_url": [],
    }
    category_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []

    for case in cases:
        if not isinstance(case, dict):
            raise PublicSourceError("source entries must be objects")
        case_id = case.get("case_id")
        category = case.get("category")
        provider = case.get("provider")
        provider_asset_id = case.get("provider_asset_id")
        license_name = case.get("license")
        profile = case.get("acquisition_profile")
        if not isinstance(case_id, str) or not case_id.startswith("qualification-public-"):
            raise PublicSourceError("invalid public source case_id")
        if category not in targets or category not in intake_policy["categories"]:
            raise PublicSourceError("unknown public source category")
        if not isinstance(provider_asset_id, str) or not provider_asset_id.isdigit():
            raise PublicSourceError("provider asset identifiers must be numeric strings")
        if license_name not in allowed_licenses or license_name not in intake_policy["allowed_licenses"]:
            raise PublicSourceError("unapproved public source license")

        source_page_url = _validated_https_url(case.get("source_page_url"), allowed_host_set, "source page")
        license_proof_url = _validated_https_url(case.get("license_proof_url"), allowed_host_set, "license proof")

        if provider == "openclipart":
            if license_name != "cc0" or profile not in {"openclipart_png", "openclipart_transparent_png"}:
                raise PublicSourceError("Openclipart source policy mismatch")
            asset_url = _validated_https_url(case.get("asset_url"), allowed_host_set, "asset")
            if urllib.parse.urlsplit(asset_url).hostname != "openclipart.org":
                raise PublicSourceError("Openclipart asset host mismatch")
            if case.get("metadata_url") is not None or case.get("rights_statement") is not None:
                raise PublicSourceError("Openclipart entries contain unapproved fields")
        elif provider == "library_of_congress":
            if license_name != "public-domain":
                raise PublicSourceError("LOC source must be public-domain")
            if profile not in {"loc_low_resolution_signage_photo", "loc_public_domain_4k_crop"}:
                raise PublicSourceError("LOC acquisition profile mismatch")
            metadata_url = _validated_https_url(case.get("metadata_url"), allowed_host_set, "metadata")
            if case.get("rights_statement") != "No known restrictions on publication.":
                raise PublicSourceError("LOC rights statement mismatch")
            if case.get("asset_url") is not None:
                raise PublicSourceError("LOC assets must be resolved from official metadata")
            if urllib.parse.urlsplit(metadata_url).hostname != "www.loc.gov":
                raise PublicSourceError("LOC metadata host mismatch")
        else:
            raise PublicSourceError("unknown public source provider")

        identities["case_id"].append(case_id)
        identities["provider_identity"].append(f"{provider}:{provider_asset_id}")
        identities["source_page_url"].append(source_page_url)
        category_counts[category] += 1
        normalized.append(dict(case, license_proof_url=license_proof_url))

    for label, values in identities.items():
        if len(values) != len(set(values)):
            raise PublicSourceError(f"duplicate {label}")
    if dict(category_counts) != targets:
        raise PublicSourceError("selected source category distribution mismatch")
    return normalized


def _fetch_url(url: str, allowed_hosts: set[str], max_bytes: int = MAX_HTTP_BYTES) -> tuple[bytes, str, str]:
    _validated_https_url(url, allowed_hosts, "network URL")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            final_url = response.geturl()
            _validated_https_url(final_url, allowed_hosts, "redirect target")
            content_type = response.headers.get_content_type()
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise PublicSourceError("network response exceeds size budget")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise PublicSourceError("network response exceeds size budget")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise PublicSourceError(f"failed to fetch approved source: {url}") from exc
    return b"".join(chunks), final_url, content_type


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def select_loc_asset_url(metadata: dict[str, Any], allowed_hosts: set[str]) -> str:
    candidates: list[tuple[int, str]] = []
    for value in _iter_strings(metadata):
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.hostname.lower() not in allowed_hosts:
            continue
        path = parsed.path.lower()
        score = 0
        if path.endswith((".tif", ".tiff")):
            score = 50
        elif path.endswith((".jpg", ".jpeg")):
            score = 40
        elif path.endswith(".png"):
            score = 30
        else:
            continue
        lowered = value.lower()
        if "original" in lowered:
            score += 20
        if "master" in lowered:
            score += 15
        if "small" in lowered or "thumb" in lowered:
            score -= 25
        candidates.append((score, value))
    if not candidates:
        raise PublicSourceError("LOC metadata contains no allowlisted raster asset")
    return max(candidates, key=lambda item: (item[0], len(item[1])))[1]


def _open_verified_image(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception as exc:
        raise PublicSourceError("downloaded asset is not a decodable image") from exc
    return image


def canonicalize_image(payload: bytes, profile: str) -> tuple[bytes, str, dict[str, Any]]:
    image = _open_verified_image(payload)
    if profile in {"openclipart_png", "openclipart_transparent_png"}:
        image = image.convert("RGBA")
        if profile == "openclipart_transparent_png":
            alpha = image.getchannel("A")
            low, high = alpha.getextrema()
            if low == high == 255:
                raise PublicSourceError("transparent profile requires real alpha transparency")
        if image.width * image.height > 8_294_400:
            image.thumbnail((2880, 2880), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        result = output.getvalue()
        return result, "png", {"width": image.width, "height": image.height, "transform": "decode_and_lossless_png"}

    image = ImageOps.exif_transpose(image).convert("RGB")
    if profile == "loc_low_resolution_signage_photo":
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True, progressive=True)
        result = output.getvalue()
        return result, "jpeg", {"width": image.width, "height": image.height, "transform": "bounded_low_resolution_photo"}

    if profile == "loc_public_domain_4k_crop":
        target_width, target_height = 3840, 2160
        if image.width < target_width or image.height < target_height:
            raise PublicSourceError("LOC source is too small for a non-upscaled 4K crop")
        target_ratio = target_width / target_height
        current_ratio = image.width / image.height
        if current_ratio > target_ratio:
            crop_width = round(image.height * target_ratio)
            left = (image.width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, image.height))
        else:
            crop_height = round(image.width / target_ratio)
            top = (image.height - crop_height) // 2
            image = image.crop((0, top, image.width, top + crop_height))
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True, progressive=True)
        result = output.getvalue()
        if len(result) > 26_214_400:
            raise PublicSourceError("canonical 4K JPEG exceeds intake file budget")
        return result, "jpeg", {"width": target_width, "height": target_height, "transform": "center_crop_16x9_non_upscaled_4k"}

    raise PublicSourceError("unknown acquisition profile")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".rfv-public-", delete=False) as temp:
            temp_name = temp.name
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def acquire_case(
    case: dict[str, Any],
    *,
    download_root: Path,
    storage_root: Path,
    records_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    allowed_hosts = {host.lower() for host in manifest["allowed_source_hosts"]}
    download_root = require_external_path(download_root, "public download root")
    storage_root = require_external_path(storage_root, "storage root")
    records_dir = require_external_path(records_dir, "records directory")
    case_dir = download_root / case["case_id"]
    if case_dir.exists() and case_dir.is_symlink():
        raise PublicSourceError("case directory symlinks are forbidden")
    case_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    source_page, source_page_final, source_page_type = _fetch_url(case["source_page_url"], allowed_hosts, 5 * 1024 * 1024)
    license_page, license_page_final, license_page_type = _fetch_url(case["license_proof_url"], allowed_hosts, 5 * 1024 * 1024)
    metadata_payload: bytes | None = None
    metadata_final: str | None = None

    if case["provider"] == "openclipart":
        raw_asset, raw_final, raw_type = _fetch_url(case["asset_url"], allowed_hosts)
    else:
        metadata_payload, metadata_final, _ = _fetch_url(case["metadata_url"], allowed_hosts, 10 * 1024 * 1024)
        try:
            metadata = json.loads(metadata_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicSourceError("LOC metadata is not valid JSON") from exc
        asset_url = select_loc_asset_url(metadata, allowed_hosts)
        raw_asset, raw_final, raw_type = _fetch_url(asset_url, allowed_hosts)

    canonical, extension, transform = canonicalize_image(raw_asset, case["acquisition_profile"])
    source_path = case_dir / f"source.{extension}"
    proof_path = case_dir / "license-proof.json"
    _write_bytes_atomic(source_path, canonical)
    proof = {
        "schema": "vektoryum-rfv2-public-license-proof-v1",
        "case_id": case["case_id"],
        "provider": case["provider"],
        "provider_asset_id": case["provider_asset_id"],
        "license": case["license"],
        "source_page_url": source_page_final,
        "source_page_content_type": source_page_type,
        "source_page_sha256": sha256_bytes(source_page),
        "license_proof_url": license_page_final,
        "license_proof_content_type": license_page_type,
        "license_proof_sha256": sha256_bytes(license_page),
        "metadata_url": metadata_final,
        "metadata_sha256": sha256_bytes(metadata_payload) if metadata_payload is not None else None,
        "asset_url": raw_final,
        "asset_content_type": raw_type,
        "downloaded_asset_sha256": sha256_bytes(raw_asset),
        "canonical_source_sha256": sha256_bytes(canonical),
        "canonicalization": transform,
        "rights_statement": case.get("rights_statement", "CC0 1.0 public-domain dedication"),
    }
    write_json_atomic(proof_path, proof)

    existing_records = None
    existing_files = sorted(records_dir.glob("*.json"))
    if existing_files:
        existing_manifest = records_dir / ".existing-records.json"
        cases = [load_json(path) for path in existing_files if path.name != existing_manifest.name]
        write_json_atomic(existing_manifest, {"cases": cases})
        existing_records = existing_manifest
    record_path = records_dir / f"{case['case_id']}.json"
    record = build_qualification_record(
        source=source_path,
        consent=proof_path,
        case_id=case["case_id"],
        category=case["category"],
        license_name=case["license"],
        storage_root=storage_root,
        record_out=record_path,
        privacy_review="approved",
        confirm_no_public_pii=True,
        existing_records=existing_records,
    )
    write_json_atomic(record_path, record)
    if existing_records:
        existing_records.unlink(missing_ok=True)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire only the reviewed RFV-2 public-source allowlist.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_json(MANIFEST_PATH)
        cases = validate_manifest(manifest)
        selected = cases if args.all else [case for case in cases if case["case_id"] == args.case_id]
        if not selected:
            raise PublicSourceError("requested case is not in the reviewed allowlist")
        acquired = [
            acquire_case(
                case,
                download_root=args.download_root,
                storage_root=args.storage_root,
                records_dir=args.records_dir,
                manifest=manifest,
            )
            for case in selected
        ]
    except (PublicSourceError, RuntimeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "acquired", "count": len(acquired), "case_ids": [item["case_id"] for item in acquired]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
