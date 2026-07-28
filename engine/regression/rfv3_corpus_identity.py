from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CASE_COUNT = 24
EXPECTED_STABLE_CASES_SHA256 = (
    "277562f698d3de0feae8ed425a5bb9f21683334c622d5bcb5f1835a4b353d181"
)
STABLE_CASE_FIELDS = (
    "case_id",
    "category",
    "source_sha256",
    "license",
    "source_format",
    "width",
    "height",
    "file_bytes",
    "storage_object_id",
    "privacy_review",
    "contains_public_pii",
    "decode_verified",
    "consent_verified",
    "object_immutable",
    "split",
)


class CorpusIdentityError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusIdentityError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise CorpusIdentityError(f"JSON root must be an object: {path}")
    return payload


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_case_projection(
    manifest: dict[str, Any],
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> list[dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        raise CorpusIdentityError("reviewed corpus must contain the exact case count")

    projected: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise CorpusIdentityError("manifest cases must be objects")
        missing = [field for field in STABLE_CASE_FIELDS if field not in case]
        if missing:
            raise CorpusIdentityError(
                "stable corpus identity fields missing: " + ",".join(missing)
            )
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise CorpusIdentityError("invalid stable corpus case_id")
        if case_id in seen_case_ids:
            raise CorpusIdentityError("duplicate stable corpus case_id")
        seen_case_ids.add(case_id)
        projected.append({field: case[field] for field in STABLE_CASE_FIELDS})

    projected.sort(key=lambda item: item["case_id"])
    return projected


def stable_cases_sha256(
    manifest: dict[str, Any],
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> str:
    return canonical_sha256(
        stable_case_projection(
            manifest,
            expected_case_count=expected_case_count,
        )
    )


def validate_reviewed_corpus_identity(
    *,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    checksums: dict[str, Any],
    bundle: Path,
    expected_case_count: int = EXPECTED_CASE_COUNT,
    expected_stable_digest: str = EXPECTED_STABLE_CASES_SHA256,
) -> str:
    if manifest.get("schema") != "vektoryum-rfv2-qualification-manifest-v1":
        raise CorpusIdentityError("qualification manifest schema drift")
    if manifest.get("status") != "qualified":
        raise CorpusIdentityError("qualification manifest is not qualified")
    if manifest.get("qualified_case_count") != expected_case_count:
        raise CorpusIdentityError("qualification manifest case-count drift")
    if manifest.get("public_repo_contains_raw_assets") is not False:
        raise CorpusIdentityError("raw corpus assets must remain outside the repository")

    if audit.get("schema") != "vektoryum-rfv2-assembly-audit-v1":
        raise CorpusIdentityError("qualification audit schema drift")
    if audit.get("complete") is not True:
        raise CorpusIdentityError("qualification audit is incomplete")
    if audit.get("qualified_case_count") != expected_case_count:
        raise CorpusIdentityError("qualification audit case-count drift")
    if audit.get("missing_categories") != []:
        raise CorpusIdentityError("qualification audit has missing categories")

    if checksums.get("schema") != "vektoryum-rfv2-live-bundle-checksums-v1":
        raise CorpusIdentityError("bundle checksum schema drift")
    if checksums.get("qualified_case_count") != expected_case_count:
        raise CorpusIdentityError("bundle checksum case-count drift")
    if checksums.get("storage_mode") != "github_actions_immutable_artifact":
        raise CorpusIdentityError("bundle storage mode drift")
    if checksums.get("retention_days") != 90:
        raise CorpusIdentityError("bundle retention policy drift")

    cases_sha256 = checksums.get("cases_sha256")
    if not isinstance(cases_sha256, str) or not cases_sha256:
        raise CorpusIdentityError("bundle cases digest missing")
    if manifest.get("cases_sha256") != cases_sha256:
        raise CorpusIdentityError("manifest and bundle cases digests differ")
    if audit.get("cases_sha256") != cases_sha256:
        raise CorpusIdentityError("audit and bundle cases digests differ")

    try:
        bundle_bytes = bundle.read_bytes()
    except OSError as exc:
        raise CorpusIdentityError(f"unable to read corpus bundle: {bundle}") from exc
    if len(bundle_bytes) != checksums.get("bundle_bytes"):
        raise CorpusIdentityError("corpus bundle byte count mismatch")
    if hashlib.sha256(bundle_bytes).hexdigest() != checksums.get("bundle_sha256"):
        raise CorpusIdentityError("corpus bundle digest mismatch")

    stable_digest = stable_cases_sha256(
        manifest,
        expected_case_count=expected_case_count,
    )
    if stable_digest != expected_stable_digest:
        raise CorpusIdentityError(
            "stable reviewed corpus identity mismatch: "
            f"{stable_digest} != {expected_stable_digest}"
        )
    return stable_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an RFV-3 live corpus without hard-pinning volatile provider "
            "HTML proof digests."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stable_digest = validate_reviewed_corpus_identity(
            manifest=load_json(args.manifest),
            audit=load_json(args.audit),
            checksums=load_json(args.checksums),
            bundle=args.bundle,
        )
    except CorpusIdentityError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "validated",
                "qualified_case_count": EXPECTED_CASE_COUNT,
                "stable_cases_sha256": stable_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
