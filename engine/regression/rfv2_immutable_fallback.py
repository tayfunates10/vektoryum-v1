from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = ROOT / "engine" / "regression" / "rfv2_qualification_manifest.json"
ARTIFACT_PREFIX = "rfv2-public-qualification-corpus-"
REQUIRED_FILES = {
    "rfv2-public-corpus.tar.gz",
    "qualification-manifest.json",
    "qualification-audit.json",
    "bundle-checksums.json",
}
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


class ImmutableCorpusError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _identity_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ImmutableCorpusError("qualification manifest cases are invalid")
    identities: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not all(key in case for key in IDENTITY_KEYS):
            raise ImmutableCorpusError("qualification manifest case identity is incomplete")
        case_id = str(case["case_id"])
        if case_id in identities:
            raise ImmutableCorpusError("qualification manifest contains duplicate cases")
        identities[case_id] = {key: case[key] for key in IDENTITY_KEYS}
    return identities


def committed_source_identity(path: Path = COMMITTED_MANIFEST) -> dict[str, dict[str, Any]]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImmutableCorpusError("committed RFV-2 manifest is unreadable") from exc
    identities = _identity_map(manifest)
    if len(identities) != 24:
        raise ImmutableCorpusError("committed RFV-2 qualification identity is incomplete")
    return identities


def request_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ImmutableCorpusError("GitHub artifact listing is invalid")
    return payload


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


def request_bytes(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        location = exc.headers.get("Location")
        if not location or not location.startswith("https://"):
            raise ImmutableCorpusError("artifact redirect location rejected") from exc
    else:
        with response:
            return response.read()

    # The repository token is intentionally not forwarded to the temporary object
    # storage host. GitHub's signed redirect URL is the complete authorization.
    with urllib.request.urlopen(location, timeout=120) as response:
        return response.read()


def _safe_archive_names(archive: zipfile.ZipFile) -> set[str]:
    names = set(archive.namelist())
    safe = {
        name
        for name in names
        if not PurePosixPath(name).is_absolute()
        and ".." not in PurePosixPath(name).parts
        and not name.endswith("/")
    }
    if names != safe:
        raise ImmutableCorpusError("artifact archive layout rejected")
    if not REQUIRED_FILES.issubset(names):
        raise ImmutableCorpusError("artifact archive is incomplete")
    return names


def _clean_outputs(out_dir: Path) -> None:
    for name in REQUIRED_FILES | {"fallback-provenance.json"}:
        (Path(out_dir) / name).unlink(missing_ok=True)


def extract_and_verify_artifact(
    payload: bytes,
    artifact: dict[str, Any],
    out_dir: Path,
    expected_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_outputs(out_dir)

    advertised = str(artifact.get("digest") or "")
    if advertised and advertised.removeprefix("sha256:") != sha256_bytes(payload):
        raise ImmutableCorpusError("artifact archive digest mismatch")

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            _safe_archive_names(archive)
            for name in REQUIRED_FILES:
                target = out_dir / name
                with archive.open(name) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        _clean_outputs(out_dir)
        raise ImmutableCorpusError("artifact archive could not be extracted") from exc

    try:
        manifest = json.loads(
            (out_dir / "qualification-manifest.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (out_dir / "qualification-audit.json").read_text(encoding="utf-8")
        )
        checksums = json.loads(
            (out_dir / "bundle-checksums.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        _clean_outputs(out_dir)
        raise ImmutableCorpusError("cached corpus metadata is invalid") from exc

    actual_cases = _identity_map(manifest)
    if actual_cases != expected_cases:
        raise ImmutableCorpusError("cached corpus source identity drift")
    if manifest.get("status") != "qualified" or manifest.get("qualified_case_count") != 24:
        raise ImmutableCorpusError("cached corpus is not a complete qualification corpus")
    if manifest.get("cases_sha256") != checksums.get("cases_sha256"):
        raise ImmutableCorpusError("cached corpus cases digest mismatch")
    if audit.get("complete") is not True or audit.get("missing_categories") != []:
        raise ImmutableCorpusError("cached corpus audit is incomplete")
    if sha256_path(out_dir / "rfv2-public-corpus.tar.gz") != checksums.get("bundle_sha256"):
        raise ImmutableCorpusError("cached corpus bundle digest mismatch")
    if sha256_path(out_dir / "qualification-manifest.json") != checksums.get(
        "qualification_manifest_sha256"
    ):
        raise ImmutableCorpusError("cached qualification manifest digest mismatch")
    if sha256_path(out_dir / "qualification-audit.json") != checksums.get(
        "qualification_audit_sha256"
    ):
        raise ImmutableCorpusError("cached qualification audit digest mismatch")
    if checksums.get("storage_mode") != "github_actions_immutable_artifact":
        raise ImmutableCorpusError("cached corpus storage mode rejected")
    if checksums.get("qualified_case_count") != 24:
        raise ImmutableCorpusError("cached corpus checksum case count drift")

    return {
        "manifest": manifest,
        "audit": audit,
        "checksums": checksums,
    }


def restore_latest_verified_corpus(
    *,
    out_dir: Path,
    repository: str,
    token: str,
    manifest_path: Path = COMMITTED_MANIFEST,
    json_fetcher: Callable[[str, str], dict[str, Any]] = request_json,
    bytes_fetcher: Callable[[str, str], bytes] = request_bytes,
) -> dict[str, Any]:
    if not repository or "/" not in repository:
        raise ImmutableCorpusError("GitHub repository identity is missing")
    if not token:
        raise ImmutableCorpusError("GitHub Actions token is missing")

    expected_cases = committed_source_identity(manifest_path)
    listing = json_fetcher(
        f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100",
        token,
    )
    artifacts = sorted(
        (
            artifact
            for artifact in listing.get("artifacts", [])
            if str(artifact.get("name", "")).startswith(ARTIFACT_PREFIX)
            and not artifact.get("expired", True)
        ),
        key=lambda artifact: str(artifact.get("created_at", "")),
        reverse=True,
    )

    failures: list[str] = []
    selected: dict[str, Any] | None = None
    verified: dict[str, Any] | None = None
    for artifact in artifacts:
        try:
            payload = bytes_fetcher(str(artifact["archive_download_url"]), token)
            verified = extract_and_verify_artifact(
                payload,
                artifact,
                out_dir,
                expected_cases,
            )
            selected = artifact
            break
        except Exception as exc:  # noqa: BLE001 - each immutable candidate is isolated
            failures.append(f"{artifact.get('id')}: {type(exc).__name__}: {exc}")
            _clean_outputs(out_dir)

    if selected is None or verified is None:
        raise ImmutableCorpusError(
            "no verified immutable RFV-2 corpus artifact available: "
            + " | ".join(failures[-5:])
        )

    checksums = verified["checksums"]
    provenance = {
        "schema": "vektoryum-rfv2-immutable-fallback-v1",
        "reason": "bounded_live_provider_attempts_exhausted",
        "source_artifact_id": int(selected["id"]),
        "source_artifact_name": selected["name"],
        "source_artifact_digest": selected.get("digest"),
        "source_created_at": selected.get("created_at"),
        "cases_sha256": checksums["cases_sha256"],
        "bundle_sha256": checksums["bundle_sha256"],
        "qualified_case_count": 24,
        "verification": "committed_source_identity_and_internal_sha256",
    }
    (Path(out_dir) / "fallback-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the latest immutable RFV-2 corpus matching committed identity."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    parser.add_argument("--manifest", type=Path, default=COMMITTED_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        provenance = restore_latest_verified_corpus(
            out_dir=args.out_dir,
            repository=args.repository,
            token=args.token,
            manifest_path=args.manifest,
        )
    except ImmutableCorpusError as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "restored", **provenance}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
