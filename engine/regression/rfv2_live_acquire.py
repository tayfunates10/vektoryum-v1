from __future__ import annotations

import argparse
import html
import json
import re
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from engine.regression import rfv2_public_source_acquire as public_acquire

FetchResult = tuple[bytes, str, str]
FetchFunction = Callable[[str, set[str], int], FetchResult]


class _AssetLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name.lower() in {
                "href",
                "src",
                "content",
                "data-src",
                "data-original",
                "data-download",
            }:
                self.values.append(value)


_URL_RE = re.compile(r"https?:\\?/\\?/[^\s\"'<>]+", re.IGNORECASE)


def _candidate_score(url: str) -> tuple[int, int, str]:
    lowered = url.lower()
    score = 0
    if lowered.endswith(".png") or ".png?" in lowered:
        score += 200
    if "/image/" in lowered:
        score += 120
    if "svg_to_png" in lowered:
        score += 80
    if "/download/" in lowered:
        score += 20
    for size, weight in (("2400px", 40), ("2000px", 35), ("1200px", 30), ("800px", 25), ("400px", 20)):
        if size in lowered:
            score += weight
            break
    return score, -len(url), url


def extract_openclipart_candidates(
    source_page: bytes,
    *,
    source_page_url: str,
    provider_asset_id: str,
    allowed_hosts: set[str],
) -> list[str]:
    try:
        text = source_page.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - bytes.decode is deterministic
        raise public_acquire.PublicSourceError("Openclipart source page could not be decoded") from exc

    parser = _AssetLinkParser()
    parser.feed(text)
    values = list(parser.values)
    values.extend(match.group(0) for match in _URL_RE.finditer(text))

    candidates: set[str] = set()
    for raw_value in values:
        value = html.unescape(raw_value).replace("\\/", "/").strip()
        if not value:
            continue
        resolved = urllib.parse.urljoin(source_page_url, value)
        parsed = urllib.parse.urlsplit(resolved)
        if parsed.scheme != "https" or not parsed.hostname:
            continue
        if parsed.hostname.lower() not in allowed_hosts or parsed.hostname.lower() != "openclipart.org":
            continue
        if parsed.username or parsed.password or parsed.fragment:
            continue
        path = parsed.path.lower()
        if provider_asset_id not in path and provider_asset_id not in parsed.query:
            continue
        if "/detail/" in path or path.rstrip("/").endswith(f"/detail/{provider_asset_id}"):
            continue
        if not any(token in path for token in ("/image/", "/download/", ".png")):
            continue
        candidates.add(urllib.parse.urlunsplit(parsed))

    return sorted(candidates, key=_candidate_score, reverse=True)


def resolve_openclipart_asset_url(
    case: dict[str, Any],
    manifest: dict[str, Any],
    *,
    fetcher: FetchFunction = public_acquire._fetch_url,
) -> str:
    allowed_hosts = {host.lower() for host in manifest["allowed_source_hosts"]}
    source_page, source_page_final, _ = fetcher(case["source_page_url"], allowed_hosts, 5 * 1024 * 1024)
    candidates = extract_openclipart_candidates(
        source_page,
        source_page_url=source_page_final,
        provider_asset_id=case["provider_asset_id"],
        allowed_hosts=allowed_hosts,
    )

    manifest_asset = case.get("asset_url")
    if isinstance(manifest_asset, str) and manifest_asset not in candidates:
        candidates.append(manifest_asset)

    last_reason = "no candidate URL found on the reviewed source page"
    for candidate in candidates:
        try:
            payload, final_url, _ = fetcher(candidate, allowed_hosts, public_acquire.MAX_HTTP_BYTES)
            public_acquire.canonicalize_image(payload, case["acquisition_profile"])
            return final_url
        except public_acquire.PublicSourceError as exc:
            last_reason = str(exc)

    raise public_acquire.PublicSourceError(
        f"Openclipart asset resolution failed for {case['case_id']}: {last_reason}"
    )


def prepare_live_provider_case(case: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(case)
    if prepared["provider"] != "library_of_congress":
        return prepared

    allowed_hosts = {host.lower() for host in manifest["allowed_source_hosts"]}
    metadata_url = public_acquire._validated_https_url(
        prepared.get("metadata_url"), allowed_hosts, "LOC metadata proof"
    )
    parsed = urllib.parse.urlsplit(metadata_url)
    if parsed.hostname != "www.loc.gov" or parsed.query != "fo=json":
        raise public_acquire.PublicSourceError("LOC metadata proof URL mismatch")
    if prepared.get("rights_statement") != "No known restrictions on publication.":
        raise public_acquire.PublicSourceError("LOC rights statement mismatch")

    # LOC HTML item pages reject automated clients. The official JSON item
    # representation contains the same catalog identity, rights metadata and
    # raster links, so it is used as both the machine-readable source snapshot
    # and public-domain proof while the original item URL remains in the
    # reviewed source-selection manifest.
    prepared["source_page_url"] = metadata_url
    prepared["license_proof_url"] = metadata_url
    return prepared


def acquire_selected(
    *,
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
    download_root: Path,
    storage_root: Path,
    records_dir: Path,
) -> list[dict[str, Any]]:
    acquired: list[dict[str, Any]] = []
    for original in cases:
        case = prepare_live_provider_case(original, manifest)
        if case["provider"] == "openclipart":
            case["asset_url"] = resolve_openclipart_asset_url(case, manifest)
        acquired.append(
            public_acquire.acquire_case(
                case,
                download_root=download_root,
                storage_root=storage_root,
                records_dir=records_dir,
                manifest=manifest,
            )
        )
    return acquired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire the reviewed RFV-2 public allowlist with live provider URL resolution."
    )
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
        manifest = public_acquire.load_json(public_acquire.MANIFEST_PATH)
        cases = public_acquire.validate_manifest(manifest)
        selected = cases if args.all else [case for case in cases if case["case_id"] == args.case_id]
        if not selected:
            raise public_acquire.PublicSourceError("requested case is not in the reviewed allowlist")
        acquired = acquire_selected(
            cases=selected,
            manifest=manifest,
            download_root=args.download_root,
            storage_root=args.storage_root,
            records_dir=args.records_dir,
        )
    except (public_acquire.PublicSourceError, RuntimeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "acquired",
                "count": len(acquired),
                "case_ids": [item["case_id"] for item in acquired],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
