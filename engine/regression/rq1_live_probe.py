"""Fail-closed RQ-1 live health and revision probe."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_MODES = {"beta", "live", "maintenance"}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


def _get_json(base_url: str, path: str) -> dict:
    target = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = urllib.request.Request(target, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}")
            if _origin(response.geturl()) != _origin(base_url):
                raise RuntimeError(f"{path} redirected to an unexpected origin")
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path} probe failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} did not return a JSON object")
    return payload


def verify(base_url: str, expected_sha: str) -> None:
    if not base_url.startswith("https://"):
        raise RuntimeError("RQ-1 production endpoint must use HTTPS")
    if len(expected_sha) != 40 or any(ch not in "0123456789abcdef" for ch in expected_sha.lower()):
        raise RuntimeError("expected SHA must be a full 40-character Git commit SHA")

    live = _get_json(base_url, "/livez")
    ready = _get_json(base_url, "/readyz")

    if live.get("status") != "ok" or live.get("check") != "liveness":
        raise RuntimeError("invalid liveness contract")
    if ready.get("status") != "ready" or ready.get("check") != "readiness":
        raise RuntimeError("invalid readiness contract")
    for name, payload in (("liveness", live), ("readiness", ready)):
        if payload.get("mode") not in ALLOWED_MODES:
            raise RuntimeError(f"{name} returned an unknown service mode")
        if payload.get("revision") != expected_sha:
            raise RuntimeError(f"{name} revision does not match expected main SHA")
    if ready.get("reasons") != []:
        raise RuntimeError("readiness reported blocking reasons")
    active = ready.get("active_requests")
    if not isinstance(active, int) or active < 0:
        raise RuntimeError("readiness active_requests is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    verify(args.base_url, args.expected_sha.lower())
    print(json.dumps({"rq": "RQ-1", "status": "health_verified", "revision": args.expected_sha.lower()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
