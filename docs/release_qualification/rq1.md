# RQ-1 — Live environment and end-to-end release verification

RQ-1 is accepted only when an exact expected `main` SHA is supplied and every required probe completes with real assertions.

## Required probes

1. `GET /livez` returns HTTP 200 JSON with `status=ok`, `check=liveness`, an allowed service mode, and `revision` exactly equal to the expected main SHA.
2. `GET /readyz` returns HTTP 200 JSON with `status=ready`, `check=readiness`, the same revision, an empty reasons list, and a non-negative active-request count.
3. Beta and live modes accept the documented authenticated write flow; maintenance rejects writes with HTTP 503 while health remains observable.
4. Login, authenticated upload, terminal job state and artifact download complete without weakening same-origin request validation.
5. A controlled restart preserves durable account/job state and immutable artifact manifest identity.

## Fail-closed rules

The qualification command fails for missing endpoint or expected SHA, redirects to an unexpected origin, non-JSON health responses, unknown mode, revision mismatch, readiness degradation, missing credentials, stale or mutable artifact metadata, incomplete jobs, or state loss after restart.

Live secrets and destructive restart operations are never embedded in the repository. CI validates the deterministic contract with a local HTTP fixture; the production probe is explicitly parameterized by environment secrets and exact deployed SHA.
