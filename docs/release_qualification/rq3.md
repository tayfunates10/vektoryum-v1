# RQ-3 — Load and resilience qualification

RQ-3 is finite and fail-closed. It qualifies bounded admission, quota enforcement, disk-budget cleanup, cancellation, timeout, restart recovery and persistence-conflict observability. It does not claim RQ-4 completion.

## Mandatory scenarios

1. Admission: 2 active + 3 queued are accepted; the sixth request is rejected before work allocation.
2. Per-user quota: a user at the configured retained-byte limit cannot create another owned artifact.
3. Disk budget: cleanup deletes only oldest eligible terminal artifacts until the low-water mark is reached.
4. Preservation: active, pinned and legal-hold artifacts are never cleanup candidates.
5. Cancellation: cancellation is terminal, idempotent and removes temporary output while retaining the immutable event record.
6. Timeout: timed-out work cannot publish a reusable artifact manifest.
7. Restart: durable queued/running jobs are recovered exactly once; stale leases become recoverable and fresh leases remain owned.
8. Conflict: remote persistence generation mismatch is observable and fails closed instead of silently overwriting state.
9. Corruption: invalid or partial artifact manifests are quarantined and cannot be downloaded.

## Release rule

Every scenario must have a named executable assertion in the mandatory RQ-3 workflow. Missing limits, negative limits, non-terminal cleanup targets, orphaned temporary files, silent remote overwrite or a non-prefix roadmap state fail the phase.
