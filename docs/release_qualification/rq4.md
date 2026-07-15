# RQ-4 — Beta release gate

RQ-4 is the final finite, fail-closed release gate. It records the beta version, release notes, rollback target, required secret names, user acceptance evidence, mandatory CI evidence, exact deployed main SHA and live health evidence.

The gate may report `approved` only when every required evidence flag is true, the candidate and deployed revisions are identical 40-character lowercase hexadecimal SHAs, the rollback target is a different qualified SHA, and no secret value is stored in the manifest.

Any missing or stale evidence, revision mismatch, non-green mandatory workflow, absent rollback target, secret value leakage or unhealthy live probe blocks release.
