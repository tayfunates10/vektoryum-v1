# RFV-2E — Live public-source acquisition evidence

RFV-2E executes the reviewed RFV-2D allowlist on a GitHub-hosted runner with real network access. It does not use generated fixtures as real corpus evidence.

## Live evidence flow

1. Download the exact 24 reviewed CC0 or public-domain sources.
2. Snapshot and hash source pages, license pages, provider metadata and downloaded bytes.
3. Decode and canonicalize each source under its bounded acquisition profile.
4. Register every canonical source through the RFV-2B secure intake.
5. Assemble exactly 24 records through the RFV-2C complete qualification gate.
6. Verify all ten difficult-image categories, independent evidence digests, duplicate absence, decoding and privacy state.
7. Build a deterministic tar.gz containing content-addressed objects, individual records, public license proofs, the qualification manifest, the qualification audit and the reviewed source-selection manifest.
8. Upload the bundle and sanitized evidence as an immutable GitHub Actions artifact with a 90-day retention period.
9. Bind the artifact ID, URL and SHA-256 artifact digest to a separate publication envelope.

## Repository boundary

Raw and canonical image bytes are written only below the runner temporary directory and uploaded as workflow artifacts. They are never committed to the application repository tree.

The qualification and publication artifacts are cryptographically bound by:

- canonical source SHA-256 per case;
- independent source-page and license-proof SHA-256 values;
- deterministic inspection SHA-256 per case;
- qualification `cases_sha256`;
- deterministic corpus bundle SHA-256;
- GitHub Actions artifact digest and artifact ID.

## Fail-closed rules

The workflow fails when any source cannot be retrieved, redirects to a non-allowlisted host, exceeds a size budget, cannot be decoded, lacks required alpha, is too small for the non-upscaled 4K profile, has incomplete category coverage, duplicates another identity, fails evidence hashing, produces fewer or more than 24 records, or cannot be uploaded as an immutable artifact.

## Progress boundary

RFV-2 remains `pending` in this preparation PR. It may advance only after the live workflow succeeds, both artifacts are inspected, the publication envelope is captured in the repository and the complete qualification manifest is committed without raw source bytes.
