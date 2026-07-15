# RFV-2 — Qualified real-world corpus

RFV-2 records the finite qualification corpus produced by the merged RFV-2E live acquisition workflow. It does not claim universal 99% visual equivalence and it does not publish raw source images in the application repository.

## Qualified corpus

The corpus contains exactly 24 independently registered public sources across all ten required difficult-image categories:

| Category | Cases |
|---|---:|
| flat_logo | 3 |
| badge_seal | 2 |
| small_text | 3 |
| monoline | 2 |
| multicolor | 2 |
| low_resolution_signage_photo | 3 |
| gradient_artwork | 2 |
| native_4k | 2 |
| transparent_dark_background | 2 |
| complex_illustration | 3 |

Nineteen cases use reviewed Openclipart CC0 graphics. Five cases use reviewed Library of Congress public-domain photographs. Every case passed format decoding, bounded size and pixel checks, source and license/consent hashing, privacy review, immutable object registration and duplicate detection.

## Sanitized repository evidence

- Qualification manifest: `engine/regression/rfv2_qualification_manifest.json`
- Qualification audit: `docs/real_world_fidelity/evidence/rfv2_qualification_audit.json`
- Bundle checksums: `docs/real_world_fidelity/evidence/rfv2_bundle_checksums.json`
- Artifact publication envelope: `docs/real_world_fidelity/evidence/rfv2_publication_envelope.json`

The common deterministic case-set digest is:

```text
5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89
```

The deterministic raw-evidence bundle digest is:

```text
1da641ad27b58985e4e1cf8d9972af5a15f5105e85e2c319151a5e373b6afc46
```

## Immutable workflow publication

The live acquisition was produced from PR #91 head SHA:

```text
d55f812f492e8e93c1956fe79bceaf7e3754d7e9
```

GitHub Actions run `29444449427` published corpus artifact `8354853386`. Its archive digest is:

```text
a8be8c0782a8aeb037a2736de8adbd357c5074ad0da6355562e2543092d6af76
```

The artifact contains 24 content-addressed image objects, 24 public license-proof records, 24 individual qualification records, the qualification manifest, the qualification audit, the source-selection manifest and the deterministic bundle index. The archive is retained for 90 days; expiry does not invalidate the committed cryptographic evidence, but a future measurement or release relying on raw bytes must refresh or migrate the immutable artifact before it expires.

## Fail-closed result

The audit confirms:

- exactly 24 qualified cases;
- all ten required categories represented;
- zero duplicate case IDs;
- zero duplicate source digests;
- zero duplicate storage objects;
- zero duplicate inspection digests;
- no missing categories;
- no raw assets or public PII committed to the repository.

Passing RFV-2 means only that this finite, licensed and privacy-reviewed qualification corpus is complete and cryptographically bound. Fidelity scores are measured in RFV-3.
