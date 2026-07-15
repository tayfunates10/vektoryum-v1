# RFV-2C — Deterministic qualification manifest assembly

RFV-2C assembles the individual offline records produced by `rfv2_secure_intake.py`. It does not collect images, does not download sources and does not mark RFV-2 complete without the full reviewed corpus.

## Operator flow

1. Keep raw images, permission/license documents and individual record JSON files outside the public repository.
2. Register each reviewed image with `engine/regression/rfv2_secure_intake.py`.
3. Place only the resulting individual record JSON files in one external directory.
4. Assemble a progress manifest and audit report:

```bash
python engine/regression/rfv2_manifest_assembler.py \
  --records-dir /private/rfv2/records \
  --manifest-out /private/rfv2/qualification-manifest.json \
  --audit-out /private/rfv2/qualification-audit.json
```

5. Run the finite completion gate only when all 24 reviewed records are present:

```bash
python engine/regression/rfv2_manifest_assembler.py \
  --records-dir /private/rfv2/records \
  --manifest-out /private/rfv2/qualification-manifest.json \
  --audit-out /private/rfv2/qualification-audit.json \
  --require-complete
```

## Fail-closed rules

The assembler rejects unapproved fields, path or URL leakage, invalid record schemas, unsupported formats, unreviewed licenses, privacy gaps, tampered inspection evidence, invalid digests, duplicate identities, corpus overflow and output paths inside the repository.

A partial collection is emitted only with `status: collecting`. `status: qualified` is possible only with exactly 24 records and coverage of all ten required difficult-image categories.

The generated manifest contains sanitized qualification fields only. Raw images, permission documents, local paths, network locations, credentials and full inspection objects are excluded.

## Honest progress boundary

RFV-1 remains the only merged RFV phase. RFV-2 remains pending until Issue #88 is completed with 24 reviewed real-world records and the `--require-complete` gate passes. Generated unit fixtures are test evidence for the assembler, not real corpus evidence.
