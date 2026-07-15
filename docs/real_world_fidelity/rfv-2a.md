# RFV-2A — Qualified corpus evidence adapter

RFV-2A prepares the fail-closed registration and verification layer for the real qualification split. It does not claim that the 24 required real assets have already been collected.

## Scope

The adapter accepts only metadata for assets stored in an external immutable object store. Raw private customer files, public URLs and personal data are forbidden in the repository. Every accepted record must bind the source, consent and inspection evidence to independent SHA-256 digests.

## Qualification requirements

A complete RFV-2 qualification manifest requires exactly 24 records from the `qualification` split and coverage of all ten difficult-image categories defined by RFV-1. Every record must pass source, consent, object immutability, decoding and privacy verification. Dimensions, source format and file size remain bounded by the RFV-1 intake policy.

## Honest incomplete state

Until real assets are available, the repository manifest remains `awaiting_real_assets`, contains zero cases and cannot pass the complete qualification gate. Generated unit-test records prove validator behavior only; they are not real corpus evidence and are never written into the production manifest.

## Fail-closed behavior

Missing records, duplicate identity, invalid digests, mutable object storage, unsupported formats, unreviewed licenses, privacy gaps, public PII, decode failures, missing category coverage and corpus shrinkage reject qualification.

RFV-2 remains pending until reviewed real-world records replace the empty manifest and the complete evidence gate passes.
