# RFV-3B — live sharded production measurement

RFV-3B executes the exact qualified 24-case RFV-2 corpus through the production vectorization pipeline without committing raw assets or measurement outputs to the repository.

## Execution model

- The reviewed RFV-2 allowlist is acquired once into a temporary GitHub-hosted runner directory.
- The deterministic corpus bundle and checksum evidence are uploaded as a workflow-local immutable artifact.
- The 24 cases are split deterministically into six shards of four cases.
- Every case is executed three times in fresh spawned processes through the production pipeline.
- Quality metrics use the conservative worst successful repeat; performance metrics use the median repeat.
- SVG artifact SHA-256 must be identical across all successful repeats for a case.
- A single retry is permitted only for bounded timeout or isolated-worker failures.

## Fail-closed boundaries

The workflow rejects source or bundle digest drift, unsafe archive paths, corpus shrinkage, duplicate cases, missing shards, missing metrics, non-finite metrics, non-deterministic SVG output, incomplete retry audit, non-transient retry use and any raw-asset repository boundary violation.

## Evidence

Successful execution publishes two 90-day GitHub Actions artifacts:

1. the aggregate 24-case `pipeline-results.json`, 72-sample `retry-audit.json` and immutable measurement envelope;
2. a publication envelope binding the evidence artifact ID, URL and digest to the exact pull-request head SHA.

RFV-3 remains pending until the complete live outputs are inspected and a separate reviewed quality/retry decision is merged. This phase does not claim universal 99% fidelity.
