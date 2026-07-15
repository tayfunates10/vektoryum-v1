# RFV-3A — Real-pipeline measurement and transient retry runner

RFV-3A connects the qualified 24-case RFV-2 corpus to the existing production pipeline benchmark adapter. It is preparation for live measurement and does not mark RFV-3 complete.

## Corpus binding

The runner accepts only an externally extracted RFV-2 evidence bundle. Before any pipeline execution it requires:

- exactly 24 qualification cases;
- case-set SHA-256 `5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89`;
- exact equality with the committed sanitized RFV-2 manifest;
- a matching deterministic bundle index;
- one content-addressed object per case;
- an exact source SHA-256 match for every object;
- approved privacy, source, license/consent, immutability and decode evidence.

Raw source bytes remain outside the repository.

## Measurement method

Every case runs through the production `run_pipeline` entry in a fresh spawned process.

- case count: 24;
- repeat count: 3;
- timeout: 1800 seconds per attempt;
- performance aggregation: median;
- quality aggregation: conservative worst case;
- artifact identity: all successful repeats must produce the same SVG SHA-256;
- unavailable metrics remain `null` and are never replaced with invented values.

The finite metric set is:

- fidelity;
- SSIM;
- edge F1;
- alpha IoU;
- CIEDE2000;
- path count;
- SVG bytes;
- render time;
- peak RSS memory.

## Automatic retry boundary

One retry is allowed for a repeat only when the isolated worker times out, exits without a result or exits with a worker failure. Source digest mismatch, corpus escape, invalid evidence, missing metric contracts, non-deterministic artifacts and future quality-threshold failures are not retryable.

Every attempt is written to `retry-audit.json`. Exhausted or non-retryable failures stop the run fail-closed.

## Completion boundary

Passing RFV-3A unit and contract CI proves that the measurement runner is bounded and fail-closed. RFV-3 remains pending until the complete 24-case live run produces reviewed pipeline results and a finite retry/quality release decision.
