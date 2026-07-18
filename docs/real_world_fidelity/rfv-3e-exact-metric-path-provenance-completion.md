# RFV-3E exact metric path provenance completion

## Purpose

This slice closes the evidence gap identified by PR #101 without changing the production vectorizer, final-artifact evaluator, winner selection, serializer, thresholds, corpus or release decision.

The earlier live artifact proved that the selected winner SVG existed and the exact evaluator was attempted for `qualification-public-10`, `qualification-public-14` and `qualification-public-18`, but it did not publish enough repeat-level evaluator detail to identify why SSIM, edge F1 and delta-E00 were unavailable.

## Added provenance

Each exact-evaluator attempt now records sanitized diagnostic metadata:

- evaluator report status and verdict;
- deterministic reason code;
- byte-read stability and evaluator determinism flag;
- hard-failure and soft-warning codes;
- unmeasured required groups;
- presence of evaluator metric groups;
- finite/missing/non-finite state for SSIM, edge F1 and delta-E00;
- missing exact component list;
- render outcome;
- deterministic SHA-256 of the sanitized report summary;
- selected and final artifact SHA bindings.

No filesystem path, raw SVG bytes, raw source bytes, token, secret or traceback is added to published provenance.

## Repeat-level publication

The conservative repeat aggregator continues to require identical artifact SHA-256 and identical metric-path decisions across all repeats. It additionally embeds three sanitized repeat provenance rows in the aggregated `BenchmarkResult.metric_provenance`:

- schema: `rfv3e-repeat-metric-provenance-v1`;
- exact repeat index;
- artifact SHA-256;
- evaluator report status, codes, group presence, component status and reason code.

Any repeat-level decision drift, field drift or artifact binding mismatch fails closed.

## Safety boundaries

This phase does not:

- modify evaluation algorithms or thresholds;
- manufacture unavailable metrics;
- treat fallback metrics as exact metrics;
- change corpus identity, repeat count, timeout or retry policy;
- authorize a production fix;
- advance RFV-3 or enable RFV-4.

The canonical state remains:

- `release_decision: no_go`;
- `rfv4_allowed: false`;
- RFV-3 pending remediation;
- RFV-4 pending.

## Next decision

The complete six-shard live workflow must publish the new repeat provenance for the three affected cases. Only that evidence may determine the next narrow diagnostic or production-fix slice. This work makes no universal 99% fidelity claim.
