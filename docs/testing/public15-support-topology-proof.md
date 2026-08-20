# P0-2 — qualification-public-15 support/topology proof

This document locks the diagnostic-first acceptance contract for Issue #172 P0-2.
It intentionally changes no production threshold, budget, evaluator rule, or
TransformJournal authority.

## Why proof comes before another support compactor

Earlier exact-colour support factoring reduced the real public-15 candidate but
remained above the unchanged byte budget. A smaller serialization is therefore
not sufficient evidence by itself. Before another representation can be accepted,
the first topology divergence point between source truth, support geometry,
complete candidate rendering, TransformJournal and FinalArtifactEvaluator must be
measured on the same pinned production candidate.

## Required measurements

For the same immutable `qualification-public-15` request and candidate lineage,
record all of the following before any acceptance decision:

1. Source connected-component count, hole count and stable topology signature.
2. Paint-support-only rendered connected-component/hole signature.
3. Complete candidate topology at a 512 max-side render.
4. Complete candidate topology at a 1024 max-side render.
5. TransformJournal parent topology, candidate topology and rejection codes.
6. FinalArtifactEvaluator topology metrics and hard-fail codes.
7. Serialized bytes, path count, node count, seam ratio and maximum seam width for
   every attempted support representation.

The diagnostic must identify the earliest stage where the topology signature
changes. If the source/support-only signatures already diverge, later renderer or
journal results may not be used to justify compaction.

## Candidate rules

A compact support representation may only be promoted when all of these are true:

- actual serialized bytes fit the existing production byte limit;
- source alpha and visible RGBA fidelity pass the existing gates;
- connected-component and hole regression are zero;
- seam metrics do not regress;
- path/node budgets remain unchanged and pass;
- unchanged FinalArtifactEvaluator passes;
- a fresh TransformJournal transaction passes;
- candidate lineage remains tied to the same source and selected paint candidate;
- no raster is embedded in SVG.

If no representation satisfies every condition, the production request must remain
fail-closed and the measured blocker must be reported. Threshold, byte/path/node
budget, corpus identity and acceptance policy are not remediation variables.

## Relationship to PR #164

PR #164 is evidence and an implementation experiment, not merge proof. Its measured
byte saving may be reused, but its representation must be re-evaluated on the
current post-P0-1 alpha stack and must satisfy the diagnostic sequence above before
any production mutation is accepted.
