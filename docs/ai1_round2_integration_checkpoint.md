# AI-1 Round 2 integration checkpoint

Status: **NO-GO / combined validation required**.

This checkpoint records manager integration provenance and intentionally changes no production quality threshold, budget, corpus policy, evaluator, fixture, or release gate.

## Combined branch

- Branch: `codex/ai1-integration-near-zero`
- Round-1 base before Round 2: `a1e3d5368c3bc0712e406ce1f6b2acc991858777`
- AI-4 Round 2 accepted head: `83cd01446a1a4b7a6e0cc9d721d3eeeecc2601a8` (Issue #146 / PR #150)
- AI-2 Round 2 technically accepted head: `2b5c93c47cf0beeb4f95d221470f1c39ff9362a9` (Issue #144 / PR #148)
- AI-3 Round 2 accepted head: `3933e07cc5c248b7833dcf6bf856d6e5f3d67dd3` (Issue #145 / PR #149)
- Combined Round-2 merge head before this checkpoint: `cd35aaa2464ea1639bab3daa4c9d31c42a68b4be`

## Manager acceptance notes

- AI-4 is measurement/diagnostic-only and preserves the verified analytic source contract. It removes only demonstrated palette/topology false positives for continuous/flattened alpha semantics; source geometry is never reconstructed into the render.
- AI-2 adds a narrow fully-opaque exact-source-palette vector candidate only when the source palette already fits the existing mode palette cap. It does not quantize, embed raster data, or increase a shared byte/path/node budget. Its separate Issue #144 full-pipeline acceptance gate passed. The specialist did not write the mandatory final Issue/PR report; this is recorded as a process defect and is not treated as quality evidence.
- AI-3 repairs visible RGB paint under an already accepted alpha mask only when existing alpha/composite and parent-relative journal budgets pass. Failed repair candidates restore the prior artifact; raster embedding is rejected.

## Required combined release evidence

Do not merge PR #143 to `main` until the combined head demonstrates all of the following:

1. native 12/12 `near_zero.ready=true` on the unchanged 12-case production suite;
2. 12/12 deterministic repeated SVG SHA;
3. verified 64/128/256/512/1024 source contract and complete topology/multiscale metrics;
4. no hard failure codes and no required metric missing;
5. Exact Final SVG, Benchmark, Core, analyzer, AI-2, AI-3, F99 and RQ gates green on the combined head;
6. RFV-3B provenance resolved through reviewed fail-closed requalification/normalization, never silent repinning;
7. RFV-3D2 integration-scope validation resolved without weakening the alpha-only patch scope contract;
8. no quality-threshold, byte/path/node budget, corpus, fixture, or release-policy relaxation;
9. raster-in-SVG count remains zero.

Refs #136 #143 #144 #145 #146 #147.
