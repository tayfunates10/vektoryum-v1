# RFV-3E exact metric path viewBox diagnostics

## Purpose

This diagnostics-only slice proves why the selected SVGs for `qualification-public-10`, `qualification-public-14`, and `qualification-public-18` still reach the exact evaluator without a usable `viewBox`.

It does not change the production vectorizer, evaluator, serializer, winner selection, thresholds, corpus, repeat policy, release decision, or RFV-4 state.

## Live evidence binding

The diagnosis is bound to the reviewed PR #103 live measurement:

- source main SHA: `19e91d10926f8709112b0afd6c576b886a5dfeb5`;
- measurement head SHA: `92fa263a938a39f44c288109c8f05a8a38c98f7e`;
- RFV-3B run: `29623130466`;
- aggregate artifact: `8424383328`;
- artifact digest: `sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0`.

All three affected cases produced the same repeat-level signature:

- evaluator report returned;
- hard fail code `viewbox_missing`;
- render outcome `not_reached`;
- SSIM, edge F1, and delta-E00 missing;
- only `A_structure` was available.

## Proven code path

`pipeline._restore_source_dimensions` already contains a safe fallback for an SVG with finite `width` and `height` but no `viewBox`. The deterministic diagnostic confirms that it creates `viewBox="0 0 48 32"` for the synthetic contract case.

The repaired bytes are then evaluated by `TransformJournal`. For RGBA sources the pipeline requests `alpha_fidelity`. The bounded journal measurement publishes complete structural and visual stage metrics for the repaired candidate, but it leaves `alpha_fidelity` in `required_unmeasured` because that journal measurement does not calculate alpha fidelity.

`TransformJournal._decide` rejects every candidate with a non-empty `required_unmeasured` list. Therefore the mandatory source-dimension/viewBox repair is rolled back with exactly:

```text
required_metric_unmeasured
```

The byte-identical control, with no unmeasured required metric, accepts the same viewBox transformation. This isolates the failure from SVG parsing, coordinate normalization, visual regression, topology regression, complexity limits, and winner routing.

## Root cause

Status: **proven**

Class:

```text
transform_journal_required_alpha_metric_deadlock
```

The viewBox repair is valid, but the RGBA transform journal rolls it back solely because `alpha_fidelity` is required and remains unmeasured by the stage gate.

## Next narrow fix scope

The next PR may do one of the following, selected by tests and minimal risk:

1. calculate alpha fidelity in the transform journal before enforcing it; or
2. place the mandatory coordinate-contract repair behind a dedicated fail-closed structural policy that cannot silently accept visual or alpha regression.

The fix must not:

- lower evaluator thresholds;
- fabricate alpha or visual metrics;
- change winner selection;
- change corpus identity, repeat count, timeout, or retry policy;
- enable RFV-4;
- claim universal 99% fidelity.

After the fix, the unchanged 24-case corpus must run three times per case. The release decision cannot change before that full live rerun.

## Canonical state

- RFV-3: pending remediation;
- release decision: `no_go`;
- `rfv4_allowed`: `false`;
- RFV-4: pending.
