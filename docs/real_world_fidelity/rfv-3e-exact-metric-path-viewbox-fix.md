# RFV-3E exact metric path viewBox alpha-journal fix

## Evidence binding

This narrow production fix follows the diagnostics merged in PR #104 at main SHA
`d41c23c7382db9d7bd317a37f8b6b77c9f5a7689`. The source live evidence remains
PR #103, RFV-3B run `29623130466`, aggregate artifact `8424383328`, digest
`sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0`.

Affected cases: `qualification-public-10`, `qualification-public-14`, and
`qualification-public-18`. Their repeat-level evaluator reports stopped at
`viewbox_missing` because the valid source-dimension repair was rolled back by the
required-but-unmeasured `alpha_fidelity` journal gate.

## Fix

`TransformJournal` now captures a bounded RGBA render alpha plane for every parent
and candidate whenever `alpha_fidelity` is required. The private ndarray remains in
the in-request measurement cache only; public journal reports contain hashes,
coverage, and parent/candidate alpha comparison metrics, never raw pixel arrays.

The parent and candidate alpha planes are compared with the existing
`alpha_plane_metrics` implementation. The existing final-artifact image-class alpha
hard gates are reused without modification:

- alpha IoU below the existing class threshold rolls the candidate back;
- alpha MAE above the existing class threshold rolls the candidate back;
- missing RGBA rendering remains `required_metric_unmeasured` and fail-closed.

A viewBox/source-coordinate repair that preserves alpha and the existing structural,
visual, topology, seam, and complexity gates is accepted. A candidate whose RGB
render is identical but whose alpha plane regresses is explicitly rolled back.

## Non-scope

This change does not alter the vectorizer, winner selection, serializer, final
evaluator implementation or thresholds, corpus identity, repeat count, timeout,
retry policy, canonical release decision, or RFV-4 state.

## Required validation

Because production journal behavior changes, the unchanged 24-case corpus must run
three repeats per case. The release decision remains `no_go` and `rfv4_allowed`
remains `false` until the complete RFV-3B aggregate is reviewed.
