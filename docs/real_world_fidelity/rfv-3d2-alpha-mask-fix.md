# RFV-3D2 alpha-mask production fix

## Evidence binding

This fix is based on the immutable RFV-3B evidence diagnosed by PR #106.

- source main SHA: `c797f11f92a8d9d5ca879a798ff7c738590dad30`
- RFV-3B measurement head: `5082e01d9777734e9d9da70a6f8d8d73e7676c30`
- RFV-3B run: `29683096355`
- aggregate artifact: `8442804012`
- aggregate digest: `sha256:43f6664b557ffc4e2cdb82a04a09ca65318721e01a7c0408968ed6ffe2a3aa22`
- corpus artifact: `8441210832`
- corpus digest: `sha256:2b768850b11fabf37c2dd761c1c477e0798dd5b709d6a0643bcf402224b67744`
- diagnostics PR: #106
- diagnostics merge commit: `cad189a61b54d933cdac2555b3d1ddcf2355a765`

The live selected-SVG renderer proved `opaque_canvas_collapse` for the five lowest true-alpha cases:

- `qualification-public-11`
- `qualification-public-17`
- `qualification-public-12`
- `qualification-public-18`
- `qualification-public-14`

For each case, selected-SVG render alpha coverage was effectively `1.0` and alpha IoU matched source soft coverage, which is the deterministic signature of a full-canvas opaque render.

## Root production path

Color preprocessing converts RGBA to an RGB white composite for palette cleanup and writes that RGB result as the raster tracer input. The source alpha plane is therefore absent before candidate generation. A white canvas that was only a comparison background becomes traceable artwork.

The gradient candidate has the same white-composite assumption and no native source-alpha region contract.

## Narrow fix

1. Keep the existing RGB preprocessing unchanged for opaque pixels.
2. Resize and transform source RGBA into the exact trace-input coordinate space.
3. Restore the source alpha plane before VTracer receives the PNG.
4. Use straight source RGB on partially transparent boundary pixels to avoid white halos on black/checker backgrounds.
5. Canonicalize fully transparent RGB to zero.
6. Read the written PNG back and require byte-equivalent alpha; failure raises instead of silently returning to the opaque path.
7. Reject the current gradient candidate for transparent sources until that engine has an alpha-aware mask contract. Other candidates continue normally.

Opaque inputs and non-color modes retain their existing preprocessing behavior.

## Validation

The dedicated workflow requires:

- unit proof that source alpha is restored exactly;
- opaque-input and non-color compatibility tests;
- fail-closed transparent-gradient behavior;
- a real VTracer plus real RGBA-render integration test;
- the existing final evaluator's image-class alpha IoU and alpha MAE thresholds, unchanged;
- narrow diff scope;
- unchanged evaluator, source-truth, transform-journal, corpus, measurement policy, retry and release-decision files.

Because this change touches `engine/app/**`, the existing RFV-3B workflow must also run the complete immutable 24-case, three-repeat production measurement. Merge is not authorized unless every triggered workflow and real job is green.

## Release state

This production slice does not declare RFV-3 complete. The canonical state remains:

- RFV-3: `pending`
- release decision: `NO-GO`
- `rfv4_allowed=false`
- no universal 99% fidelity claim
