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

Color preprocessing converts RGBA to a white-composited RGB image for palette cleanup and writes that RGB result as the raster tracer input. The source alpha plane is absent before candidate generation, so a comparison background becomes traceable artwork.

Sending RGBA directly to the tracer is not sufficient for the existing hard contract: tracer-specific handling of soft alpha can still miss the required `alpha_iou_min=0.995` / `alpha_mae_max=0.005` clean-logo gates. Source alpha must therefore remain an explicit production invariant rather than an engine hint.

The gradient candidate has the same white-composite assumption and no native source-alpha region contract.

## Narrow fix

1. Keep the established color preprocessing unchanged for fully opaque pixels.
2. Resize and mirror-transform source RGBA into the exact trace-input coordinate space.
3. Keep the tracer input deliberately opaque RGB so alpha cannot be multiplied or interpreted differently by individual engines.
4. Use straight source RGB on partially transparent boundary pixels to prevent white halos.
5. Canonicalize fully transparent trace RGB to black and verify the written trace input by read-back.
6. Preserve a SHA-256 binding for the transformed source alpha in the preprocess report.
7. Reject the current gradient candidate for transparent sources until that engine has an alpha-aware mask contract; other candidates continue normally.
8. Before constructing any mask XML, stream-count merged alpha runs and compare both rectangle count and a conservative serialized-size upper bound with the unchanged `TransformJournal` byte-growth budget.
9. Fail closed before allocation or file mutation when a noisy/checkerboard alpha plane would exceed that budget.
10. After candidate selection and every SVG mutator, wrap the selected production content in a vector-only SVG mask generated from the transformed source alpha plane.
11. Encode alpha levels as editable SVG `<g>` groups and vertically merged `<rect>` primitives, without any embedded raster or data URI.
12. Remove candidate opacity attributes before wrapping so source alpha is applied exactly once.
13. Render the masked artifact through the production RGBA renderer and accept it only when the unchanged image-class alpha IoU and alpha MAE hard gates pass.
14. Submit the exact masked candidate to the real `TransformJournal`, which enforces structural safety, SSIM, edge, topology and hard path/node/byte complexity gates.
15. Merge the accepted alpha stage into the existing journal and require a valid SHA chain through the final artifact.
16. Re-score the exact accepted masked artifact before publication.
17. Preserve the selected vector-engine candidate identity after finalization. The artifact path, mask report and journal SHA move to the masked SVG, but names such as `geo_standard` or `logo_gradient` remain unchanged because `source_alpha_vector_mask` is a journaled artifact transform, not a new candidate generator.

Opaque inputs and non-color modes retain their existing behavior.

## Validation

The dedicated workflow requires:

- unit proof that transparent RGB is canonicalized and source alpha is staged deterministically;
- opaque-input and non-color compatibility tests;
- proof that non-color outputs bypass the final alpha stage byte-for-byte;
- fail-closed transparent-gradient behavior;
- checkerboard/noisy-alpha proof that the preconstruction gate rejects before the SVG bytes change;
- a real VTracer test that first reproduces the full-canvas opaque signature;
- application of the final vector-only source-alpha mask to that exact SVG;
- proof that the mask uses editable `<g>`/`<rect>` primitives and contains no `<image>` element;
- a real RGBA-render proof against the existing final evaluator's unchanged alpha IoU and alpha MAE thresholds;
- real `TransformJournal` acceptance and final-SHA chain validation;
- stable candidate-name proof for alpha-finalized outputs, with fail-closed rejection when the source candidate cannot be bound;
- unchanged structural, visual, topology and hard complexity gates;
- narrow diff scope;
- unchanged evaluator, source-truth, transform-journal, corpus, measurement policy, retry and release-decision files.

Because this change touches `engine/app/**`, the existing RFV-3B workflow must also run the complete immutable 24-case, three-repeat production measurement. Merge is not authorized unless every triggered workflow and real job is green.

## Release state

This production slice does not declare RFV-3 complete. The canonical state remains:

- RFV-3: `pending`
- release decision: `NO-GO`
- `rfv4_allowed=false`
- no universal 99% fidelity claim
