# RFV-3D2 alpha/mask cluster diagnostics

## Scope

This slice is diagnostics only. It does not change the vectorizer, pipeline,
winner selection, serializer, evaluator thresholds, benchmark policy, corpus,
retry/timeout policy, release decision, or RFV-4 state.

Evidence is bound to:

- main after PR #105: `c797f11f92a8d9d5ca879a798ff7c738590dad30`
- RFV-3B head: `5082e01d9777734e9d9da70a6f8d8d73e7676c30`
- run: `29683096355`
- aggregate artifact: `8442804012`
- immutable corpus artifact: `8441210832`

## Observed signature

The five lowest true-alpha cases are `qualification-public-11`, `-17`, `-12`,
`-18`, and `-14`. In every case the aggregate alpha IoU equals the source alpha
plane's soft coverage within `4.48e-6`.

For the existing soft-IoU definition, a full-canvas opaque render produces
exactly that signature: the intersection is the source alpha sum while the union
is the full canvas. This is strong evidence of an opaque-canvas collapse, but the
committed evidence deliberately labels it as pending direct render confirmation.

## Live confirmation

The dedicated workflow downloads the exact immutable corpus artifact, runs each
of the five cases once through the production pipeline, renders the selected SVG
to RGBA, and publishes only sanitized hashes and scalar alpha metrics. It must
observe:

- source has real transparency;
- render soft coverage at least `0.995`;
- alpha IoU equal to source soft coverage within `1e-4`;
- diagnosis `opaque_canvas_collapse`.

Raw images, SVG bytes, alpha arrays, filesystem paths, tokens, and tracebacks are
not published.

## State

RFV-3 remains `pending`, release remains `NO-GO`, and `rfv4_allowed=false`.
A production fix is not authorized by this diagnostic slice.
