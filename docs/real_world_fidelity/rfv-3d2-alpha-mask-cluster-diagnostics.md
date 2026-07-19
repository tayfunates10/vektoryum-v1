# RFV-3D2 alpha/mask cluster diagnostics

## Scope

This slice is diagnostics only. It does not change the vectorizer, pipeline, winner selection, serializer, final evaluator, thresholds, benchmark policy, corpus, repeat/retry/timeout policy, canonical quality decision, roadmap, release decision, or RFV-4 state.

Source evidence remains bound to main `c797f11f92a8d9d5ca879a798ff7c738590dad30`, RFV-3B head `5082e01d9777734e9d9da70a6f8d8d73e7676c30`, source run `29683096355`, aggregate artifact `8442804012`, and immutable corpus artifact `8441210832`.

## Proven diagnosis

The five lowest true-alpha cases are `qualification-public-11`, `-17`, `-12`, `-18`, and `-14`. The dedicated workflow run `29689639516`, at PR head `fc923cb2af7e5d9dfa3e63666dd3b0757185c6cc`, ran each case through the unchanged production pipeline, rendered the selected SVG to RGBA, and produced sanitized scalar/hash evidence.

All five independently satisfy:

- source has real transparency;
- selected SVG render soft coverage is at least `0.995`;
- alpha IoU equals source soft coverage within `1e-4`;
- diagnosis is `opaque_canvas_collapse`.

Artifact bindings:

- `qualification-public-11`: artifact `8444221003`, digest `sha256:a3ca5eac7452a2aa3872617e998404e2e811db770fc427476cdfbd1968215682`
- `qualification-public-17`: artifact `8443263360`, digest `sha256:1101433d860da4d3f6e5af1ded7b86255997ef31996adb908235900d086dafdd`
- `qualification-public-12`: artifact `8443463050`, digest `sha256:de1ae09b8ffd00a82af24a97e71c8e3b8dbb5030e3b75c94db172b8b5b48dc54`
- `qualification-public-18`: artifact `8443238140`, digest `sha256:3809702dd88229eefea184ab56c705ee56fc2e7924df067ca517ed4b093d5a69`
- `qualification-public-14`: artifact `8443326981`, digest `sha256:2e21a528ef0c7a2b386119f1cf1883f92b30243635d9b1c3c776ca82c8fcd27a`

The committed diagnosis status is therefore `proven_by_live_selected_svg_render`. Raw images, SVG bytes, alpha arrays, filesystem paths, tokens, secrets, and tracebacks are not published.

## State

A production fix is not authorized by this diagnostic slice. RFV-3 remains `pending`, release remains `NO-GO`, and `rfv4_allowed=false`. A separate small production-fix branch may be proposed from this evidence only after this diagnostics PR is merged.
