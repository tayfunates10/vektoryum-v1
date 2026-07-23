# FAZ 3D — Painter alpha reconstruction fidelity fix

## Change

A comparison canvas proven by the existing color-agnostic classifier is no
longer archived outside the painter mask. It is moved into the preserved
paint group, tagged `comparison-canvas-underpaint`, stripped of inherited
alpha and rendered exactly once through the existing source-alpha mask.

The canvas is therefore invisible where source alpha is zero and supplies
deterministic paint support wherever source alpha is positive. It remains
transform-owned and excluded from artwork identity; path data, palette and
gradient definitions are unchanged.

## Unchanged contracts

- Alpha IoU/MAE and all final evaluator thresholds.
- TransformJournal SSIM, edge, seam, topology, path, node and byte gates.
- FAZ 3B.1 attempt ledger, primary error precedence and three-tier encoding
  tournament.
- FAZ 3C retry eligibility.
- Corpus, shard count, repeats, retry count and timeouts.
- Vector-only output; no `<image>`, data URI or external raster.
- Atomic byte-identical rollback on rejection.
- `release_decision=no_go` and `rfv4_allowed=false`.

## Verification scope

The dedicated fidelity suite covers production-renderer transfer values,
masked canvas support, fractional/non-zero viewBox mapping, polygon/rect/
contour native equivalence, deterministic output, artwork fingerprint,
absent-canvas compatibility and end-to-end alpha/evaluator acceptance.
Public-05/public-06 remain subject to the unchanged RFV-3B live measurement.

## Underpaint support preflight

For a proven comparison canvas, the existing 1.0px support candidate is recorded
in the deterministic attempt ledger but is not admitted to alpha/journal
validation: a 1.0px stroke expands only 0.5px per side and the renderer evidence
shows a source-hole delta of -3. The next existing candidate, 1.5px, keeps alpha
IoU at 1.0, passes the unchanged journal and restores the existing topology
sentinel. Byte-rejected encodings still record all four stroke candidates, so the
FAZ 3B.1 observability and primary-error contracts are preserved.
