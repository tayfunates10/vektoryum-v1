# Low-resolution broad neutral fill validation — 2026-07-28

## Scope

This report validates the targeted correction for `qa-lowres-badge`, where two
intentional neutral design tones were collapsed by the geometric-logo cleanup
chain:

- dark outer artwork: `#191919` → canonical black;
- light inner artwork: `#ebebeb` → canonical white.

The validation branch is stacked on PR #126 and remains draft/unmerged.

## Validated revision

- Branch: `agent/fix-lowres-badge-neutral-fill`
- Head: `1127acfcb68440aa4e9f30cf984d1a199fe1a366`
- Pull request: #127
- Output-quality workflow run: `30325442398`
- Artifact ID: `8675580492`
- Artifact digest: `sha256:085ae47232be02792919a9435a355b61da57626f02a6c68dd2b3e644704f3483`

## Root cause

The loss occurred at two independent stages.

1. `preprocess_geometric_logo` used legacy black/white palette hardening. The
   intentional light gray was close enough to white to be treated as an
   antialias tone. The dark gray was similarly canonicalized toward black.
2. SVG palette consolidation could snap a restored broad neutral fill back to a
   canonical endpoint.

A first light-only guard proved the light-gray hypothesis, but a combined dark +
light neutral mask failed on the exact production fixture because directly
adjacent dark and light regions merged into an oversized component. The final
implementation segments dark and light neutral classes independently.

## Final implementation

The established implementations remain byte-for-byte in:

- `engine/app/preprocess_base.py`
- `engine/app/geometry_cleanup_base.py`

Compatibility wrappers add only bounded neutral-fill guards.

### Preprocess guard

A source region is restored only when all relevant evidence passes:

- uniform near-white corner background;
- low chroma;
- dark or light neutral tone range;
- independent dark/light connected-component segmentation;
- minimum and maximum area bounds;
- not connected to the image border;
- erosion survival proving a broad region;
- distance-transform radius proving it is not a thin antialias film;
- source representative color visibly distinct from the background.

The existing dominant-palette reducer is reused after restoration.

### SVG canonical-snap guard

Canonical black or white is removed from one consolidation call only when a
separate, substantial traced neutral fill supplies evidence for that endpoint.
The dark guard excludes near-black cleanup clusters such as `#050505` and
`#0c0c0c`; these retain legacy black snapping. Exact black, exact white and red
canonical behavior remain unchanged.

## Production result

Comparison uses the validated PR #126 baseline against the final PR #127 run.

| Metric | PR #126 baseline | PR #127 final |
|---|---:|---:|
| Severity | high | **pass** |
| Evaluator verdict | failed | **production_ready** |
| Hard failures | `seam_gap`, `ssim_below_min` | **none** |
| Soft warnings | `component_iou_below_min` | **none** |
| SSIM | 0.7449274460 | **0.9998699497** |
| MS-SSIM | 0.7638266691 | **0.9998851382** |
| Delta-E00 mean | 2.2809219360 | **0.1971326768** |
| Delta-E00 p95 | 5.3717584609 | **0.4957423210** |
| Palette agreement | 0.8750000000 | **0.9968719482** |
| Minimum component IoU | 0.0000000000 | **0.9749755859** |
| Mean component IoU | 0.6902985075 | **0.9917964861** |
| Seam ratio | 0.2077922078 | **0.0051998782** |
| Edge F1, 1 px | 1.0000000000 | **1.0000000000** |
| Edge F1, 2 px | 1.0000000000 | **1.0000000000** |
| Path count | 7 | 10 |
| Node count | 192 | 236 |
| SVG bytes | 2,095 | 7,312 |

The selected candidate changed from a three-color `geo_clean` artifact with
fidelity `62.59` to `geo_detail` with fidelity `98.70`. The final SVG preserves
the source's four semantic design tones while retaining bounded edge shades from
vector tracing.

## Determinism and regression boundary

- Cases: 7
- Repeats: 2 per case
- Production pipeline executions: 14
- Structural failures: 0
- Byte deterministic: 7 / 7
- Target repeat SHA-256 values: identical
- Final target SVG SHA-256:
  `0dc7a8da21e1fbca00348d5618dc5bb1a9682d9bd86917ffac7faadacf2b28cd`

All six non-target selected SVG SHA-256 values are byte-for-byte identical to the
PR #126 baseline:

| Case | Baseline preserved |
|---|---|
| `qa-gray-border-counter` | yes |
| `qa-monoline` | yes |
| `qa-ring-holes` | yes |
| `qa-shared-boundary` | yes |
| `qa-small-details` | yes |
| `qa-transparent-overlap` | yes |

An intermediate implementation changed `qa-small-details` and reduced its minimum
component IoU. That revision was rejected. The final near-black exclusion restores
both its selected SVG SHA and every recorded quality metric to the PR #126
baseline.

## Severity movement

| Severity | PR #126 baseline | PR #127 final |
|---|---:|---:|
| Critical | 0 | 0 |
| High | 2 | **1** |
| Medium | 1 | 1 |
| Pass | 4 | **5** |

## Contract coverage

The validated run passed:

- base output-quality contracts;
- neutral multi-tone routing contracts;
- alpha-parent selection contracts;
- compact candidate-knockout contracts;
- chromatic gradient-boundary contracts;
- geometry-corroborated SSIM contracts;
- broad neutral preprocess contracts;
- canonical neutral-snap contracts;
- directly touching hard-edge dark/light fixture;
- thin light and dark antialias negative controls;
- broad near-black cleanup-cluster negative control;
- non-white canvas and chromatic-fill negative controls;
- diagnostic CLI and evidence-completeness checks.

## Release state

This closes only the targeted low-resolution badge neutral-fill loss. Project-wide
release state remains unchanged:

- `RFV-3: pending`
- `release_decision: NO-GO`

The remaining high-severity target is `qa-gray-border-counter`; the remaining
medium target is `qa-small-details`.
