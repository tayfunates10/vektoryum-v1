# Neutral multi-tone routing validation — 2026-07-27

## Scope

- Fix PR: #124
- Branch: `agent/fix-neutral-multitone-routing`
- Validated head: `acbe9eca36f81edf32fa84352c3936b80c2d7a43`
- Output-quality run: `30303256225`
- Cases: 7
- Repeats per case: 2
- Full production pipeline executions: 14
- Structural failures: 0
- Byte-deterministic artifacts: 7 / 7

## Root cause

The original analyzer recommended the binary `single_color` profile for artwork containing three meaningful neutral design tones: white, dark gray and a substantial mid-gray border. The binary candidate pool cannot preserve all three regions, so the border disappeared and the inner geometry degraded.

The fix preserves the original analyzer implementation byte-for-byte in `app/analyzer_base.py` and adds a compatibility wrapper in `app/analyzer.py`. The wrapper only intervenes when:

1. anti-alias film removal has already completed;
2. a neutral mid-tone still occupies at least 1% of the image;
3. the tone is visibly separated from black and white;
4. the original recommendation is `single_color` or `lineart`.

That narrow class is routed to the color-preserving `logo_color` profile. True binary silhouettes and anti-alias-only gray films keep their previous routing.

## Target-case comparison

Case: `qa-gray-border-counter`

| Metric | Baseline `single_color` | Intermediate `geometric_logo` | Final `logo_color` |
|---|---:|---:|---:|
| SSIM | 0.563917 | 0.726569 | **0.982917** |
| Edge F1 (1 px) | 0.286379 | 1.000000 | **1.000000** |
| Delta E 2000 p95 | 26.055353 | 2.515978 | **0.224853** |
| Component delta | 21 | 3 | **4** |
| Hole delta | 0 | 0 | **0** |
| Minimum component IoU | 0.002129 | 0.969045 | **0.969138** |
| Seam ratio | 0.364551 | 0.000000 | **0.000000** |
| Deterministic | yes | yes | **yes** |

Selected pipeline decisions:

- Baseline: `single_color` → `single_contour / opencv_contour`
- Intermediate: `geometric_logo` → `geo_standard / vtracer`
- Final: `logo_color` → `logo_clean / vtracer`

## Outcome

The material production defect is closed:

- the gray outer border is retained;
- the dark inner region keeps its actual tone instead of being forced to canonical black;
- the seam/gap failure is eliminated;
- edge fidelity reaches 1.0;
- p95 color error falls from 26.06 to 0.22;
- the smallest-region IoU rises from 0.002 to 0.969;
- both repeats produce the same SVG SHA-256.

The final evaluator still reports `ssim_below_min` and `topology_component_delta`. Visual evidence and the remaining metrics indicate these are now evaluator sensitivity issues at anti-aliased one-pixel boundaries, not the original missing-region production defect. They belong with the existing OQ-02/OQ-03 evaluator-remediation work and are not hidden or reclassified by this change.

## Regression boundary

The other six diagnostic cases retained the exact same selected SVG SHA-256 as the baseline run. Their modes, severities and metrics did not change:

- `qa-lowres-badge`
- `qa-monoline`
- `qa-ring-holes`
- `qa-shared-boundary`
- `qa-small-details`
- `qa-transparent-overlap`

## Contract results

The following completed successfully on the validated head:

- Output quality diagnostic contract
- Output quality production diagnostic
- AI analyzer release contract
- Core all-mode release contract
- RFV-1, RFV-2A/B/C/D/E, RFV-3A/C/D1/D2/E contracts
- Release qualification RQ-3 and RQ-4

## Release effect

This validation closes OQ-01's material auto-routing loss only. It does not change the project-wide `RFV-3: pending` or `release_decision: NO-GO` state, and it does not waive the remaining evaluator, small-detail or alpha/gradient findings.
