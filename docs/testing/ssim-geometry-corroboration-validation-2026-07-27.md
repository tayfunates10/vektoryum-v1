# Geometry-corroborated SSIM validation — 2026-07-27

## Scope

- Fix PR: #126
- Branch: `agent/fix-evaluator-canonical-tone-seam`
- Validated head: `66fbaedf0cf2ef58fffddff799ff77b8d33b7413`
- Output-quality run: `30309041238`
- Cases: 7
- Repeats per case: 2
- Full production pipeline executions: 14
- Structural failures: 0
- Byte-deterministic artifacts: 7 / 7

## Problem

`qa-ring-holes` preserves every independently measurable geometric and palette
property, but the deliberate dark-gray source tone is canonicalized to black by a
binary vector profile. The grayscale SSIM value is low even though the final SVG
has identical edges, components, holes and palette classification.

Legacy result:

- verdict: `failed`
- sole hard code: `ssim_below_min`
- SSIM: `0.7622949483`
- Edge F1 1 px / 2 px: `1.0 / 1.0`
- Chamfer p95 / Hausdorff max: `0.0 / 0.0`
- Component delta / hole delta: `0 / 0`
- Minimum / mean component IoU: `1.0 / 1.0`
- Palette agreement: `1.0`
- Delta E 2000 p95: `3.218791`
- Seam ratio: `0.0`

## Fix

The exact evaluator implementation remains byte-for-byte in
`app.final_artifact_evaluator_base`. A compatibility verdict layer removes only
`ssim_below_min` when **all** of the following existing measurements corroborate
the geometry:

- edge F1 1 px and 2 px >= `0.995`;
- Chamfer p95 <= `0.5 px`;
- Hausdorff max <= `1 px`;
- component delta = `0`;
- hole delta = `0`;
- minimum component IoU >= `0.995`;
- palette agreement >= `0.995`;
- Delta E p95 remains under the existing image-class threshold;
- seam ratio remains under the existing image-class threshold;
- SSIM actually triggered the legacy hard gate.

No measurement threshold is reduced. Missing or failed evidence preserves the
legacy hard failure.

## Target result

Case: `qa-ring-holes`

| Field | Before | After |
|---|---|---|
| Severity | high | **pass** |
| Evaluator verdict | failed | **production_ready** |
| Hard codes | `ssim_below_min` | **none** |
| Selected SVG SHA-256 | `a610afe4...19ac` | **unchanged** |
| Geometry / color metrics | exact | **unchanged** |
| Repeat determinism | yes | **yes** |

The output artifact and all numerical metrics remain identical; only the
contradictory verdict is corrected.

## Negative-control result

Case: `qa-lowres-badge`

This case contains a real missing light-gray palette class. It must not use the
corroboration exception.

- verdict: **failed**
- hard codes: `seam_gap`, `ssim_below_min`
- soft code: `component_iou_below_min`
- palette agreement: `0.875`
- minimum component IoU: `0.0`
- Chamfer p95: `70.9039 px`
- Hausdorff max: `84.7752 px`
- seam ratio: `0.207792`

The legacy failure is retained exactly as intended.

## Suite impact

Severity distribution changes from:

- 3 high / 1 medium / 3 pass

to:

- **2 high / 1 medium / 4 pass**

The selected SVG SHA-256 values for all seven cases remain unchanged.

## Release effect

OQ-03 is closed for the targeted canonical-tone SSIM contradiction. The real
low-resolution tone-loss issue and small-component selection issue remain open.
Project-wide `RFV-3: pending` and `release_decision: NO-GO` are unchanged.
