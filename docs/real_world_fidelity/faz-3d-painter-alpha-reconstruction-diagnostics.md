# FAZ 3D — Painter alpha reconstruction diagnostics

## Scope

This is diagnostics-only evidence produced with the production `resvg_py`
renderer. No evaluator threshold, TransformJournal policy, corpus, retry,
timeout or release decision was changed.

## Proven root cause

`root_cause: canvas_knockout_removes_required_alpha_support`

Painter encoded the source alpha mask correctly, but when a border-connected
full-canvas comparison shape was classified as a tracer background it was
removed from the visible paint and archived outside the mask. The rendered
alpha therefore became:

`source_alpha_mask ∩ residual_artwork_coverage`

instead of the required source alpha plane. Sparse artwork could cover only
part of the source-positive region, producing the public-05/public-06 class
of native alpha IoU failures.

## Controlled renderer evidence

The production renderer maps implicit and explicit luminance grays byte for
byte: 0→0, 1→1, 32→32, 64→64, 128→128, 192→192, 254→254, 255→255. This
disproves a luminance gamma or transfer-function defect.

On the same deterministic 64×64 synthetic alpha support:

| Mechanism | Alpha IoU | Alpha MAE | Render coverage |
|---|---:|---:|---:|
| Proven canvas removed outside mask | 0.2685607374 | 0.3201832771 | 0.1175608933 |
| Same canvas retained inside source-alpha mask | 1.0 | 0.0 | 0.4377441406 |

Source coverage was 0.4377441406 in both runs. Encoding, mask geometry,
quantization, renderer and stroke width were held constant; only the canvas
support placement changed.

## Authorized narrow fix

A proven comparison canvas may remain only as transform-owned underpaint
inside the existing source-alpha mask. It must not remain as an unmasked
visible canvas, and it must remain excluded from artwork identity. All final
alpha, appearance, topology, seam and complexity gates remain unchanged.

## Underpaint support-stroke measurement

The same fixture was rendered with every existing support-stroke candidate.
Alpha stayed exact (`IoU=1.0`, `MAE=0.0`) for all four widths. The source
topology sentinel separated them:

| Stroke | Component delta | Hole delta | Journal |
|---:|---:|---:|---|
| 1.0 | +2 | -3 | pass |
| 1.5 | +2 | -1 | pass |
| 2.0 | -2 | -2 | pass |
| 3.0 | +2 | -1 | fail: edge_f1_regression |

`1.5 px` is the smallest already-existing candidate that preserves exact alpha,
passes the unchanged journal and stays inside the pre-existing topology sentinel.
The fix therefore treats only sub-1.5px support as inadmissible when a proven
canvas is retained as transform-owned underpaint. No evaluator or journal
threshold is changed.
