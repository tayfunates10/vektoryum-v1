# RFV-3D1 — Missing metric coverage diagnosis

RFV-3C remains an empirical `NO-GO`. This slice diagnoses the three committed RFV-3 cases whose `ssim` and `edge_f1` values are absent without lowering thresholds, removing cases, fabricating metrics or enabling RFV-4.

## Finding

The affected cases are:

- `qualification-public-10`
- `qualification-public-14`
- `qualification-public-18`

Each has the same signature: `ssim=null` and `edge_f1=null`, while `alpha_iou` and `delta_e00` are measured.

The exact final-artifact evaluator creates visual SSIM and edge geometry metrics together after a successful final SVG render. Therefore this signature is classified as `partial_quality_report_fallback`: the benchmark extraction path did not have usable exact-winner metrics and fell back to a partial quality/legacy report.

## Safety behavior

- Unknown missing-metric patterns remain unclassified rather than guessed.
- Any required component metric gap is fail-closed.
- The release decision remains `NO-GO`.
- `rfv4_allowed` remains `false`.
- No raw assets are committed.

## Next slice

RFV-3D2 must instrument the production run so the exact-winner SVG path and evaluator provenance are retained for every repeat, then rerun only after tests prove that missing exact metrics cannot silently degrade to nullable fallback output.
