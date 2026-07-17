# RFV-3D2 — Exact metric path provenance instrumentation

RFV-3D1 proved the committed `qualification-public-10/-14/-18` rows carry a
`partial_quality_report_fallback` signature instead of exact final-artifact
metrics. This slice instruments the runtime so every benchmark row proves which
path produced its metrics and, when the exact path could not run, records the
fail-closed reason class. No production quality algorithm, threshold, corpus or
release decision changes.

## What is recorded

Every `BenchmarkResult` now carries an optional `metric_provenance` object
(`rfv3d2-metric-provenance-v1`):

- `metric_source`: `exact_final_artifact` or `partial_quality_report`;
- `exact_evaluator_attempted` / `exact_evaluator_completed`;
- `exact_evaluator_failure_class`: `selected_svg_path_missing`,
  `selected_svg_file_missing`, `evaluator_exception`, `render_failure` or
  `exact_metrics_incomplete`;
- `exact_evaluator_failure_message_sanitized`: absolute paths and memory
  addresses redacted, length-capped;
- `selected_svg_path_present` / `selected_svg_file_present` /
  `selected_svg_sha256`;
- `fallback_used` / `fallback_source`;
- `artifact_sha256`: binds the provenance to the exact measured artifact.

## Fail-closed rules

- The exact path only counts as completed when `ssim`, `edge_f1` and
  `delta_e00` are finite together; anything else records an explicit class and
  falls back without fabricating values (missing metrics stay `null`).
- Fallback is never silent: `fallback_used=true` is written into evidence, and
  the existing RFV-3 gates already treat missing required metrics as `NO-GO`.
- Repeat aggregation requires identical provenance decisions across repeats
  (sanitized message text excluded); disagreement raises instead of averaging.
- Legacy result rows without `metric_provenance` remain valid (backward
  compatible), so previously published evidence is untouched.

## Safety behavior

- No raw assets, corpus or threshold changes.
- The release decision remains `NO-GO`; `rfv4_allowed` remains `false`.
- Published diagnostics contain no runner-local absolute paths or secrets.

## Next slice

RFV-3D3 must fix only the root cause this instrumentation proves live (the
selected winner SVG path/file not reaching the exact evaluator for the three
affected cases), then RFV-3D4 reruns the same 24-case corpus as a new immutable
evidence generation.
