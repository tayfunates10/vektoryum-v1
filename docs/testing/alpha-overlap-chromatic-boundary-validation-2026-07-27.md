# Alpha overlap chromatic-boundary validation — 2026-07-27

## Scope

- Fix PR: #125
- Branch: `agent/fix-alpha-overlap-parent-selection`
- Validated head: `4e7f15b5906bda9b110a78f3a4d06e38ba0ab821`
- Output-quality run: `30308206200`
- Cases: 7
- Repeats per case: 2
- Full production pipeline executions: 14
- Structural failures: 0
- Byte-deterministic artifacts: 7 / 7

## Root cause chain

The original `qa-transparent-overlap` output preserved source alpha accurately, but
created a gray/cyan transition strip between the blue and red shapes.

Investigation separated three possible causes:

1. **Parent selection:** alpha-parent trials originally measured only gradient
   variants. Engine-diverse telemetry added the highest-fidelity VTracer parent.
2. **Alpha serialization:** the VTracer trial initially exceeded the existing byte
   budget. A compact `<use>`-based candidate-knockout encoder removed that byte
   inflation without changing path/node geometry or any quality gate.
3. **Candidate quality:** once measurable, the VTracer parent still failed the
   unchanged alpha-IoU gate (`0.871483 < 0.995`). It was correctly rejected.

The decisive production defect was therefore inside the gradient segmenter, not
inside the final parent chooser. Region segmentation used only grayscale Canny.
Blue and red had similar luminance, so their real hard chromatic boundary vanished
in grayscale and both areas became one fitted linear gradient.

## Fix

The original gradient implementation remains byte-for-byte in
`app.gradient_vectorize_base`. A compatibility wrapper replaces only region
segmentation:

- retain existing luminance edges;
- add per-channel RGB Canny edges;
- add a minimum local RGB-jump signal to close weak hard-edge gaps;
- keep thresholds above the derivative of broad smooth gradients;
- retain the existing nearest-region Voronoi fill for edge pixels.

No fidelity threshold, alpha threshold, journal gate, candidate margin or release
gate was lowered.

## Target-case comparison

Case: `qa-transparent-overlap`

| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| Severity | medium | **pass** | warning closed |
| Edge F1, 1 px | 0.916943 | **1.000000** | +0.083057 |
| Edge F1, 2 px | 0.918459 | **1.000000** | +0.081541 |
| Delta E 2000 mean | 1.393767 | **0.273551** | -1.120216 |
| Delta E 2000 p95 | 10.143423 | **0.491648** | -9.651775 |
| SSIM | 0.987544 | **0.992143** | +0.004598 |
| MS-SSIM | 0.988596 | **0.993799** | +0.005203 |
| Min component IoU | 0.980665 | **0.995908** | +0.015243 |
| Mean component IoU | 0.987706 | **0.996867** | +0.009161 |
| Palette agreement | 0.995239 | **0.998367** | +0.003128 |
| Seam ratio | 0.002294 | **0.000892** | -0.001402 |
| Alpha IoU | 0.997672 | **0.996773** | -0.000899 |
| Alpha MAE | 0.000781 | **0.001083** | +0.000302 |
| Paths | 2 | **3** | explicit color regions |
| SVG bytes | 5,116 | **5,335** | +219 bytes |

Alpha fidelity remains above the unchanged production threshold (`0.995` IoU),
while visible edge and color fidelity improve materially. The selected artifact is
byte-deterministic across both repeats.

## Selection evidence

- Mode: `logo_color`
- Winner: `logo_gradient / gradient`
- Final candidate fidelity: `98.64`
- Final candidate edge F1: `1.0`
- Selected SVG SHA-256:
  `1591e81094ab2cc453bc49a95392817df4c6d1293ca961067e87a45abc279eb8`
- Hard failures: none
- Soft warnings: none

The candidate remains named `logo_gradient`, but its segmentation now creates
three explicit hard-boundary regions instead of one false blue-to-red gradient.
Real smooth-ramp contract tests still produce `<linearGradient>` output.

## Regression boundary

The other six diagnostic cases retained the exact same selected SVG SHA-256 as the
pre-fix run:

- `qa-gray-border-counter`
- `qa-lowres-badge`
- `qa-monoline`
- `qa-ring-holes`
- `qa-shared-boundary`
- `qa-small-details`

Their quality metrics are unchanged apart from runtime noise and sub-micro floating
measurement noise in one Delta-E record.

## Contract coverage

The validated head passed:

- base output-quality contracts;
- neutral multi-tone routing contracts;
- alpha-parent selection contracts;
- compact candidate-knockout encoder contract;
- chromatic hard-boundary vs smooth-ramp gradient contracts;
- two-repeat production output diagnostic;
- output evidence completeness and determinism validation.

## Release effect

OQ-05 is closed for the targeted synthetic alpha-overlap failure. This does not
change the project-wide `RFV-3: pending` or `release_decision: NO-GO` state. The
remaining open diagnostic findings are evaluator sensitivity for low-resolution /
canonical-tone cases and small-component selection quality.
