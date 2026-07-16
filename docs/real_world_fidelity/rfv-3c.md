# RFV-3C — reviewed real-world quality and retry decision

RFV-3C reviews the immutable live evidence produced by RFV-3B. It completes the measurement and automatic retry-gate phase, but it does **not** approve release or claim universal 99% fidelity.

## Bound evidence

- Measurement head SHA: `4abb6840887bb0803ea7bad2b6f91babba621e4f`
- GitHub Actions run: `29457765838`
- Aggregate artifact ID: `8362376085`
- Aggregate artifact SHA-256: `a98861c6bd9fb09a9d948b25317d11f53c69c25243bfa1eb6e1d1306cba91b77`
- Qualified corpus: 24 unique cases, case-set SHA-256 `5f151a6cb1a433b0cb0989a67bd7cc7940162f4b36d67903d6ccdd173f9e7d89`
- Repeats: 72 successful samples, zero retries, maximum attempt count one
- Raw assets committed to the repository: no

## Finite decision contract

The existing final-fidelity contract requires at least 99% overall fidelity per artifact and at least 0.98 for measured component scores. RFV-3C applies those finite thresholds to the real-world qualification corpus:

- overall fidelity: 0 of 24 cases passed; minimum 21.34, median 81.005, maximum 92.47;
- SSIM: 8 passed, 13 were below 0.98 and 3 were unavailable;
- edge F1: 8 passed, 13 were below 0.98 and 3 were unavailable;
- alpha IoU: 3 passed and 21 were below 0.98.

Missing required component metrics fail closed. They are not estimated or replaced.

## Decision

- Measurement completeness gate: **passed**
- Automatic retry gate: **passed**
- Repeat artifact determinism gate: **passed**
- Real-world quality gate: **failed**
- Release decision: **NO-GO**

RFV-3 is implemented as a measurement and decision gate. RFV-4 remains blocked until engine remediation is followed by a new immutable 24-case run whose reviewed quality decision is `go`.

## Performance observations

These are measurements, not release thresholds:

- render time: median 175,824.30 ms; p95 1,258,261.12 ms; maximum 1,945,554.61 ms;
- peak RSS: median 568.73 MB; p95 1,274.05 MB; maximum 1,800.38 MB;
- SVG size: median 75,238.5 bytes; p95 12,358,999 bytes; maximum 28,409,228 bytes;
- path count: median 37; p95 1,543; maximum 142,058 among measured cases.

The evidence and deterministic evaluator are stored under `docs/real_world_fidelity/evidence/` and `engine/regression/rfv3_quality_decision.py`.
