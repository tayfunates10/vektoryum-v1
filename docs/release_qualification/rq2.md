# RQ-2 — Real image acceptance corpus

RQ-2 is accepted only when the immutable corpus manifest contains exactly the eight required difficult-image categories and every case declares finite quality and export requirements.

## Required categories

1. flat logo
2. badge or seal
3. small text
4. monoline artwork
5. multicolor artwork
6. low-resolution signage photograph
7. gradient artwork
8. native 4K artwork

## Mandatory outputs

Every case requires SVG, PDF, EPS and DXF. SVG must contain vector geometry, must not embed raster images, and all outputs must be non-empty and associated with the exact fixture digest.

## Fail-closed rules

Qualification fails for a missing or duplicate category, mutable or malformed digest, dimensions outside the declared disk and pixel budget, absent source classification, missing output format, embedded bitmap, non-finite threshold, threshold outside its legal range, artifact-fixture digest mismatch or incomplete result evidence.

The manifest is intentionally finite. Adding or replacing a fixture requires a reviewed manifest change and a new digest; silent corpus drift is forbidden.
