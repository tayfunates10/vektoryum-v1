# RFV-1 — Real-world corpus intake foundation

RFV-1 creates the fail-closed intake contract required before Vektoryum can make any evidence-based real-world fidelity claim.

The existing RQ-2 corpus is intentionally synthetic and remains useful for deterministic release regression. It is not treated as proof that arbitrary customer images achieve 99% fidelity.

## Finite target

The qualified corpus target is 120 real-world cases, which is inside the fixed 100–300 case range. Cases are distributed across ten difficult-image categories and three immutable splits: calibration, qualification and holdout.

## Asset storage

Raw private or customer-provided images must not be committed to the public repository. Assets live in an immutable external object store. The repository stores only opaque object identifiers, source SHA-256 digests, consent or ownership evidence digests and non-identifying technical metadata.

## Source and license rules

Accepted sources are owned originals, explicit user-consented submissions, explicit customer-consented submissions, CC0 material and public-domain material. Unknown provenance, unverified scraping or sources whose terms prohibit evaluation are rejected.

## Privacy rules

Every record requires an approved privacy review. Public manifests must contain no names, email addresses, phone numbers, street addresses, faces, vehicle plates or other identifying content. Where redaction is required, the redacted source receives its own immutable digest before qualification.

## What this phase does not claim

RFV-1 does not claim that 120 real assets have already been acquired, does not publish a real-world score and does not change the production vectorization result. RFV-2 will populate and qualify the actual corpus against this contract.
