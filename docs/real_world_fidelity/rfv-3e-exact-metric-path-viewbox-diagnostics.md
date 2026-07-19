# RFV-3E exact metric path viewBox diagnosis — historical evidence

## Status

This document and its JSON evidence are an immutable historical record of the defect proved by PR #104. They are no longer recomputed against current production code.

The bound source remains:

- PR #103;
- main SHA `19e91d10926f8709112b0afd6c576b886a5dfeb5`;
- RFV-3B run `29623130466`;
- aggregate artifact `8424383328`;
- digest `sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0`.

## Historical finding

For `qualification-public-10`, `qualification-public-14`, and `qualification-public-18`, `_restore_source_dimensions` created a valid viewBox, but the pre-fix RGBA journal rolled the candidate back because `alpha_fidelity` remained unmeasured. The proven class is:

```text
transform_journal_required_alpha_metric_deadlock
```

## Superseding production contract

PR #105 introduces a separate production contract. Alpha preservation is measured only for the mandatory `restore_source_dimensions` stage. Downstream mutators retain the prior fail-closed scope. Current behavior is tested in `engine/test_transform_journal.py` and `.github/workflows/real-world-fidelity-rfv3e-viewbox-fix.yml`.

The historical JSON is intentionally unchanged; changing current code must not rewrite past evidence.

## Canonical state

- RFV-3: pending;
- release decision: `no_go`;
- `rfv4_allowed`: `false`;
- RFV-4: pending.
