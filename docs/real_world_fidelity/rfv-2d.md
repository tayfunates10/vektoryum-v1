# RFV-2D — Reviewed public-source acquisition

RFV-2D replaces the manual customer-image collection path with an exact allowlist of 24 publicly reusable sources. It does not place raw images in GitHub and does not mark RFV-2 complete before the assets are downloaded, decoded, hashed, stored externally and assembled through the complete qualification gate.

## Selected corpus

The selected set contains:

- 19 Openclipart graphics under CC0 1.0;
- 5 historic Library of Congress photographs from the Detroit Publishing Company collection;
- exactly 24 unique provider assets;
- exact coverage of all ten RFV difficult-image categories.

The source manifest is `engine/regression/rfv2_public_source_manifest.json`.

Category distribution:

| Category | Count |
| --- | ---: |
| flat_logo | 3 |
| badge_seal | 2 |
| small_text | 3 |
| monoline | 2 |
| multicolor | 2 |
| low_resolution_signage_photo | 3 |
| gradient_artwork | 2 |
| native_4k | 2 |
| transparent_dark_background | 2 |
| complex_illustration | 3 |

## Rights boundary

Openclipart source pages are paired with the provider's CC0/public-domain sharing statement. Library of Congress records are limited to the selected historic Detroit Publishing Company items and require the item page to state `No known restrictions on publication.`

The acquisition tool rejects unknown providers, non-HTTPS sources, hosts outside the allowlist, URL credentials, URL fragments, unapproved license classes, duplicate provider identities and category drift.

## Acquisition

Run the tool only with output directories outside the repository:

```bash
python engine/regression/rfv2_public_source_acquire.py \
  --all \
  --download-root /private/rfv2/public-downloads \
  --storage-root /private/rfv2/object-store \
  --records-dir /private/rfv2/records
```

The tool:

1. downloads only the exact reviewed source allowlist;
2. snapshots and hashes the source page and license evidence;
3. resolves Library of Congress image files only from official metadata and allowlisted hosts;
4. decodes every image with Pillow;
5. creates bounded low-resolution photograph fixtures for the signage category;
6. creates a non-upscaled 3840×2160 center crop from sufficiently large public-domain scans for the 4K category;
7. preserves reviewed Openclipart images as PNG, requiring real alpha for transparent cases;
8. passes each canonical source and its license-proof bundle through the merged RFV-2B secure intake;
9. writes raw files and records only to external paths.

After all 24 records exist, RFV-2C must be run with `--require-complete`.

## Honest progress boundary

`selected_not_acquired` means the source identities and rights paths are fixed, but no acquisition claim is made. RFV-2 remains pending until all 24 sources are successfully acquired and the external qualification manifest reaches `status: qualified`.
