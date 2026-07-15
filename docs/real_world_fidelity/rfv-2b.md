# RFV-2B — Offline operator intake

RFV-2B provides an offline command-line tool for registering reviewed qualification images without placing raw customer assets in the public repository.

The tool must read one image and one license or consent proof, verify both as regular non-symlink files, decode the image with Pillow, enforce the RFV-1 size and format budgets, calculate SHA-256 digests, and copy the image to an external content-addressed directory using an atomic write.

The generated record must use the RFV-2 qualification fields, the `qualification` split, an opaque `rfv/qualification/...` object identifier, approved privacy review, and explicit confirmation that public personal information is absent.

The storage root and record output must resolve outside the repository. Existing records may be supplied so duplicate case IDs, source digests, or object IDs fail closed.

This phase does not populate the real qualification manifest and does not advance RFV-2. The manifest remains `awaiting_real_assets` until 24 reviewed real records covering all required categories are supplied.
