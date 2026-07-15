from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

MANIFEST = Path(__file__).with_name("rq2_corpus_manifest.json")
REQUIRED_CATEGORIES = {
    "flat_logo", "badge_seal", "small_text", "monoline", "multicolor",
    "low_resolution_signage_photo", "gradient_artwork", "native_4k",
}
REQUIRED_EXPORTS = ["svg", "pdf", "eps", "dxf"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def fixture_png(width: int, height: int, seed: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressor = zlib.compressobj(9)
    parts: list[bytes] = []
    for y in range(height):
        rgb = bytes((((y * seed) + 17) % 256, (y + seed * 31) % 256, (y * 3 + seed * 11) % 256))
        parts.append(compressor.compress(b"\x00" + rgb * width))
    parts.append(compressor.flush())
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", b"".join(parts)) + _chunk(b"IEND", b"")


def validate() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "vektoryum-rq2-corpus-v1":
        raise RuntimeError("invalid RQ-2 schema")
    if manifest.get("required_exports") != REQUIRED_EXPORTS:
        raise RuntimeError("mandatory export set drifted")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise RuntimeError("RQ-2 requires exactly eight cases")
    ids = [case.get("id") for case in cases]
    categories = [case.get("category") for case in cases]
    if len(set(ids)) != 8 or set(categories) != REQUIRED_CATEGORIES:
        raise RuntimeError("missing or duplicate RQ-2 category")
    for case in cases:
        width, height, seed = case.get("width"), case.get("height"), case.get("seed")
        if not all(isinstance(value, int) and value > 0 for value in (width, height, seed)):
            raise RuntimeError(f"invalid dimensions or seed for {case.get('id')}")
        if width * height > 3840 * 2160:
            raise RuntimeError(f"pixel budget exceeded for {case['id']}")
        threshold = case.get("min_iou")
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"invalid threshold for {case['id']}")
        digest = case.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"invalid digest for {case['id']}")
        actual = hashlib.sha256(fixture_png(width, height, seed)).hexdigest()
        if actual != digest:
            raise RuntimeError(f"fixture digest drift for {case['id']}")
        if case.get("license") != "synthetic-reviewed":
            raise RuntimeError(f"unreviewed source classification for {case['id']}")


if __name__ == "__main__":
    validate()
    print(json.dumps({"rq": "RQ-2", "status": "corpus_verified", "cases": 8}, sort_keys=True))
