"""FAZ 3 — raw-byte, ICC-normalized RGBA and CMYK fallback metadata."""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

from PIL import Image, ImageCms

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def test_raw_hash_and_normalized_rgba_hash_are_separate_truths() -> None:
    from app.input_guard import validate_and_load

    image = Image.new("RGB", (12, 9), (27, 144, 211))
    plain = io.BytesIO()
    image.save(plain, "PNG")
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    profiled = io.BytesIO()
    image.save(profiled, "PNG", icc_profile=profile)

    a = validate_and_load(plain.getvalue())
    b = validate_and_load(profiled.getvalue())
    assert a.sha256 == hashlib.sha256(plain.getvalue()).hexdigest()
    assert b.sha256 == hashlib.sha256(profiled.getvalue()).hexdigest()
    assert a.sha256 != b.sha256
    assert a.normalized_rgba_sha256 == b.normalized_rgba_sha256
    assert a.color_profile_status == "not_present"
    assert b.color_profile_status == "icc_to_srgb"


def test_cmyk_without_valid_icc_reports_explicit_fallback() -> None:
    from app.input_guard import validate_and_load

    payload = io.BytesIO()
    Image.new("CMYK", (16, 12), (10, 80, 30, 5)).save(payload, "JPEG", quality=95)
    loaded = validate_and_load(payload.getvalue())
    assert loaded.image.mode == "RGB"
    assert loaded.color_profile_status == "cmyk_fallback_srgb"
    assert "cmyk_without_valid_icc_fallback" in loaded.normalization_warnings
    assert loaded.normalized is True
    assert loaded.normalized_rgba_sha256


def test_icc_conversion_preserves_alpha_plane() -> None:
    from app.input_guard import validate_and_load

    image = Image.new("RGBA", (8, 8), (227, 0, 11, 0))
    for x in range(8):
        for y in range(8):
            image.putpixel((x, y), (227, 0, 11, x * 32))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    payload = io.BytesIO()
    image.save(payload, "PNG", icc_profile=profile)
    loaded = validate_and_load(payload.getvalue())
    assert loaded.has_alpha is True
    assert loaded.image.mode == "RGBA"
    assert list(loaded.image.getchannel("A").getdata()) == list(image.getchannel("A").getdata())
