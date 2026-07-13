"""Safe image intake with explicit raw-byte and color-managed RGBA truths."""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field

from PIL import Image, ImageOps, UnidentifiedImageError

from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)
_SAFE_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


class InputError(Exception):
    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


@dataclass
class LoadedImage:
    image: Image.Image
    format: str
    has_alpha: bool
    sha256: str
    width: int
    height: int
    normalized: bool
    safe_suffix: str
    normalized_rgba_sha256: str | None = None
    color_profile_status: str = "not_present"
    normalization_warnings: list[str] = field(default_factory=list)


def _has_transparency(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def _canonicalize_transparent_rgb(img: Image.Image) -> tuple[Image.Image, bool]:
    """Zero hidden RGB where alpha is exactly zero.

    Fully transparent RGB values are visually undefined, but keeping arbitrary
    encoder residue makes normalized hashes and later color statistics depend on
    invisible data. Canonicalization makes equivalent transparent inputs share
    one source truth without touching partially transparent edge colors.
    """
    if "A" not in img.getbands():
        return img, False

    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    transparent = alpha.point(lambda value: 255 if value == 0 else 0, mode="L")
    if transparent.getbbox() is None:
        return rgba, img.mode != "RGBA"

    canonical = rgba.copy()
    zero_rgb = Image.new("RGB", rgba.size, (0, 0, 0))
    canonical.paste(zero_rgb, mask=transparent)
    canonical.putalpha(alpha)
    return canonical, True


def _normalized_rgba_sha(img: Image.Image) -> str:
    rgba, _changed = _canonicalize_transparent_rgb(img.convert("RGBA"))
    payload = rgba.width.to_bytes(8, "big") + rgba.height.to_bytes(8, "big") + rgba.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _profile_to_srgb_preserve_alpha(img: Image.Image, icc: bytes) -> Image.Image:
    """Apply ICC to color channels while preserving the exact alpha plane."""
    from PIL import ImageCms  # noqa: PLC0415

    alpha = img.getchannel("A") if "A" in img.getbands() else None
    color = img.convert("RGB") if img.mode not in ("RGB", "CMYK", "LAB", "L") else img
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    dst = ImageCms.createProfile("sRGB")
    converted = ImageCms.profileToProfile(color, src, dst, outputMode="RGB")
    if alpha is not None:
        converted.putalpha(alpha)
    return converted


def _to_srgb_with_metadata(img: Image.Image) -> tuple[Image.Image, bool, str, list[str]]:
    """Normalize ICC/CMYK source and report the exact conversion path."""
    icc = img.info.get("icc_profile")
    warnings: list[str] = []
    if icc:
        try:
            return _profile_to_srgb_preserve_alpha(img, icc), True, "icc_to_srgb", warnings
        except Exception as exc:  # noqa: BLE001
            warnings.append("icc_conversion_failed")
            logger.warning("ICC->sRGB dönüşümü başarısız; güvenli fallback deneniyor: %s", exc)
    if img.mode == "CMYK":
        warnings.append("cmyk_without_valid_icc_fallback")
        return img.convert("RGB"), True, "cmyk_fallback_srgb", warnings
    if icc:
        return img.convert("RGBA" if _has_transparency(img) else "RGB"), True, "icc_unusable_fallback", warnings
    return img, False, "not_present", warnings


def _to_srgb(img: Image.Image) -> tuple[Image.Image, bool]:
    """Backward-compatible helper retained for callers/tests."""
    converted, changed, _status, _warnings = _to_srgb_with_metadata(img)
    return converted, changed


def validate_and_load(
    contents: bytes,
    filename: str | None = None,
    settings: Settings | None = None,
) -> LoadedImage:
    """Validate bytes, decode once and expose raw hash + normalized RGBA hash."""
    del filename
    s = settings or get_settings()
    if len(contents) > s.max_upload_bytes:
        raise InputError(
            "file_too_large", 413,
            f"Dosya çok büyük (en fazla {s.max_upload_bytes // (1024 * 1024)} MiB).",
        )
    if not contents:
        raise InputError("corrupt_image", 400, "Boş dosya.")

    try:
        header = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise InputError("unsupported_format", 415, "Tanınmayan/desteklenmeyen görüntü formatı.") from None
    except Exception:
        raise InputError("corrupt_image", 400, "Görsel dosyası bozuk veya okunamıyor.") from None

    fmt = (header.format or "").upper()
    if fmt not in s.allowed_formats:
        raise InputError(
            "unsupported_format", 415,
            f"Desteklenmeyen format: {fmt or 'bilinmiyor'}. İzin verilen: {sorted(s.allowed_formats)}.",
        )
    if int(getattr(header, "n_frames", 1) or 1) > 1:
        raise InputError("animated_not_supported", 415, "Animasyonlu/çok kareli görseller desteklenmiyor.")

    width, height = header.size
    if width <= 0 or height <= 0:
        raise InputError("corrupt_image", 400, "Geçersiz görsel boyutu.")
    if width > s.max_side or height > s.max_side or width * height > s.max_pixels:
        raise InputError(
            "image_too_large", 413,
            f"Görsel çok büyük ({width}x{height}); en fazla {s.max_side}px kenar / {s.max_pixels // 1_000_000} MP.",
        )

    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        raise InputError("corrupt_image", 400, "Görsel bütünlük doğrulaması başarısız.") from None

    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InputError("image_too_large", 413, "Görsel çözümleme sınırı aşıldı.") from None
    except Exception:
        raise InputError("corrupt_image", 400, "Görsel çözümlenemedi.") from None

    before_size = img.size
    orientation = 1
    try:
        orientation = int(header.getexif().get(0x0112, 1) or 1)
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        pass
    exif_changed = img.size != before_size or orientation != 1

    source_has_alpha = _has_transparency(img)
    img, color_changed, profile_status, warnings = _to_srgb_with_metadata(img)
    if source_has_alpha and img.mode != "RGBA":
        img = img.convert("RGBA")
        color_changed = True
    elif not source_has_alpha and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
        color_changed = True

    transparent_rgb_changed = False
    if source_has_alpha:
        img, transparent_rgb_changed = _canonicalize_transparent_rgb(img)
        if transparent_rgb_changed:
            warnings.append("transparent_rgb_canonicalized")

    raw_sha = hashlib.sha256(contents).hexdigest()
    normalized_rgba_sha = _normalized_rgba_sha(img)
    return LoadedImage(
        image=img,
        format=fmt,
        has_alpha=source_has_alpha,
        sha256=raw_sha,
        width=img.width,
        height=img.height,
        normalized=bool(exif_changed or color_changed or transparent_rgb_changed),
        safe_suffix=_SAFE_SUFFIX.get(fmt, ".png"),
        normalized_rgba_sha256=normalized_rgba_sha,
        color_profile_status=profile_status,
        normalization_warnings=warnings,
    )