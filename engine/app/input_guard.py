"""Güvenli görüntü alımı — magic/format/boyut/piksel/bomb/animated + EXIF/ICC/alpha.

Canlı hatalar (giderilen): istemci Content-Type'a güvenme, byte/piksel sınırı yok,
decompression bomb korumasız, EXIF orientation/ICC normalize yok, şeffaflık kaybı.

Sözleşme: ``validate_and_load(contents, filename)`` -> ``LoadedImage`` ya da
``InputError(code, status)``. Hata kodları frontend ile ORTAK:
file_too_large, image_too_large, unsupported_format, animated_not_supported,
corrupt_image.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps
from PIL import UnidentifiedImageError

from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Pillow decompression-bomb sert tavanı: piksel sınırının biraz üstünde tutulur;
# .size (header) ön-kontrolü zaten decode ETMEDEN büyük görseli reddeder.
_SAFE_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


class InputError(Exception):
    """Kullanıcıya dönük güvenli alım hatası (kod + HTTP durumu)."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


@dataclass
class LoadedImage:
    image: Image.Image           # normalize edilmiş (EXIF/ICC/sRGB) PIL görüntüsü
    format: str                  # kaynak format (PNG/JPEG/WEBP)
    has_alpha: bool
    sha256: str                  # ham girdi baytları hash'i
    width: int
    height: int
    normalized: bool             # EXIF/ICC/mode dönüşümü pikselleri değiştirdi mi
    safe_suffix: str             # sunucu tarafından verilen güvenli uzantı


def _to_srgb(img: Image.Image) -> tuple[Image.Image, bool]:
    """CMYK/ICC kaynağı sRGB'ye normalize eder. Döner: (image, changed)."""
    icc = img.info.get("icc_profile")
    changed = False
    if img.mode == "CMYK" or icc:
        try:
            from PIL import ImageCms  # noqa: PLC0415
            if icc:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst = ImageCms.createProfile("sRGB")
                out_mode = "RGBA" if "A" in img.getbands() else "RGB"
                img = ImageCms.profileToProfile(img, src, dst, outputMode=out_mode)
                return img, True
        except Exception as e:  # noqa: BLE001
            logger.warning("ICC->sRGB dönüşümü atlandı: %s", e)
        if img.mode == "CMYK":
            return img.convert("RGB"), True
    return img, changed


def validate_and_load(contents: bytes, filename: str | None = None,
                      settings: Settings | None = None) -> LoadedImage:
    """Ham baytları güvenli biçimde doğrular ve normalize eder.

    Sıra: byte sınırı -> magic/format -> animated -> piksel sınırı (header,
    decode ETMEDEN) -> verify -> load -> EXIF transpose -> ICC/CMYK->sRGB.
    """
    s = settings or get_settings()

    # 1) byte sınırı (decode/piksel işinden ÖNCE)
    if len(contents) > s.max_upload_bytes:
        raise InputError("file_too_large", 413,
                         f"Dosya çok büyük (en fazla {s.max_upload_bytes // (1024 * 1024)} MiB).")
    if not contents:
        raise InputError("corrupt_image", 400, "Boş dosya.")

    # 2) magic + format (istemci Content-Type'a GÜVENME)
    try:
        header = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise InputError("unsupported_format", 415, "Tanınmayan/desteklenmeyen görüntü formatı.")
    except Exception:  # noqa: BLE001
        raise InputError("corrupt_image", 400, "Görsel dosyası bozuk veya okunamıyor.")

    fmt = (header.format or "").upper()
    if fmt not in s.allowed_formats:
        raise InputError("unsupported_format", 415,
                         f"Desteklenmeyen format: {fmt or 'bilinmiyor'}. "
                         f"İzin verilen: {sorted(s.allowed_formats)}.")

    # 3) animated / multi-frame (sessizce ilk kareye indirme YOK)
    n_frames = int(getattr(header, "n_frames", 1) or 1)
    if n_frames > 1:
        raise InputError("animated_not_supported", 415,
                         "Animasyonlu/çok kareli görseller desteklenmiyor.")

    # 4) piksel sınırı — HEADER'dan (decode ETMEDEN → bomb-güvenli)
    w, h = header.size
    if w <= 0 or h <= 0:
        raise InputError("corrupt_image", 400, "Geçersiz görsel boyutu.")
    if w > s.max_side or h > s.max_side or (w * h) > s.max_pixels:
        raise InputError("image_too_large", 413,
                         f"Görsel çok büyük ({w}x{h}); en fazla {s.max_side}px kenar / "
                         f"{s.max_pixels // 1_000_000} MP.")

    # 5) verify (bütünlük) — verify() nesneyi kullanılmaz bırakır → taze aç + load
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:  # noqa: BLE001
        raise InputError("corrupt_image", 400, "Görsel bütünlük doğrulaması başarısız.")

    try:
        img = Image.open(io.BytesIO(contents))
        img.load()
    except Image.DecompressionBombError:
        raise InputError("image_too_large", 413, "Görsel çözümleme sınırı aşıldı (decompression bomb).")
    except Image.DecompressionBombWarning:
        raise InputError("image_too_large", 413, "Görsel çözümleme sınırı aşıldı.")
    except Exception:  # noqa: BLE001
        raise InputError("corrupt_image", 400, "Görsel çözümlenemedi.")

    # 6) EXIF orientation
    before_size = img.size
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        pass
    exif_changed = img.size != before_size or bool(header.getexif().get(0x0112, 1) not in (1, None))

    # 7) şeffaflık tespiti + ICC/CMYK -> sRGB
    has_alpha = (img.mode in ("RGBA", "LA", "PA")
                 or (img.mode == "P" and "transparency" in img.info))
    img, icc_changed = _to_srgb(img)

    sha = hashlib.sha256(contents).hexdigest()
    return LoadedImage(
        image=img, format=fmt, has_alpha=has_alpha, sha256=sha,
        width=img.width, height=img.height,
        normalized=bool(exif_changed or icc_changed),
        safe_suffix=_SAFE_SUFFIX.get(fmt, ".png"))
