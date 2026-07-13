"""Tek kanonik ayar/limit şeması (env-güdümlü, GÜVENLİ varsayılanlar).

Canlı hatalar: dosya byte/piksel sınırı ve gerçek MIME kontrolü yoktu;
VEKTORYUM_MAX_INPUT_SIDE canlıda 0'dı (sınırsız). Bu modül tüm sınırları TEK
yerde toplar; production'da tehlikeli 0/sınırsız varsayılan bırakmaz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


@dataclass(frozen=True)
class Settings:
    # yükleme byte sınırı (varsayılan 15 MiB)
    max_upload_bytes: int
    # tek kenar piksel sınırı (0 = sınırsız DEĞİL; güvenli üst sınır)
    max_side: int
    # toplam piksel sınırı (decompression bomb koruması, ~40 MP)
    max_pixels: int
    # kabul edilen görüntü formatları (Pillow format adları, büyük harf)
    allowed_formats: frozenset[str]
    # analiz/işleme için opsiyonel küçültme kenarı (0 = küçültme kapalı)
    processing_max_side: int

    @classmethod
    def load(cls) -> "Settings":
        allowed = os.environ.get("VEKTORYUM_ALLOWED_FORMATS", "PNG,JPEG,WEBP")
        return cls(
            max_upload_bytes=_int_env("VEKTORYUM_MAX_UPLOAD_BYTES", 15 * 1024 * 1024),
            max_side=_int_env("VEKTORYUM_MAX_IMAGE_SIDE", 12000),
            max_pixels=_int_env("VEKTORYUM_MAX_IMAGE_PIXELS", 40_000_000),
            allowed_formats=frozenset(f.strip().upper() for f in allowed.split(",") if f.strip()),
            processing_max_side=_int_env("VEKTORYUM_MAX_INPUT_SIDE", 0),
        )


def get_settings() -> Settings:
    """Her çağrıda env'den taze okur (test/deploy env değişimi anında etkili)."""
    return Settings.load()
