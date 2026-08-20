"""Siluet kısayolunun sihirli sayı yerine çağıran niyetiyle seçilmesi.

Üretimde ``alpha_candidate_identity``, ``painter._requantize_alpha``'yı bir
sarmalayıcıyla değiştirir; sarmalayıcı belirli bir kafes için tam nicemleme
yerine ikili siluet döndürür. Kısayol contour merdiveni için KASITLIDIR.

Ancak seçim ``max_levels`` DEĞERİNE bağlıydı: aynı kafes değerini kullanan her
yeni tüketici kısayola sessizce yakalanıyordu. Ölçüldü (public-04, run
31872027946): düğüm eklemeyen polygon merdiveninin orta basamağı 31 seviye
yerine 1 seviye üretti, bayt monotonluğu kırıldı (976 K → 144 K → 778 K) ve
#162'nin canlı etkisi ölçülemez hâle geldi.

Bu testler seçimin niyete bağlı kaldığını sabitler.
"""
from __future__ import annotations

import unittest

import numpy as np

from app.alpha_candidate_painter import (
    _NODE_FREE_QUANTIZED_LEVELS,
    _requantize_alpha,
)

_SILHOUETTE_LEVELS = 32


def _dense_alpha() -> np.ndarray:
    """255 farklı pozitif alfa: üretimdeki public-04 imzası."""
    return np.tile(np.arange(0, 256, dtype=np.uint8), (64, 1))


def _positive_levels(opacity: dict[int, float]) -> int:
    return len([key for key in opacity if int(key) > 0])


def _production_style_wrapper(original):
    """Üretimdeki sarmalayıcının davranışını taklit eder."""

    def wrapped(alpha, max_levels, *, allow_silhouette_shortcut=True):
        if allow_silhouette_shortcut and int(max_levels) == _SILHOUETTE_LEVELS:
            quantized = (np.asarray(alpha, dtype=np.uint8) > 0).astype(np.int32)
            return quantized, {1: 1.0}
        return original(
            alpha, max_levels, allow_silhouette_shortcut=allow_silhouette_shortcut
        )

    return wrapped


class SilhouetteShortcutIntentTests(unittest.TestCase):
    def test_quantizer_accepts_and_ignores_flag_when_unpatched(self) -> None:
        alpha = _dense_alpha()
        for allow in (True, False):
            _q, opacity = _requantize_alpha(
                alpha, _SILHOUETTE_LEVELS, allow_silhouette_shortcut=allow
            )
            self.assertEqual(
                _positive_levels(opacity),
                _SILHOUETTE_LEVELS - 1,
                "yamasız nicemleyici her iki bayrakta da tam kafes üretmeli",
            )

    def test_wrapper_still_serves_silhouette_when_requested(self) -> None:
        """Contour merdiveninin kasıtlı kısayolu korunmalı."""
        wrapped = _production_style_wrapper(_requantize_alpha)
        _q, opacity = wrapped(_dense_alpha(), _SILHOUETTE_LEVELS)
        self.assertEqual(
            _positive_levels(opacity), 1, "kısayol istendiğinde siluet gelmeli"
        )

    def test_wrapper_honours_opt_out_at_the_shortcut_value(self) -> None:
        """Asıl regresyon: aynı kafes değerinde niyet 'hayır' ise tam nicemleme."""
        wrapped = _production_style_wrapper(_requantize_alpha)
        _q, opacity = wrapped(
            _dense_alpha(), _SILHOUETTE_LEVELS, allow_silhouette_shortcut=False
        )
        self.assertEqual(
            _positive_levels(opacity),
            _SILHOUETTE_LEVELS - 1,
            "kısayol reddedildiğinde değer 32 olsa bile tam kafes gelmeli",
        )

    def test_node_free_ladder_level_counts_are_monotonic_under_wrapper(self) -> None:
        """Merdiven inceden kabaya gider: seviye sayısı ARTMAMALI."""
        wrapped = _production_style_wrapper(_requantize_alpha)
        alpha = _dense_alpha()
        counts = []
        for target in _NODE_FREE_QUANTIZED_LEVELS:
            _q, opacity = wrapped(alpha, target, allow_silhouette_shortcut=False)
            counts.append(_positive_levels(opacity))
        self.assertEqual(
            counts,
            sorted(counts, reverse=True),
            f"merdiven seviye sayıları monoton azalmalı, bulunan: {counts}",
        )
        self.assertNotIn(
            1,
            counts,
            "hiçbir basamak ikili siluete çökmemeli (üretimdeki kusur buydu)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
