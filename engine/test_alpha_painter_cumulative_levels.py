"""Kümülatif paint-deficit kafesinin seyreltme merdiveni testleri.

Dikişi (seam_gap) kapatan TEK maske kodlaması kümülatif eşiktir; ayrık seviye
poligonları komşu seviye sınırında tasarımı gereği dikiş bırakır (bkz.
``_cumulative_threshold_children`` docstring'i). Kümülatif kodlama seviye başına
bir ``<path>`` yazdığı için bayt maliyeti kafes çözünürlüğüyle ölçeklenir:
yoğun alfa rampalı kaynaklarda sabit q24 kafesi bayt bütçesini aşıyor ve aday
journal geometri kapısına HİÇ ulaşamıyordu — yani seam reddi onarılamaz kalıyordu.

Bu testler, kafesin seyreltilebilir olduğunu ve seyreltmenin baytı gerçekten
düşürdüğünü sabitler. Kabul yetkisine dokunulmaz: seyreltme alfa sadakatini
düşürürse aday, değişmemiş alfa IoU/MAE ve TransformJournal kapılarında
reddedilir.

Testler sentetiktir; fixture adına veya vaka numarasına bağlı değildir.
"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_paint_deficit import (
    _ALPHA_LEVELS,
    _fixed_alpha_levels,
    build_paint_deficit_reconstruction_tree,
)
from app.alpha_candidate_painter import (
    _PAINT_DEFICIT_CUMULATIVE_LEVELS,
    _cumulative_threshold_children,
)

_SVG = "http://www.w3.org/2000/svg"


def _qname(name: str) -> str:
    return f"{{{_SVG}}}{name}"


def _dense_alpha_ramp(side: int = 320) -> np.ndarray:
    """Yoğun alfa rampası + ince yapı: kümülatif kodlamayı pahalı kılan kaynak."""
    yy, xx = np.mgrid[0:side, 0:side]
    centre = side / 2.0
    radius = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    alpha = np.clip(255.0 - radius * (255.0 / (centre * 0.9)), 0, 255)
    # İnce dikey yapı kontur sayısını artırır (gerçek logolardaki detay gibi).
    alpha[xx % 31 < 2] = np.clip(alpha[xx % 31 < 2] * 0.5, 0, 255)
    return alpha.astype(np.uint8)


def _cumulative_mask_bytes(alpha: np.ndarray, levels: int) -> int:
    quantized, opacity = _fixed_alpha_levels(alpha, levels)
    result = _cumulative_threshold_children(quantized, opacity, _qname)
    assert result is not None, f"kümülatif kontur üretilemedi: q{levels}"
    children, _stats = result
    return sum(len(ET.tostring(child)) for child in children)


class CumulativeLevelLadderTests(unittest.TestCase):
    def test_ladder_is_descending_and_coarser_than_default(self) -> None:
        ladder = _PAINT_DEFICIT_CUMULATIVE_LEVELS
        self.assertTrue(ladder, "merdiven boş olmamalı")
        self.assertEqual(
            list(ladder),
            sorted(ladder, reverse=True),
            "merdiven kabadan inceye değil, inceden kabaya azalmalı",
        )
        self.assertLess(
            max(ladder),
            _ALPHA_LEVELS,
            "merdivenin tamamı varsayılan q24 kafesinden daha kaba olmalı",
        )
        self.assertGreaterEqual(
            min(ladder), 2, "tek seviyeli (ikili) kafes alfa rampasını yok eder"
        )

    def test_coarser_lattice_strictly_reduces_cumulative_mask_bytes(self) -> None:
        alpha = _dense_alpha_ramp()
        baseline = _cumulative_mask_bytes(alpha, _ALPHA_LEVELS)
        previous = baseline
        for levels in _PAINT_DEFICIT_CUMULATIVE_LEVELS:
            current = _cumulative_mask_bytes(alpha, levels)
            self.assertLess(
                current,
                previous,
                f"q{levels} bir öncekinden küçük olmalı ({current} >= {previous})",
            )
            previous = current
        # Merdivenin en kaba basamağı anlamlı bir bütçe payı açmalı; aksi hâlde
        # bütçeyi %12 aşan bir adayı kurtaramaz.
        self.assertLess(
            previous,
            baseline * 0.6,
            "en kaba basamak q24'ün %60'ının altına inmeli",
        )

    def test_levels_argument_changes_encoded_lattice(self) -> None:
        alpha = _dense_alpha_ramp()
        for levels in (_ALPHA_LEVELS, *_PAINT_DEFICIT_CUMULATIVE_LEVELS):
            _quantized, opacity = _fixed_alpha_levels(alpha, levels)
            encoded = len([key for key in opacity if int(key) > 0])
            self.assertLessEqual(
                encoded,
                levels - 1,
                f"q{levels} kafesi {levels - 1} pozitif seviyeyi aşmamalı",
            )
            self.assertGreater(encoded, 0, f"q{levels} kafesi boş kalmamalı")

    def test_builder_accepts_levels_and_reports_encoded_count(self) -> None:
        """``levels`` üreticiye kadar taşınmalı ve ledger'a doğru yansımalı."""
        side = 96
        # Kaynak: tüm tuvale yayılan bağlı bir alfa rampası.
        alpha = _dense_alpha_ramp(side)
        alpha = np.maximum(alpha, 40)  # bağlı tek bileşen (anchor için)
        rgba = np.zeros((side, side, 4), dtype=np.uint8)
        rgba[:, :, 0] = 200
        rgba[:, :, 1] = 30
        rgba[:, :, 2] = 30
        rgba[:, :, 3] = alpha

        # Sanat eseri yalnız SOL yarıyı boyar → sağ yarı ölçülebilir bir
        # paint-deficit bırakır (kaynakta mürekkep var, render'da beyaz).
        root = ET.Element(
            _qname("svg"),
            {
                "xmlns": _SVG,
                "width": str(side),
                "height": str(side),
                "viewBox": f"0 0 {side} {side}",
            },
        )
        ET.SubElement(
            root,
            _qname("rect"),
            {
                "x": "0",
                "y": "0",
                "width": str(side // 2),
                "height": str(side),
                "fill": "#c81e1e",
            },
        )

        counts: list[int] = []
        for levels in (_ALPHA_LEVELS, min(_PAINT_DEFICIT_CUMULATIVE_LEVELS)):
            _tree, geometry = build_paint_deficit_reconstruction_tree(
                root,
                None,
                rgba,
                f"txn-q{levels}",
                mask_encoding="cumulative",
                levels=levels,
            )
            counts.append(int(geometry["encoded_alpha_level_count"]))
        self.assertGreater(
            counts[0],
            counts[1],
            "kaba kafes daha az kodlanmış seviye bildirmeli",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
