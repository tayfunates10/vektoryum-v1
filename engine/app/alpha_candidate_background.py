"""Color-agnostic comparison-background classification for source-alpha masks.

The renderer-native painter reconstruction previously required a proven opaque
**white** full-canvas child to knock out. Real logos have backgrounds of any
colour — white, black, coloured or gradient — or no background trace at all, so
that assumption crashed with ``canvas_not_proven`` on canvas-less artwork.

This module classifies a comparison background purely from render geometry, not
colour. A direct child of the SVG root is a comparison background when, rendered
alone, it simultaneously:

* fills almost all of the **source-transparent** region (it paints where the
  source has no content — the trace's transparent composite), and
* is border-connected (covers the outer frame ring), and
* covers a large fraction of the whole canvas.

Real artwork — even large or border-touching — stays inside the source-opaque
region, so it never fills the transparent region and is never mistaken for a
background. Three outcomes are separated safely:

* ``proven``  — exactly one border-connected background: knock out only it.
* ``absent``  — no element fills the transparent region: reconstruct source
  alpha over the unchanged candidate paint without any knockout.
* ``ambiguous`` — the background cannot be identified without guessing: the
  caller must fail closed and restore the original SVG byte-for-byte.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from app.alpha_candidate_knockout import (
    _RENDERABLE_TAGS,
    _local_name,
    _probe_root,
    _render_root,
)

# Geometri eşikleri (renkten bağımsız). Bir aday comparison-background sayılması
# için şeffaf bölgenin neredeyse tamamını doldurmalı, kenar halkasına bağlı olmalı
# ve tuvalin büyük kısmını kaplamalıdır. Eşikler ölçümle doğrulanır; gevşetilmez.
_COVERED_ALPHA = 128  # bir pikselin "kaplandı" sayılması için asgari alfa
_BACKGROUND_TRANSPARENT_FILL_MIN = 0.90
_BACKGROUND_BORDER_RING_MIN = 0.90
_BACKGROUND_CANVAS_COVERAGE_MIN = 0.85


def _border_ring_mask(height: int, width: int) -> np.ndarray:
    ring = np.zeros((height, width), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def _child_background_scores(
    root: ET.Element,
    child: ET.Element,
    transparent: np.ndarray,
    ring: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    """Return (transparent_fill, border_ring_coverage, canvas_coverage) or None."""
    rendered = _render_root(_probe_root(root, child), width, height)
    if rendered is None:
        return None
    alpha = np.asarray(rendered[:, :, 3], dtype=np.uint8)
    covered = alpha >= _COVERED_ALPHA
    transparent_count = int(np.count_nonzero(transparent))
    ring_count = int(np.count_nonzero(ring))
    if transparent_count == 0 or ring_count == 0:
        return None
    transparent_fill = float(np.count_nonzero(covered & transparent)) / transparent_count
    border_coverage = float(np.count_nonzero(covered & ring)) / ring_count
    canvas_coverage = float(np.count_nonzero(covered)) / covered.size
    return transparent_fill, border_coverage, canvas_coverage


def classify_comparison_background(
    root: ET.Element,
    source_eval: np.ndarray,
    eval_width: int,
    eval_height: int,
) -> tuple[str, ET.Element | None]:
    """Classify the comparison background color-agnostically.

    Returns ``("proven", element)``, ``("absent", None)`` or
    ``("ambiguous", None)``. Never raises for a missing canvas.
    """
    source_alpha = np.asarray(source_eval[:, :, 3], dtype=np.uint8)
    transparent = source_alpha == 0
    if not np.any(transparent):
        # Fully opaque source has no transparent region to fill, so there is no
        # comparison background to remove; reconstruct over the paint directly.
        return ("absent", None)

    ring = _border_ring_mask(eval_height, eval_width)

    # Yalnız 3 geometri kapısını da geçen elemanlar "güçlü background" adayıdır.
    # Şeffaf bölgeyi kısmen dolduran (eşik altı) elemanlar background sayılmaz ve
    # tek başına belirsizlik üretmez: knockout yapılmaz, Case B (knockout'suz
    # reconstruction) güvenli varsayılan olur. Belirsizlik yalnız BİRDEN ÇOK güçlü
    # tam-tuval background bulunduğunda (hangisinin kazınacağı gerçekten belirsiz)
    # ortaya çıkar; o zaman tahmin edilmez ve fail-closed kalınır.
    backgrounds: list[ET.Element] = []
    for child in list(root):
        if _local_name(str(child.tag)).lower() not in _RENDERABLE_TAGS:
            continue
        scores = _child_background_scores(
            root, child, transparent, ring, eval_width, eval_height
        )
        if scores is None:
            continue
        transparent_fill, border_coverage, canvas_coverage = scores
        if (
            transparent_fill >= _BACKGROUND_TRANSPARENT_FILL_MIN
            and border_coverage >= _BACKGROUND_BORDER_RING_MIN
            and canvas_coverage >= _BACKGROUND_CANVAS_COVERAGE_MIN
        ):
            backgrounds.append(child)

    if len(backgrounds) >= 2:
        return ("ambiguous", None)
    if len(backgrounds) == 1:
        return ("proven", backgrounds[0])
    return ("absent", None)
