"""Shadow half-edge graph orkestrasyonu + feature flag'ler (SHADOW).

Tam shadow akışı: canonical segmentation → region consolidation → half-edge
graph → canonical curve fit → graph serializer. Production preview/download/API
YOLUNA BAĞLI DEĞİLDİR; yalnız debug/regression/ölçüm içindir.

Feature flag'ler (varsayılan production'da KAPALI):
- VEKTORYUM_HALF_EDGE_SHADOW           — shadow graph kurulumu
- VEKTORYUM_REGION_CONSOLIDATION_SHADOW — consolidation adımı
- VEKTORYUM_CANONICAL_CURVE_SHADOW      — alt-piksel curve fit
- VEKTORYUM_GRAPH_SERIALIZER_SHADOW     — shadow SVG serileştirme

Güvenlik: ``build_shadow_graph_safe`` shadow hatasını YUTAR (loglar), None döner;
production isteğini asla başarısız yapmaz.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def half_edge_shadow_enabled() -> bool:
    return _flag("VEKTORYUM_HALF_EDGE_SHADOW")


def consolidation_shadow_enabled() -> bool:
    return _flag("VEKTORYUM_REGION_CONSOLIDATION_SHADOW")


def canonical_curve_shadow_enabled() -> bool:
    return _flag("VEKTORYUM_CANONICAL_CURVE_SHADOW")


def graph_serializer_shadow_enabled() -> bool:
    return _flag("VEKTORYUM_GRAPH_SERIALIZER_SHADOW")


@dataclass
class ShadowGraphResult:
    graph: Any
    fills_rgb: np.ndarray
    source_hash: str
    consolidation: Any = None
    fit_report: dict[str, Any] = field(default_factory=dict)
    svg: str | None = None
    svg_metrics: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        s = self.graph.stats()
        return {**s, "fit": self.fit_report, "svg": self.svg_metrics,
                "source_hash": self.source_hash}


def build_shadow_graph(source_rgb: np.ndarray,
                       fills_rgb: np.ndarray | None = None,
                       k: int = 6,
                       consolidate: bool = True,
                       fit: bool = True,
                       serialize: bool = True) -> ShadowGraphResult:
    """Tam shadow akışını çalıştırır (flag'lerden bağımsız; doğrudan çağrı).

    ``fills_rgb`` verilirse (production paleti) yeniden türetilmez. Determinist.
    """
    from app.graph_source import canonical_segmentation, fills_to_hex
    from app.half_edge_graph import build_half_edge_graph
    from app.region_consolidation import consolidate_regions
    from app.canonical_curve import fit_canonical_curves
    from app.graph_serializer import serialize_graph_svg

    h, w = source_rgb.shape[:2]
    labels, fills = canonical_segmentation(source_rgb, fills_rgb, k=k)
    source_hash = hashlib.blake2b(source_rgb.tobytes(), digest_size=16).hexdigest()

    cons = None
    used_labels = labels
    if consolidate:
        cons = consolidate_regions(labels, source_rgb, fills)
        used_labels = cons.consolidated_labels

    graph = build_half_edge_graph(used_labels, fills_hex=fills_to_hex(fills))

    res = ShadowGraphResult(graph=graph, fills_rgb=fills, source_hash=source_hash,
                            consolidation=cons)
    if fit:
        res.fit_report = fit_canonical_curves(graph, source_rgb, fills)
    if serialize:
        svg, metrics = serialize_graph_svg(graph, w, h, source_hash=source_hash)
        res.svg = svg
        res.svg_metrics = metrics
    return res


def build_shadow_graph_safe(source_rgb: np.ndarray, **kw) -> ShadowGraphResult | None:
    """Flag-korumalı, hata-yutan sarmalayıcı (production isteğini bozmaz)."""
    if not half_edge_shadow_enabled():
        return None
    try:
        return build_shadow_graph(
            source_rgb,
            consolidate=kw.pop("consolidate", consolidation_shadow_enabled()),
            fit=kw.pop("fit", canonical_curve_shadow_enabled()),
            serialize=kw.pop("serialize", graph_serializer_shadow_enabled()),
            **kw)
    except Exception as e:  # noqa: BLE001 — shadow asla production'ı düşürmez
        logger.warning("shadow graph başarısız (legacy korunur): %s", e)
        return None
