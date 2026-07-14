"""Build a deterministic HG-2..HG-7 canonical SVG candidate from a raster.

This module is intentionally side-effect free: it never writes production files and
never selects the canonical document for publication. Any invalid input or upstream
invariant failure returns an invalid report with no SVG payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .canonical_curve_fit import CanonicalFitReport, fit_canonical_curves
from .canonical_face_builder import build_canonical_face_paths
from .half_edge_graph import SharedBoundaryHalfEdgeGraph, build_half_edge_graph
from .shadow_svg_document import ShadowSvgDocumentReport, assemble_shadow_svg_document
from .shadow_svg_promotion_gate import (
    ShadowSvgPromotionGateReport,
    evaluate_shadow_svg_promotion_gate,
)
from .shadow_svg_serializer import serialize_shadow_svg_paths


@dataclass(frozen=True)
class CanonicalSvgCandidateReport:
    document: ShadowSvgDocumentReport | None
    promotion: ShadowSvgPromotionGateReport | None
    fit: CanonicalFitReport | None
    graph_stats: dict[str, Any]
    palette_size: int
    valid: bool
    errors: tuple[str, ...]


def _invalid(*errors: str) -> CanonicalSvgCandidateReport:
    return CanonicalSvgCandidateReport(
        document=None,
        promotion=None,
        fit=None,
        graph_stats={},
        palette_size=0,
        valid=False,
        errors=tuple(errors),
    )


def _rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _palette_labels(image: Image.Image, max_colors: int) -> tuple[np.ndarray, list[str]]:
    rgb = _rgb_on_white(image)
    indexed = rgb.quantize(
        colors=max_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    labels = np.asarray(indexed, dtype=np.uint8)
    palette = indexed.getpalette() or []
    highest = int(labels.max(initial=0))
    fills: list[str] = []
    for index in range(highest + 1):
        offset = index * 3
        if offset + 2 >= len(palette):
            raise ValueError("quantized palette is incomplete")
        fills.append(
            f"#{int(palette[offset]):02x}{int(palette[offset + 1]):02x}"
            f"{int(palette[offset + 2]):02x}"
        )
    return labels, fills


def _shift_graph_to_source(graph: SharedBoundaryHalfEdgeGraph) -> None:
    """Remove the one-pixel exterior padding used by the crack-edge builder."""
    for vertex in graph.vertices.values():
        x, y = vertex.point
        vertex.point = (float(x) - 1.0, float(y) - 1.0)
    for curve in graph.curves.values():
        curve.polyline = [(float(x) - 1.0, float(y) - 1.0) for x, y in curve.polyline]


def build_canonical_svg_candidate(
    image: Image.Image,
    *,
    max_colors: int = 32,
    geometry_version: int = 1,
    repeat_runs: int = 3,
    max_pixels: int = 16_000_000,
) -> CanonicalSvgCandidateReport:
    """Build and promotion-check a canonical SVG candidate, or fail closed."""
    if not isinstance(image, Image.Image):
        return _invalid("image must be a PIL Image")
    if isinstance(max_colors, bool) or not isinstance(max_colors, int) or not 2 <= max_colors <= 64:
        return _invalid("max_colors must be an integer between 2 and 64")
    if isinstance(geometry_version, bool) or not isinstance(geometry_version, int) or geometry_version < 0:
        return _invalid("geometry_version must be a non-negative integer")
    if isinstance(repeat_runs, bool) or not isinstance(repeat_runs, int) or repeat_runs < 3:
        return _invalid("repeat_runs must be an integer greater than or equal to 3")
    if image.width <= 0 or image.height <= 0:
        return _invalid("image dimensions are invalid")
    if image.width * image.height > max_pixels:
        return _invalid("image exceeds canonical candidate pixel budget")

    try:
        labels, fills = _palette_labels(image, max_colors)
        graph = build_half_edge_graph(labels, fills, geometry_version=geometry_version)
        if not graph.valid:
            return CanonicalSvgCandidateReport(
                None, None, None, graph.stats(), len(fills), False,
                tuple(graph.validation_errors or ("half-edge graph is invalid",)),
            )

        _shift_graph_to_source(graph)
        fit = fit_canonical_curves(graph)
        if not fit.valid:
            return CanonicalSvgCandidateReport(
                None, None, fit, graph.stats(), len(fills), False,
                tuple(fit.errors or ("canonical curve fitting is invalid",)),
            )

        faces = build_canonical_face_paths(graph)
        if not faces.valid:
            return CanonicalSvgCandidateReport(
                None, None, fit, graph.stats(), len(fills), False,
                tuple(faces.errors or ("canonical face build is invalid",)),
            )

        serialization = serialize_shadow_svg_paths(graph, faces)
        if not serialization.valid:
            return CanonicalSvgCandidateReport(
                None, None, fit, graph.stats(), len(fills), False,
                tuple(serialization.errors or ("canonical serialization is invalid",)),
            )

        reports = tuple(
            assemble_shadow_svg_document(
                serialization,
                width=image.width,
                height=image.height,
                geometry_version=geometry_version,
            )
            for _ in range(repeat_runs)
        )
        promotion = evaluate_shadow_svg_promotion_gate(reports, minimum_runs=repeat_runs)
        document = reports[0]
        if not document.valid or not promotion.ready:
            errors = tuple(document.errors) + tuple(promotion.errors)
            return CanonicalSvgCandidateReport(
                None, promotion, fit, graph.stats(), len(fills), False,
                errors or ("canonical candidate promotion failed",),
            )

        return CanonicalSvgCandidateReport(
            document=document,
            promotion=promotion,
            fit=fit,
            graph_stats=graph.stats(),
            palette_size=len(fills),
            valid=True,
            errors=(),
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed boundary
        return _invalid(f"canonical candidate build failed: {type(exc).__name__}: {exc}")
