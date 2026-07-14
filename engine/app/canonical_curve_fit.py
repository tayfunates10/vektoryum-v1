"""HG-3 canonical shared-boundary fitting (SHADOW).

Fits each CanonicalBoundaryCurve exactly once. Twin half-edges continue to
reference the same curve object, so they cannot diverge geometrically.
Production serialization is intentionally unchanged in this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .half_edge_graph import CanonicalBoundaryCurve, Point, SharedBoundaryHalfEdgeGraph


@dataclass(frozen=True)
class CanonicalFitConfig:
    tolerance_px: float = 0.35
    max_error_px: float = 0.75
    linearity_epsilon_px: float = 0.20
    min_confidence: float = 0.50


@dataclass(frozen=True)
class CanonicalFitReport:
    fitted_curves: int
    fallback_curves: int
    line_curves: int
    cubic_curves: int
    max_error_px: float
    p95_error_px: float
    valid: bool
    errors: tuple[str, ...]


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def _rdp(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) <= 2:
        return points[:]
    a, b = points[0], points[-1]
    distances = [_point_segment_distance(p, a, b) for p in points[1:-1]]
    if not distances:
        return [a, b]
    index = int(np.argmax(distances)) + 1
    if distances[index - 1] <= epsilon:
        return [a, b]
    left = _rdp(points[: index + 1], epsilon)
    right = _rdp(points[index:], epsilon)
    return left[:-1] + right


def _sample_cubic(seg: tuple[Point, Point, Point, Point], n: int = 24) -> list[Point]:
    p0, c1, c2, p1 = seg
    out: list[Point] = []
    for i in range(n + 1):
        t = i / n; u = 1.0 - t
        x = u**3*p0[0] + 3*u*u*t*c1[0] + 3*u*t*t*c2[0] + t**3*p1[0]
        y = u**3*p0[1] + 3*u*u*t*c1[1] + 3*u*t*t*c2[1] + t**3*p1[1]
        out.append((x, y))
    return out


def _polyline_error(points: list[Point], fitted: list[Point]) -> tuple[float, float]:
    if not points or len(fitted) < 2:
        return float("inf"), float("inf")
    distances = []
    for p in points:
        distances.append(min(_point_segment_distance(p, fitted[i], fitted[i+1])
                             for i in range(len(fitted)-1)))
    return float(max(distances, default=0.0)), float(np.percentile(distances, 95))


def _fit_curve(curve: CanonicalBoundaryCurve, cfg: CanonicalFitConfig) -> None:
    pts = [(float(x), float(y)) for x, y in curve.polyline]
    if len(pts) < 2 or curve.confidence < cfg.min_confidence:
        curve.fitted_segments = []
        curve.command_count = max(0, len(pts) - 1)
        curve.primitive_kind = ""
        curve.fit_fallback = True
        curve.fit_error_max = 0.0
        curve.fit_error_p95 = 0.0
        return

    simplified = _rdp(pts, cfg.tolerance_px)
    line_err_max, line_err_p95 = _polyline_error(pts, [simplified[0], simplified[-1]])
    if line_err_max <= cfg.linearity_epsilon_px:
        p0, p1 = simplified[0], simplified[-1]
        curve.fitted_segments = [(p0, p0, p1, p1)]
        curve.command_count = 1
        curve.primitive_kind = "line"
        curve.fit_fallback = False
        curve.fit_error_max = line_err_max
        curve.fit_error_p95 = line_err_p95
        return

    segments: list[tuple[Point, Point, Point, Point]] = []
    for i in range(len(simplified) - 1):
        p0, p1 = simplified[i], simplified[i + 1]
        prev = simplified[i - 1] if i > 0 else p0
        nxt = simplified[i + 2] if i + 2 < len(simplified) else p1
        c1 = (p0[0] + (p1[0] - prev[0]) / 6.0,
              p0[1] + (p1[1] - prev[1]) / 6.0)
        c2 = (p1[0] - (nxt[0] - p0[0]) / 6.0,
              p1[1] - (nxt[1] - p0[1]) / 6.0)
        segments.append((p0, c1, c2, p1))

    sampled: list[Point] = []
    for seg in segments:
        s = _sample_cubic(seg)
        sampled.extend(s if not sampled else s[1:])
    err_max, err_p95 = _polyline_error(pts, sampled)
    if not np.isfinite(err_max) or err_max > cfg.max_error_px:
        curve.fitted_segments = []
        curve.command_count = max(0, len(simplified) - 1)
        curve.primitive_kind = ""
        curve.fit_fallback = True
    else:
        curve.fitted_segments = segments
        curve.command_count = len(segments)
        curve.primitive_kind = "cubic"
        curve.fit_fallback = False
    curve.fit_error_max = err_max
    curve.fit_error_p95 = err_p95


def fit_canonical_curves(graph: SharedBoundaryHalfEdgeGraph,
                         config: CanonicalFitConfig | None = None) -> CanonicalFitReport:
    """Fit every canonical curve once and fail closed on invariant violations."""
    cfg = config or CanonicalFitConfig()
    errors: list[str] = []
    all_errors: list[float] = []

    for curve_id in sorted(graph.curves):
        curve = graph.curves[curve_id]
        _fit_curve(curve, cfg)
        if not np.isfinite(curve.fit_error_max) or not np.isfinite(curve.fit_error_p95):
            errors.append(f"{curve_id}: non-finite fit error")
        if curve.fitted_segments:
            first = curve.fitted_segments[0][0]
            last = curve.fitted_segments[-1][3]
            start = graph.vertices[curve.start_vertex_id].point
            end = graph.vertices[curve.end_vertex_id].point
            if first != start or last != end:
                errors.append(f"{curve_id}: fitted endpoints are not locked vertices")
        all_errors.append(curve.fit_error_max)

    # Twin geometry is structurally shared through curve_id; assert both twins
    # still point to one canonical object after fitting.
    for edge in graph.half_edges.values():
        twin = graph.half_edges.get(edge.twin_id or "")
        if twin is None or twin.curve_id != edge.curve_id:
            errors.append(f"{edge.half_edge_id}: twin canonical curve mismatch")

    fallback = sum(1 for c in graph.curves.values() if c.fit_fallback)
    report = CanonicalFitReport(
        fitted_curves=len(graph.curves) - fallback,
        fallback_curves=fallback,
        line_curves=sum(1 for c in graph.curves.values() if c.primitive_kind == "line"),
        cubic_curves=sum(1 for c in graph.curves.values() if c.primitive_kind == "cubic"),
        max_error_px=float(max(all_errors, default=0.0)),
        p95_error_px=float(np.percentile(all_errors, 95)) if all_errors else 0.0,
        valid=not errors,
        errors=tuple(errors),
    )
    if errors:
        for curve in graph.curves.values():
            curve.fitted_segments = []
            curve.fit_fallback = True
            curve.primitive_kind = ""
    return report
