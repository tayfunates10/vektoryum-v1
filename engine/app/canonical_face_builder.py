"""HG-4 canonical face builder (SHADOW).

Builds deterministic, closed face-cycle command streams from the HG-1/HG-2
half-edge graph and HG-3 canonical curve fits. Production SVG serialization is
intentionally unchanged. Any invariant failure returns an invalid report and no
face paths, preserving fail-closed behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .half_edge_graph import Point, SharedBoundaryHalfEdgeGraph

CommandKind = Literal["L", "C"]
PathCommand = tuple[CommandKind, tuple[Point, ...]]


@dataclass(frozen=True)
class CanonicalCyclePath:
    cycle_id: str
    face_id: str
    commands: tuple[PathCommand, ...]
    start: Point
    end: Point
    closed: bool
    used_fallback: bool
    signed_area: float


@dataclass(frozen=True)
class CanonicalFacePath:
    face_id: str
    outer: CanonicalCyclePath
    holes: tuple[CanonicalCyclePath, ...]
    fill_rule: str = "evenodd"


@dataclass(frozen=True)
class CanonicalFaceBuildReport:
    faces: tuple[CanonicalFacePath, ...]
    built_faces: int
    built_cycles: int
    fallback_cycles: int
    valid: bool
    errors: tuple[str, ...]


def _curve_commands(graph: SharedBoundaryHalfEdgeGraph, half_edge_id: str) -> tuple[list[PathCommand], bool]:
    edge = graph.half_edges[half_edge_id]
    curve = graph.curves[edge.curve_id]
    fallback = curve.fit_fallback or not curve.fitted_segments

    if fallback:
        pts = list(curve.polyline)
        if edge.reversed:
            pts.reverse()
        return [("L", (pts[i], pts[i + 1])) for i in range(len(pts) - 1)], True

    segments = list(curve.fitted_segments)
    if not edge.reversed:
        return [("C", tuple(seg)) for seg in segments], False

    reversed_commands: list[PathCommand] = []
    for p0, c1, c2, p1 in reversed(segments):
        reversed_commands.append(("C", (p1, c2, c1, p0)))
    return reversed_commands, False


def _command_start(command: PathCommand) -> Point:
    return command[1][0]


def _command_end(command: PathCommand) -> Point:
    return command[1][-1]


def _build_cycle(graph: SharedBoundaryHalfEdgeGraph, cycle_id: str, errors: list[str]) -> CanonicalCyclePath | None:
    cycle = graph.cycles.get(cycle_id)
    if cycle is None:
        errors.append(f"{cycle_id}: cycle missing")
        return None
    if not cycle.closed or not cycle.half_edge_ids:
        errors.append(f"{cycle_id}: cycle is not closed")
        return None

    commands: list[PathCommand] = []
    used_fallback = False
    for half_edge_id in cycle.half_edge_ids:
        if half_edge_id not in graph.half_edges:
            errors.append(f"{cycle_id}: missing half-edge {half_edge_id}")
            return None
        part, fallback = _curve_commands(graph, half_edge_id)
        if not part:
            errors.append(f"{cycle_id}: empty curve command stream")
            return None
        if commands and _command_end(commands[-1]) != _command_start(part[0]):
            errors.append(f"{cycle_id}: discontinuity at {half_edge_id}")
            return None
        commands.extend(part)
        used_fallback = used_fallback or fallback

    start = _command_start(commands[0])
    end = _command_end(commands[-1])
    closed = start == end
    if not closed:
        errors.append(f"{cycle_id}: command stream is open")
        return None

    for command in commands:
        for point in command[1]:
            if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                errors.append(f"{cycle_id}: non-finite command coordinate")
                return None

    return CanonicalCyclePath(
        cycle_id=cycle_id,
        face_id=cycle.face_id,
        commands=tuple(commands),
        start=start,
        end=end,
        closed=True,
        used_fallback=used_fallback,
        signed_area=cycle.signed_area,
    )


def build_canonical_face_paths(graph: SharedBoundaryHalfEdgeGraph) -> CanonicalFaceBuildReport:
    """Build one deterministic evenodd path model per visible graph face."""
    errors: list[str] = []
    built: list[CanonicalFacePath] = []
    fallback_cycles = 0
    built_cycles = 0

    if not graph.valid:
        errors.append("graph is invalid")

    for face_id in sorted(graph.faces):
        face = graph.faces[face_id]
        if not face.visible or face.is_exterior:
            continue
        if face.outer_cycle_id is None:
            errors.append(f"{face_id}: outer cycle missing")
            continue

        outer = _build_cycle(graph, face.outer_cycle_id, errors)
        holes = tuple(
            path for cycle_id in sorted(face.inner_cycle_ids)
            if (path := _build_cycle(graph, cycle_id, errors)) is not None
        )
        if outer is None:
            continue
        if any(path.face_id != face_id for path in (outer, *holes)):
            errors.append(f"{face_id}: cycle ownership mismatch")
            continue
        if not outer.signed_area < 0:
            errors.append(f"{face_id}: outer cycle orientation invalid")
            continue
        if any(path.signed_area <= 0 for path in holes):
            errors.append(f"{face_id}: hole orientation invalid")
            continue

        built.append(CanonicalFacePath(face_id=face_id, outer=outer, holes=holes))
        built_cycles += 1 + len(holes)
        fallback_cycles += int(outer.used_fallback) + sum(int(h.used_fallback) for h in holes)

    expected_faces = sum(1 for face in graph.faces.values() if face.visible and not face.is_exterior)
    if len(built) != expected_faces:
        errors.append(f"visible face coverage mismatch: {len(built)}/{expected_faces}")

    if errors:
        return CanonicalFaceBuildReport(
            faces=(), built_faces=0, built_cycles=0, fallback_cycles=0,
            valid=False, errors=tuple(errors),
        )

    return CanonicalFaceBuildReport(
        faces=tuple(built), built_faces=len(built), built_cycles=built_cycles,
        fallback_cycles=fallback_cycles, valid=True, errors=(),
    )
