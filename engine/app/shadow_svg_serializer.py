"""HG-5 deterministic SVG path serializer (SHADOW).

Serializes HG-4 canonical face paths into auditable SVG path payloads without
changing the production serializer. Invalid graph/build input, unsupported
commands, non-finite coordinates or incomplete face coverage fail closed and
return no serialized faces.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import re

from .canonical_face_builder import CanonicalCyclePath, CanonicalFaceBuildReport
from .half_edge_graph import SharedBoundaryHalfEdgeGraph

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class ShadowSerializedFace:
    face_id: str
    fill_color: str
    fill_rule: str
    path_data: str
    z_order: int
    digest_sha256: str


@dataclass(frozen=True)
class ShadowSvgSerializationReport:
    faces: tuple[ShadowSerializedFace, ...]
    serialized_faces: int
    serialized_cycles: int
    valid: bool
    errors: tuple[str, ...]


def _format_number(value: float, precision: int) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite coordinate")
    rounded = round(float(value), precision)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


def _point(point: tuple[float, float], precision: int) -> str:
    return f"{_format_number(point[0], precision)} {_format_number(point[1], precision)}"


def _serialize_cycle(cycle: CanonicalCyclePath, precision: int) -> str:
    if not cycle.closed or cycle.start != cycle.end or not cycle.commands:
        raise ValueError(f"{cycle.cycle_id}: cycle is not closed")

    parts = [f"M {_point(cycle.start, precision)}"]
    current = cycle.start
    for kind, points in cycle.commands:
        if not points or points[0] != current:
            raise ValueError(f"{cycle.cycle_id}: discontinuous command stream")
        if kind == "L" and len(points) == 2:
            parts.append(f"L {_point(points[1], precision)}")
        elif kind == "C" and len(points) == 4:
            parts.append(
                "C "
                f"{_point(points[1], precision)} "
                f"{_point(points[2], precision)} "
                f"{_point(points[3], precision)}"
            )
        else:
            raise ValueError(f"{cycle.cycle_id}: unsupported command {kind}")
        current = points[-1]

    if current != cycle.start:
        raise ValueError(f"{cycle.cycle_id}: command stream is open")
    parts.append("Z")
    return " ".join(parts)


def serialize_shadow_svg_paths(
    graph: SharedBoundaryHalfEdgeGraph,
    build: CanonicalFaceBuildReport,
    *,
    precision: int = 4,
) -> ShadowSvgSerializationReport:
    """Serialize canonical face paths deterministically, or return no output."""
    errors: list[str] = []
    serialized: list[ShadowSerializedFace] = []
    cycle_count = 0

    if precision < 0 or precision > 9:
        errors.append("precision must be between 0 and 9")
    if not graph.valid:
        errors.append("graph is invalid")
    if not build.valid:
        errors.extend(build.errors or ("canonical face build is invalid",))

    expected_ids = {
        face_id for face_id, face in graph.faces.items()
        if face.visible and not face.is_exterior
    }
    actual_ids = {face.face_id for face in build.faces}
    if actual_ids != expected_ids:
        errors.append("visible face coverage mismatch")

    if errors:
        return ShadowSvgSerializationReport((), 0, 0, False, tuple(errors))

    try:
        ordered_faces = sorted(
            build.faces,
            key=lambda item: (graph.faces[item.face_id].z_order, item.face_id),
        )
        for face_path in ordered_faces:
            face = graph.faces.get(face_path.face_id)
            if face is None:
                raise ValueError(f"{face_path.face_id}: face missing")
            if not _HEX_COLOR.fullmatch(face.fill_color):
                raise ValueError(f"{face_path.face_id}: invalid fill color")
            if face_path.fill_rule != "evenodd":
                raise ValueError(f"{face_path.face_id}: unsupported fill rule")

            cycle_paths = (face_path.outer, *face_path.holes)
            path_data = " ".join(_serialize_cycle(cycle, precision) for cycle in cycle_paths)
            payload = f"{face_path.face_id}\n{face.fill_color.lower()}\nevenodd\n{path_data}"
            serialized.append(
                ShadowSerializedFace(
                    face_id=face_path.face_id,
                    fill_color=face.fill_color.lower(),
                    fill_rule="evenodd",
                    path_data=path_data,
                    z_order=face.z_order,
                    digest_sha256=sha256(payload.encode("utf-8")).hexdigest(),
                )
            )
            cycle_count += len(cycle_paths)
    except (KeyError, ValueError) as exc:
        return ShadowSvgSerializationReport((), 0, 0, False, (str(exc),))

    return ShadowSvgSerializationReport(
        faces=tuple(serialized),
        serialized_faces=len(serialized),
        serialized_cycles=cycle_count,
        valid=True,
        errors=(),
    )
