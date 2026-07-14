"""HG-6 deterministic full SVG document assembler (SHADOW).

Builds a complete, auditable SVG document from the HG-5 serialized face report.
The production serializer remains untouched. Invalid dimensions, invalid HG-5
input, duplicate/out-of-order faces, malformed path payloads, or digest drift
fail closed and return no document.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import math
import re
import xml.etree.ElementTree as ET

from .shadow_svg_serializer import ShadowSerializedFace, ShadowSvgSerializationReport

_PATH_START = re.compile(r"^M(?:\s|$)")
_HEX_COLOR = re.compile(r"^#[0-9a-f]{6}$")


@dataclass(frozen=True)
class ShadowSvgDocumentReport:
    svg_text: str
    width: int
    height: int
    face_count: int
    document_sha256: str
    valid: bool
    errors: tuple[str, ...]


def _validate_face(face: ShadowSerializedFace) -> None:
    if not face.face_id:
        raise ValueError("face id is empty")
    if not _HEX_COLOR.fullmatch(face.fill_color):
        raise ValueError(f"{face.face_id}: invalid fill color")
    if face.fill_rule != "evenodd":
        raise ValueError(f"{face.face_id}: unsupported fill rule")
    if not face.path_data or not _PATH_START.match(face.path_data):
        raise ValueError(f"{face.face_id}: malformed path data")
    payload = f"{face.face_id}\n{face.fill_color}\nevenodd\n{face.path_data}"
    expected = sha256(payload.encode("utf-8")).hexdigest()
    if expected != face.digest_sha256:
        raise ValueError(f"{face.face_id}: digest mismatch")


def assemble_shadow_svg_document(
    serialization: ShadowSvgSerializationReport,
    *,
    width: int,
    height: int,
    geometry_version: int = 0,
) -> ShadowSvgDocumentReport:
    """Assemble an SVG document deterministically, or return no output."""
    errors: list[str] = []
    if not serialization.valid:
        errors.extend(serialization.errors or ("HG-5 serialization is invalid",))
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        errors.append("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        errors.append("height must be a positive integer")
    if isinstance(geometry_version, bool) or not isinstance(geometry_version, int) or geometry_version < 0:
        errors.append("geometry_version must be a non-negative integer")
    if serialization.serialized_faces != len(serialization.faces):
        errors.append("serialized face count mismatch")
    if not serialization.faces:
        errors.append("document has no faces")
    if errors:
        return ShadowSvgDocumentReport("", 0, 0, 0, "", False, tuple(errors))

    try:
        faces = tuple(serialization.faces)
        ids = [face.face_id for face in faces]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate face ids")
        expected_order = sorted(faces, key=lambda item: (item.z_order, item.face_id))
        if list(faces) != expected_order:
            raise ValueError("faces are not in canonical z-order")

        path_lines: list[str] = []
        for face in faces:
            _validate_face(face)
            path_lines.append(
                "  <path"
                f" id=\"{html.escape(face.face_id, quote=True)}\""
                f" d=\"{html.escape(face.path_data, quote=True)}\""
                f" fill=\"{face.fill_color}\""
                " fill-rule=\"evenodd\""
                f" data-sha256=\"{face.digest_sha256}\"/>"
            )

        body = "\n".join(path_lines)
        svg_text = (
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
            f"viewBox=\"0 0 {width} {height}\" data-geometry-version=\"{geometry_version}\">\n"
            f"{body}\n"
            "</svg>"
        )
        root = ET.fromstring(svg_text)
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise ValueError("root element is not svg")
        if len(root) != len(faces):
            raise ValueError("document face count mismatch")
        digest = sha256(svg_text.encode("utf-8")).hexdigest()
    except (ValueError, ET.ParseError) as exc:
        return ShadowSvgDocumentReport("", 0, 0, 0, "", False, (str(exc),))

    return ShadowSvgDocumentReport(
        svg_text=svg_text,
        width=width,
        height=height,
        face_count=len(faces),
        document_sha256=digest,
        valid=True,
        errors=(),
    )
