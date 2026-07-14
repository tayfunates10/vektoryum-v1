from dataclasses import replace
from hashlib import sha256
import xml.etree.ElementTree as ET

from app.shadow_svg_document import assemble_shadow_svg_document
from app.shadow_svg_serializer import ShadowSerializedFace, ShadowSvgSerializationReport


def _face(face_id: str = "face-0001", z_order: int = 0, color: str = "#112233") -> ShadowSerializedFace:
    path = "M 0 0 L 4 0 L 4 4 L 0 4 L 0 0 Z"
    payload = f"{face_id}\n{color}\nevenodd\n{path}"
    return ShadowSerializedFace(
        face_id=face_id,
        fill_color=color,
        fill_rule="evenodd",
        path_data=path,
        z_order=z_order,
        digest_sha256=sha256(payload.encode("utf-8")).hexdigest(),
    )


def _report(*faces: ShadowSerializedFace) -> ShadowSvgSerializationReport:
    return ShadowSvgSerializationReport(
        faces=faces,
        serialized_faces=len(faces),
        serialized_cycles=len(faces),
        valid=True,
        errors=(),
    )


def test_document_is_deterministic_and_well_formed():
    source = _report(_face())
    first = assemble_shadow_svg_document(source, width=8, height=6, geometry_version=7)
    second = assemble_shadow_svg_document(source, width=8, height=6, geometry_version=7)

    assert first.valid
    assert first == second
    assert first.face_count == 1
    assert first.document_sha256 == sha256(first.svg_text.encode("utf-8")).hexdigest()
    root = ET.fromstring(first.svg_text)
    assert root.attrib["viewBox"] == "0 0 8 6"
    assert root.attrib["data-geometry-version"] == "7"


def test_document_preserves_canonical_z_order():
    low = _face("face-low", 1, "#112233")
    high = _face("face-high", 2, "#445566")
    result = assemble_shadow_svg_document(_report(low, high), width=4, height=4)

    assert result.valid
    assert result.svg_text.index('id="face-low"') < result.svg_text.index('id="face-high"')


def test_document_fails_closed_on_digest_drift():
    damaged = replace(_face(), digest_sha256="0" * 64)
    result = assemble_shadow_svg_document(_report(damaged), width=4, height=4)

    assert not result.valid
    assert result.svg_text == ""
    assert result.document_sha256 == ""
    assert "digest mismatch" in result.errors[0]


def test_document_fails_closed_on_noncanonical_order_and_duplicate_ids():
    low = _face("face-low", 1)
    high = _face("face-high", 2)
    out_of_order = assemble_shadow_svg_document(_report(high, low), width=4, height=4)
    duplicate = assemble_shadow_svg_document(_report(low, low), width=4, height=4)

    assert not out_of_order.valid
    assert out_of_order.svg_text == ""
    assert not duplicate.valid
    assert duplicate.svg_text == ""


def test_document_fails_closed_on_invalid_dimensions_or_upstream_report():
    face = _face()
    invalid_upstream = ShadowSvgSerializationReport((), 0, 0, False, ("upstream invalid",))

    assert not assemble_shadow_svg_document(_report(face), width=0, height=4).valid
    assert not assemble_shadow_svg_document(_report(face), width=4, height=True).valid
    result = assemble_shadow_svg_document(invalid_upstream, width=4, height=4)
    assert not result.valid
    assert result.svg_text == ""
    assert "upstream invalid" in result.errors


def test_document_escapes_face_identifier_without_mutating_digest_contract():
    face = _face('face-"quoted"')
    result = assemble_shadow_svg_document(_report(face), width=4, height=4)

    assert result.valid
    assert 'id="face-&quot;quoted&quot;"' in result.svg_text
    ET.fromstring(result.svg_text)
