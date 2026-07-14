from hashlib import sha256

import pytest

from app.controlled_svg_cutover import ControlledSvgCutoverReport
from app.production_svg_selector import atomic_publish_svg, select_production_svg


LEGACY = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0Z"/></svg>'
CANONICAL = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 0 L1 1 Z"/></svg>'


def _promoted(svg_text: str = CANONICAL) -> ControlledSvgCutoverReport:
    digest = sha256(svg_text.encode("utf-8")).hexdigest()
    return ControlledSvgCutoverReport(
        svg_text=svg_text,
        selected_path="canonical-half-edge",
        promoted=True,
        output_sha256=digest,
        errors=(),
    )


def test_disabled_gate_preserves_exact_legacy_bytes():
    result = select_production_svg(legacy_svg_bytes=LEGACY, cutover=_promoted(), enabled=False)
    assert result.svg_bytes == LEGACY
    assert result.selected_path == "legacy"
    assert result.promoted is False
    assert result.output_sha256 == sha256(LEGACY).hexdigest()


def test_promoted_digest_pinned_candidate_is_selected():
    result = select_production_svg(legacy_svg_bytes=LEGACY, cutover=_promoted(), enabled=True)
    assert result.svg_bytes == CANONICAL.encode("utf-8")
    assert result.selected_path == "canonical-half-edge"
    assert result.promoted is True
    assert result.errors == ()


def test_digest_drift_fails_closed_to_legacy():
    report = _promoted()
    drifted = ControlledSvgCutoverReport(
        svg_text=report.svg_text,
        selected_path=report.selected_path,
        promoted=True,
        output_sha256="0" * 64,
        errors=(),
    )
    result = select_production_svg(legacy_svg_bytes=LEGACY, cutover=drifted, enabled=True)
    assert result.svg_bytes == LEGACY
    assert result.selected_path == "legacy"
    assert "canonical output digest mismatch" in result.errors


def test_malformed_candidate_fails_closed_to_legacy():
    result = select_production_svg(
        legacy_svg_bytes=LEGACY,
        cutover=_promoted("not-svg"),
        enabled=True,
    )
    assert result.svg_bytes == LEGACY
    assert "canonical payload is not valid SVG" in result.errors


def test_invalid_legacy_never_becomes_publishable_fallback():
    result = select_production_svg(legacy_svg_bytes=b"broken", cutover=None, enabled=False)
    assert result.svg_bytes == b""
    assert result.promoted is False


def test_atomic_publish_writes_exact_validated_bytes(tmp_path):
    selection = select_production_svg(legacy_svg_bytes=LEGACY, cutover=_promoted(), enabled=True)
    destination = tmp_path / "final.svg"
    atomic_publish_svg(selection, destination)
    assert destination.read_bytes() == CANONICAL.encode("utf-8")


def test_atomic_publish_rejects_digest_drift(tmp_path):
    selection = select_production_svg(legacy_svg_bytes=LEGACY, cutover=_promoted(), enabled=True)
    tampered = type(selection)(
        svg_bytes=selection.svg_bytes + b" ",
        selected_path=selection.selected_path,
        promoted=selection.promoted,
        output_sha256=selection.output_sha256,
        errors=selection.errors,
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        atomic_publish_svg(tampered, tmp_path / "final.svg")
