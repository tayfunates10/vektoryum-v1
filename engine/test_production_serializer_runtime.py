from hashlib import sha256
from pathlib import Path

from app.controlled_svg_cutover import ControlledSvgCutoverReport
from app.production_serializer_runtime import publish_runtime_svg


LEGACY = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0Z"/></svg>'
CANONICAL = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M1 1Z"/></svg>'


def _cutover() -> ControlledSvgCutoverReport:
    digest = sha256(CANONICAL.encode()).hexdigest()
    return ControlledSvgCutoverReport(
        svg_text=CANONICAL,
        selected_path="canonical-half-edge",
        promoted=True,
        output_sha256=digest,
        errors=(),
    )


def test_disabled_runtime_publishes_exact_legacy(tmp_path: Path):
    legacy = tmp_path / "legacy.svg"
    destination = tmp_path / "final.svg"
    legacy.write_bytes(LEGACY)

    report = publish_runtime_svg(
        legacy_svg=legacy,
        destination=destination,
        cutover=_cutover(),
        environ={},
    )

    assert report.published
    assert report.selection.selected_path == "legacy"
    assert destination.read_bytes() == LEGACY


def test_digest_pinned_runtime_publishes_canonical(tmp_path: Path):
    legacy = tmp_path / "legacy.svg"
    destination = tmp_path / "final.svg"
    legacy.write_bytes(LEGACY)
    cutover = _cutover()

    report = publish_runtime_svg(
        legacy_svg=legacy,
        destination=destination,
        cutover=cutover,
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "true",
            "VEKTORYUM_CANONICAL_SVG_SHA256": cutover.output_sha256,
        },
    )

    assert report.published
    assert report.selection.promoted
    assert report.selection.selected_path == "canonical-half-edge"
    assert destination.read_text() == CANONICAL


def test_wrong_approved_digest_fails_closed_to_legacy(tmp_path: Path):
    legacy = tmp_path / "legacy.svg"
    destination = tmp_path / "final.svg"
    legacy.write_bytes(LEGACY)

    report = publish_runtime_svg(
        legacy_svg=legacy,
        destination=destination,
        cutover=_cutover(),
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "1",
            "VEKTORYUM_CANONICAL_SVG_SHA256": "0" * 64,
        },
    )

    assert report.published
    assert not report.selection.promoted
    assert report.selection.selected_path == "legacy"
    assert destination.read_bytes() == LEGACY


def test_missing_legacy_and_rejected_candidate_publish_nothing(tmp_path: Path):
    destination = tmp_path / "final.svg"
    report = publish_runtime_svg(
        legacy_svg=tmp_path / "missing.svg",
        destination=destination,
        cutover=_cutover(),
        environ={},
    )

    assert not report.published
    assert not destination.exists()
