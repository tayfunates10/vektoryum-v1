from dataclasses import replace
from hashlib import sha256

from app.controlled_svg_cutover import select_controlled_svg_output
from app.shadow_svg_document import ShadowSvgDocumentReport
from app.shadow_svg_promotion_gate import ShadowSvgPromotionGateReport


LEGACY = '<svg xmlns="http://www.w3.org/2000/svg"><path id="legacy"/></svg>'
CANONICAL = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
    'viewBox="0 0 16 16" data-geometry-version="8">\n'
    '  <path id="face-0" d="M 0 0 L 16 0 L 16 16 L 0 16 Z" '
    'fill="#112233" fill-rule="evenodd" data-sha256="abc"/>\n'
    '</svg>'
)
DIGEST = sha256(CANONICAL.encode("utf-8")).hexdigest()


def candidate() -> ShadowSvgDocumentReport:
    return ShadowSvgDocumentReport(
        svg_text=CANONICAL,
        width=16,
        height=16,
        face_count=1,
        document_sha256=DIGEST,
        valid=True,
        errors=(),
    )


def promotion() -> ShadowSvgPromotionGateReport:
    return ShadowSvgPromotionGateReport(
        ready=True,
        checked_runs=3,
        document_sha256=DIGEST,
        face_count=1,
        errors=(),
    )


def test_cutover_promotes_only_exact_digest_pinned_candidate():
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=candidate(),
        promotion=promotion(),
        cutover_enabled=True,
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is True
    assert report.selected_path == "canonical-half-edge"
    assert report.svg_text == CANONICAL
    assert report.output_sha256 == DIGEST
    assert report.errors == ()


def test_cutover_is_disabled_by_default_and_preserves_legacy_output():
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=candidate(),
        promotion=promotion(),
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is False
    assert report.selected_path == "legacy"
    assert report.svg_text == LEGACY
    assert "canonical cutover is disabled" in report.errors


def test_cutover_fails_closed_on_approved_digest_drift():
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=candidate(),
        promotion=promotion(),
        cutover_enabled=True,
        approved_document_sha256="0" * 64,
    )

    assert report.promoted is False
    assert report.svg_text == LEGACY
    assert "approved digest does not match candidate" in report.errors


def test_cutover_fails_closed_when_promotion_gate_is_not_ready():
    blocked = replace(promotion(), ready=False, errors=("blocked",))
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=candidate(),
        promotion=blocked,
        cutover_enabled=True,
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is False
    assert report.svg_text == LEGACY
    assert "HG-7 promotion gate is not ready" in report.errors


def test_cutover_fails_closed_on_candidate_payload_tampering():
    tampered = replace(candidate(), svg_text=CANONICAL + "\n")
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=tampered,
        promotion=promotion(),
        cutover_enabled=True,
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is False
    assert report.svg_text == LEGACY
    assert "candidate document digest mismatch" in report.errors


def test_cutover_fails_closed_on_face_count_drift():
    drifted = replace(promotion(), face_count=2)
    report = select_controlled_svg_output(
        legacy_svg_text=LEGACY,
        candidate=candidate(),
        promotion=drifted,
        cutover_enabled=True,
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is False
    assert report.svg_text == LEGACY
    assert "promotion face count does not match candidate" in report.errors


def test_invalid_legacy_output_never_promotes_silently():
    report = select_controlled_svg_output(
        legacy_svg_text="",
        candidate=candidate(),
        promotion=promotion(),
        cutover_enabled=True,
        approved_document_sha256=DIGEST,
    )

    assert report.promoted is False
    assert report.svg_text == ""
    assert "legacy SVG is empty" in report.errors
