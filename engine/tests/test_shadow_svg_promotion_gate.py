from dataclasses import replace
from hashlib import sha256

from app.shadow_svg_document import ShadowSvgDocumentReport
from app.shadow_svg_promotion_gate import evaluate_shadow_svg_promotion_gate


def _report(svg_text: str = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4" viewBox="0 0 4 4"><path id="f" d="M 0 0 Z" fill="#112233" fill-rule="evenodd"/></svg>') -> ShadowSvgDocumentReport:
    return ShadowSvgDocumentReport(
        svg_text=svg_text,
        width=4,
        height=4,
        face_count=1,
        document_sha256=sha256(svg_text.encode("utf-8")).hexdigest(),
        valid=True,
        errors=(),
    )


def test_gate_accepts_three_identical_valid_runs():
    report = _report()
    result = evaluate_shadow_svg_promotion_gate((report, report, report))

    assert result.ready
    assert result.checked_runs == 3
    assert result.document_sha256 == report.document_sha256
    assert result.face_count == 1
    assert result.errors == ()


def test_gate_fails_closed_on_too_few_runs():
    result = evaluate_shadow_svg_promotion_gate((_report(),), minimum_runs=3)

    assert not result.ready
    assert result.document_sha256 == ""
    assert result.face_count == 0
    assert "expected at least 3 reports" in result.errors


def test_gate_fails_closed_on_digest_or_payload_drift():
    baseline = _report()
    changed = _report('<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4" viewBox="0 0 4 4"><path id="g" d="M 0 0 Z" fill="#112233" fill-rule="evenodd"/></svg>')
    damaged = replace(baseline, document_sha256="0" * 64)

    drift = evaluate_shadow_svg_promotion_gate((baseline, baseline, changed))
    mismatch = evaluate_shadow_svg_promotion_gate((baseline, baseline, damaged))

    assert not drift.ready
    assert any("nondeterministic" in error for error in drift.errors)
    assert not mismatch.ready
    assert any("digest mismatch" in error for error in mismatch.errors)


def test_gate_fails_closed_on_invalid_report_and_metadata_drift():
    baseline = _report()
    invalid = replace(baseline, valid=False, svg_text="", document_sha256="", errors=("invalid",))
    dimensions = replace(baseline, width=5)
    faces = replace(baseline, face_count=2)

    invalid_result = evaluate_shadow_svg_promotion_gate((baseline, baseline, invalid))
    metadata_result = evaluate_shadow_svg_promotion_gate((baseline, dimensions, faces))

    assert not invalid_result.ready
    assert any("HG-6 report is invalid" in error for error in invalid_result.errors)
    assert not metadata_result.ready
    assert any("dimensions drifted" in error for error in metadata_result.errors)
    assert any("face count drifted" in error for error in metadata_result.errors)


def test_gate_rejects_invalid_configuration_and_malformed_xml():
    malformed = _report("<svg")

    bad_config = evaluate_shadow_svg_promotion_gate((_report(), _report()), minimum_runs=True)
    bad_xml = evaluate_shadow_svg_promotion_gate((malformed, malformed, malformed))

    assert not bad_config.ready
    assert "minimum_runs must be an integer greater than or equal to 2" in bad_config.errors
    assert not bad_xml.ready
    assert any("malformed XML" in error for error in bad_xml.errors)
