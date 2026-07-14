"""HG-7 promotion-readiness gate for the shadow SVG document.

This phase does not switch production serialization. It verifies that an HG-6
shadow document is internally consistent, deterministic across repeated builds,
and complete enough to be considered for a later production cutover.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import xml.etree.ElementTree as ET

from .shadow_svg_document import ShadowSvgDocumentReport


@dataclass(frozen=True)
class ShadowSvgPromotionGateReport:
    ready: bool
    checked_runs: int
    document_sha256: str
    face_count: int
    errors: tuple[str, ...]


def evaluate_shadow_svg_promotion_gate(
    reports: tuple[ShadowSvgDocumentReport, ...],
    *,
    minimum_runs: int = 3,
) -> ShadowSvgPromotionGateReport:
    """Return promotion readiness only when repeated HG-6 reports agree exactly."""
    errors: list[str] = []
    if isinstance(minimum_runs, bool) or not isinstance(minimum_runs, int) or minimum_runs < 2:
        errors.append("minimum_runs must be an integer greater than or equal to 2")
    if len(reports) < minimum_runs:
        errors.append(f"expected at least {minimum_runs} reports")
    if errors:
        return ShadowSvgPromotionGateReport(False, len(reports), "", 0, tuple(errors))

    baseline = reports[0]
    if not baseline.valid or not baseline.svg_text or not baseline.document_sha256:
        errors.append("baseline HG-6 report is invalid")

    for index, report in enumerate(reports):
        if not report.valid:
            errors.append(f"run {index}: HG-6 report is invalid")
            continue
        if report.errors:
            errors.append(f"run {index}: valid report contains errors")
        if not report.svg_text:
            errors.append(f"run {index}: document is empty")
            continue
        digest = sha256(report.svg_text.encode("utf-8")).hexdigest()
        if digest != report.document_sha256:
            errors.append(f"run {index}: document digest mismatch")
        if report.document_sha256 != baseline.document_sha256:
            errors.append(f"run {index}: nondeterministic document digest")
        if report.svg_text != baseline.svg_text:
            errors.append(f"run {index}: nondeterministic document payload")
        if report.width != baseline.width or report.height != baseline.height:
            errors.append(f"run {index}: document dimensions drifted")
        if report.face_count != baseline.face_count:
            errors.append(f"run {index}: face count drifted")
        try:
            root = ET.fromstring(report.svg_text)
        except ET.ParseError:
            errors.append(f"run {index}: malformed XML")
            continue
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            errors.append(f"run {index}: root element is not svg")
        if len(root) != report.face_count:
            errors.append(f"run {index}: XML face count mismatch")

    if baseline.face_count <= 0:
        errors.append("document has no faces")
    if baseline.width <= 0 or baseline.height <= 0:
        errors.append("document dimensions are invalid")

    if errors:
        return ShadowSvgPromotionGateReport(False, len(reports), "", 0, tuple(errors))
    return ShadowSvgPromotionGateReport(
        ready=True,
        checked_runs=len(reports),
        document_sha256=baseline.document_sha256,
        face_count=baseline.face_count,
        errors=(),
    )
