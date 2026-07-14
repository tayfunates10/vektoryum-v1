"""HG-8 controlled production cutover for canonical half-edge SVG output.

The new serializer is selected only when an operator explicitly enables it and
an HG-7 promotion report exactly matches the approved document digest. Every
other condition fails closed to the caller-provided legacy SVG, preserving the
existing production path by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

from .shadow_svg_document import ShadowSvgDocumentReport
from .shadow_svg_promotion_gate import ShadowSvgPromotionGateReport

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ControlledSvgCutoverReport:
    svg_text: str
    selected_path: str
    promoted: bool
    output_sha256: str
    errors: tuple[str, ...]


def select_controlled_svg_output(
    *,
    legacy_svg_text: str,
    candidate: ShadowSvgDocumentReport,
    promotion: ShadowSvgPromotionGateReport,
    cutover_enabled: bool = False,
    approved_document_sha256: str = "",
) -> ControlledSvgCutoverReport:
    """Select canonical output only after explicit, digest-pinned promotion.

    Invalid or incomplete promotion state never removes the production output:
    the function returns the legacy document and records the rejection reasons.
    """
    errors: list[str] = []

    if not isinstance(legacy_svg_text, str) or not legacy_svg_text:
        errors.append("legacy SVG is empty")
    if not isinstance(cutover_enabled, bool):
        errors.append("cutover_enabled must be a boolean")
    if not isinstance(approved_document_sha256, str):
        errors.append("approved_document_sha256 must be a string")
        approved_document_sha256 = ""

    if not cutover_enabled:
        errors.append("canonical cutover is disabled")
    if not _SHA256.fullmatch(approved_document_sha256):
        errors.append("approved document digest is invalid")
    if not promotion.ready or promotion.errors:
        errors.append("HG-7 promotion gate is not ready")
    if not candidate.valid or candidate.errors or not candidate.svg_text:
        errors.append("HG-6 candidate document is invalid")

    if candidate.valid and candidate.svg_text:
        candidate_digest = sha256(candidate.svg_text.encode("utf-8")).hexdigest()
        if candidate_digest != candidate.document_sha256:
            errors.append("candidate document digest mismatch")
        if promotion.document_sha256 != candidate.document_sha256:
            errors.append("promotion digest does not match candidate")
        if approved_document_sha256 != candidate.document_sha256:
            errors.append("approved digest does not match candidate")
        if promotion.face_count != candidate.face_count:
            errors.append("promotion face count does not match candidate")

    if errors:
        legacy_digest = (
            sha256(legacy_svg_text.encode("utf-8")).hexdigest()
            if isinstance(legacy_svg_text, str) and legacy_svg_text
            else ""
        )
        return ControlledSvgCutoverReport(
            svg_text=legacy_svg_text if isinstance(legacy_svg_text, str) else "",
            selected_path="legacy",
            promoted=False,
            output_sha256=legacy_digest,
            errors=tuple(errors),
        )

    return ControlledSvgCutoverReport(
        svg_text=candidate.svg_text,
        selected_path="canonical-half-edge",
        promoted=True,
        output_sha256=candidate.document_sha256,
        errors=(),
    )
