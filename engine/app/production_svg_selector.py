"""Fail-closed production SVG selection and atomic publication.

This module is the only supported bridge between the legacy production SVG and
HG-8 canonical half-edge output. Canonical output is accepted only when the
controlled cutover report is promoted, internally self-consistent, valid XML,
and explicitly enabled by the caller. Every rejected state preserves the exact
legacy bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

from .controlled_svg_cutover import ControlledSvgCutoverReport


@dataclass(frozen=True)
class ProductionSvgSelection:
    svg_bytes: bytes
    selected_path: str
    promoted: bool
    output_sha256: str
    errors: tuple[str, ...]


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _is_svg(payload: bytes) -> bool:
    try:
        root = ET.fromstring(payload)
    except (ET.ParseError, ValueError):
        return False
    return root.tag in {"svg", "{http://www.w3.org/2000/svg}svg"}


def select_production_svg(
    *,
    legacy_svg_bytes: bytes,
    cutover: ControlledSvgCutoverReport | None,
    enabled: bool = False,
) -> ProductionSvgSelection:
    """Return canonical bytes only after all production invariants pass.

    The exact legacy payload is returned for disabled, absent, malformed, or
    inconsistent canonical state. Empty/non-SVG legacy input is itself invalid
    and therefore returns no publishable bytes.
    """
    errors: list[str] = []
    if not isinstance(legacy_svg_bytes, bytes) or not legacy_svg_bytes:
        errors.append("legacy SVG bytes are empty")
        legacy_svg_bytes = b""
    elif not _is_svg(legacy_svg_bytes):
        errors.append("legacy payload is not valid SVG")

    if not isinstance(enabled, bool):
        errors.append("enabled must be a boolean")
        enabled = False
    if not enabled:
        errors.append("production canonical serializer is disabled")
    if cutover is None:
        errors.append("controlled cutover report is missing")

    candidate_bytes = b""
    if cutover is not None:
        if not cutover.promoted or cutover.selected_path != "canonical-half-edge":
            errors.append("controlled cutover did not promote canonical output")
        if cutover.errors:
            errors.append("controlled cutover contains errors")
        if not isinstance(cutover.svg_text, str) or not cutover.svg_text:
            errors.append("canonical SVG text is empty")
        else:
            candidate_bytes = cutover.svg_text.encode("utf-8")
            if not _is_svg(candidate_bytes):
                errors.append("canonical payload is not valid SVG")
            if _digest(candidate_bytes) != cutover.output_sha256:
                errors.append("canonical output digest mismatch")

    if errors:
        return ProductionSvgSelection(
            svg_bytes=legacy_svg_bytes if _is_svg(legacy_svg_bytes) else b"",
            selected_path="legacy",
            promoted=False,
            output_sha256=_digest(legacy_svg_bytes) if legacy_svg_bytes else "",
            errors=tuple(errors),
        )

    return ProductionSvgSelection(
        svg_bytes=candidate_bytes,
        selected_path="canonical-half-edge",
        promoted=True,
        output_sha256=_digest(candidate_bytes),
        errors=(),
    )


def atomic_publish_svg(selection: ProductionSvgSelection, destination: Path) -> None:
    """Atomically publish a validated selection without partial output."""
    if not selection.svg_bytes or not _is_svg(selection.svg_bytes):
        raise ValueError("selection has no valid SVG payload")
    if _digest(selection.svg_bytes) != selection.output_sha256:
        raise ValueError("selection digest mismatch")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(selection.svg_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
