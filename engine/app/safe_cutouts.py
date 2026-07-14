"""Transactional topology gate for production ``shape_stacking=cutouts``.

The legacy pyclipper transform is valid for closed polygonal fill paths, but it
samples Bezier and arc geometry into dense line segments.  Production now uses
that transform only when every visible geometry element is already polygonal.
Curves, arcs, transforms, strokes, unsupported primitives, unavailable boolean
dependencies or invalid output fail closed: source bytes remain untouched and
the caller keeps the stacked document.
"""
from __future__ import annotations

from hashlib import sha256
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET

_ALLOWED_COMMANDS = frozenset("MLHVZmlhvz")
_CURVE_COMMANDS = frozenset("CQASTcqast")
_GEOMETRY_TAGS = {
    "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
    "image", "use", "text",
}
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_COMMAND_RE = re.compile(r"[A-Za-z]")
_MAX_COMMAND_MULTIPLIER = 8
_MAX_COMMAND_ABSOLUTE_GROWTH = 1000


def _digest(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _local_name(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def _path_commands(d: str) -> list[str]:
    return _COMMAND_RE.findall(d or "")


def _document_contract(path: Path) -> dict[str, Any]:
    """Return a strict polygonal-document contract or fail with reason codes."""
    try:
        root = ET.parse(str(path)).getroot()
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "reason": "xml_parse_failed", "error": str(exc)}

    path_count = 0
    command_count = 0
    reasons: list[str] = []
    curve_commands: list[str] = []
    for element in root.iter():
        tag = _local_name(element)
        if element.get("transform"):
            reasons.append("unsupported_transform")
        if tag in _GEOMETRY_TAGS and tag != "path":
            reasons.append(f"unsupported_geometry:{tag}")
        if tag != "path":
            continue

        path_count += 1
        d = element.get("d") or ""
        commands = _path_commands(d)
        command_count += len(commands)
        unsupported = [command for command in commands if command not in _ALLOWED_COMMANDS]
        curve_commands.extend(command for command in unsupported if command in _CURVE_COMMANDS)
        if unsupported:
            reasons.append("curve_preservation_unavailable" if curve_commands else "unsupported_path_command")
        if not commands or sum(command in "Mm" for command in commands) != sum(command in "Zz" for command in commands):
            reasons.append("open_or_unclosed_path")
        fill = (element.get("fill") or "").strip().lower()
        stroke = (element.get("stroke") or "").strip().lower()
        if fill in {"", "none"} or fill.startswith("url("):
            reasons.append("unsupported_fill_model")
        if stroke not in {"", "none"}:
            reasons.append("stroke_geometry_unsupported")
        for raw in _NUMBER_RE.findall(d):
            try:
                value = float(raw)
            except ValueError:
                reasons.append("invalid_coordinate")
                break
            if not math.isfinite(value):
                reasons.append("non_finite_coordinate")
                break

    if path_count < 2:
        reasons.append("insufficient_path_coverage")
    return {
        "valid": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "path_count": path_count,
        "command_count": command_count,
        "curve_commands": sorted(set(curve_commands)),
    }


def _validate_candidate(
    source_contract: dict[str, Any],
    candidate: Path,
    converter_status: str,
    source_sha256: str,
) -> tuple[bool, dict[str, Any]]:
    candidate_contract = _document_contract(candidate)
    reasons = list(candidate_contract.get("reasons") or [])
    if not candidate_contract.get("valid"):
        reasons.append("candidate_contract_failed")

    before_paths = int(source_contract.get("path_count", 0))
    after_paths = int(candidate_contract.get("path_count", 0))
    if after_paths <= 0 or after_paths > before_paths:
        reasons.append("path_coverage_mismatch")

    before_commands = int(source_contract.get("command_count", 0))
    after_commands = int(candidate_contract.get("command_count", 0))
    command_limit = max(
        before_commands * _MAX_COMMAND_MULTIPLIER,
        before_commands + _MAX_COMMAND_ABSOLUTE_GROWTH,
    )
    if after_commands > command_limit:
        reasons.append("unbounded_command_growth")

    candidate_sha256 = _digest(candidate)
    if converter_status == "no_change" and candidate_sha256 != source_sha256:
        reasons.append("no_change_digest_mismatch")

    report = {
        "candidate_sha256": candidate_sha256,
        "path_count_before": before_paths,
        "path_count_after": after_paths,
        "command_count_before": before_commands,
        "command_count_after": after_commands,
        "command_count_limit": command_limit,
        "coverage_complete": after_paths > 0 and after_paths <= before_paths,
        "curve_preserved": True,
        "topology_valid": not reasons,
        "validation_reasons": list(dict.fromkeys(reasons)),
    }
    return not reasons, report


def _atomic_publish(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".publish",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_safe_cutout_candidate(
    source_svg: Path,
    destination_svg: Path,
    converter: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Build one validated candidate or leave the exact source document intact."""
    source_svg = Path(source_svg)
    destination_svg = Path(destination_svg)
    source_sha256 = _digest(source_svg)
    source_contract = _document_contract(source_svg)
    base_report: dict[str, Any] = {
        "schema_version": "curve-safe-cutouts-v1",
        "source_sha256": source_sha256,
        "curve_preserved": True,
        "transactional": True,
        "source_contract": source_contract,
    }
    if not source_contract.get("valid"):
        destination_svg.unlink(missing_ok=True) if destination_svg != source_svg else None
        reasons = source_contract.get("reasons") or ["source_contract_failed"]
        return {
            **base_report,
            "status": "skipped",
            "reason": reasons[0],
            "reason_codes": reasons,
            "fallback": "stacked",
        }

    work_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_svg.parent,
            prefix=f".{destination_svg.name}.",
            suffix=".candidate.svg",
            delete=False,
        ) as handle:
            work_path = Path(handle.name)
        shutil.copyfile(source_svg, work_path)
        try:
            converter_report = converter(work_path)
        except Exception as exc:  # noqa: BLE001
            return {
                **base_report,
                "status": "failed",
                "reason": "converter_exception",
                "reason_codes": ["converter_exception"],
                "error": f"{type(exc).__name__}: {exc}",
                "fallback": "stacked",
            }

        converter_status = str((converter_report or {}).get("status", "failed"))
        if converter_status not in {"completed", "no_change"}:
            error = str((converter_report or {}).get("error") or "")
            reason = "dependency_unavailable" if "yok" in error.lower() else "converter_rejected"
            return {
                **base_report,
                "status": "skipped" if converter_status == "skipped" else "failed",
                "reason": reason,
                "reason_codes": [reason],
                "converter_report": converter_report,
                "fallback": "stacked",
            }

        valid, validation = _validate_candidate(
            source_contract,
            work_path,
            converter_status,
            source_sha256,
        )
        if not valid:
            return {
                **base_report,
                **validation,
                "status": "failed",
                "reason": "candidate_validation_failed",
                "reason_codes": validation["validation_reasons"],
                "converter_report": converter_report,
                "fallback": "stacked",
            }

        _atomic_publish(work_path, destination_svg)
        published_sha256 = _digest(destination_svg)
        if published_sha256 != validation["candidate_sha256"]:
            if destination_svg != source_svg:
                destination_svg.unlink(missing_ok=True)
            return {
                **base_report,
                **validation,
                "status": "failed",
                "reason": "published_digest_mismatch",
                "reason_codes": ["published_digest_mismatch"],
                "fallback": "stacked",
            }
        return {
            **base_report,
            **validation,
            "status": converter_status,
            "reason_codes": [],
            "converter_report": converter_report,
            "published_sha256": published_sha256,
        }
    finally:
        if work_path is not None:
            work_path.unlink(missing_ok=True)
        if _digest(source_svg) != source_sha256:
            raise RuntimeError("safe cutout transaction modified source bytes")


__all__ = ["build_safe_cutout_candidate"]
