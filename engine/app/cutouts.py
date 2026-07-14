"""Shape stacking conversion: stacked -> safe cut-outs.

The pyclipper implementation remains available for already-polygonal, closed,
fill-only paths. The public production entry point never samples Bezier or arc
geometry. A finite path-level translate on a polygon is normalized analytically;
all other transforms, unsupported geometry, missing dependencies or invalid
post-transform topology leave the exact stacked bytes unchanged.
"""
from __future__ import annotations

from hashlib import sha256
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
_SCALE = 100.0
_OVERLAP_PX = 0.25
_FLATTEN_STEP = 0.8
_LINEAR_COMMANDS = frozenset("MLHVZmlhvz")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TRANSLATE_RE = re.compile(
    rf"translate\(\s*({_NUMBER})(?:\s*[, ]\s*({_NUMBER}))?\s*\)",
    re.IGNORECASE,
)

try:
    import pyclipper
except ImportError:  # pragma: no cover
    pyclipper = None

try:
    from svgpathtools import parse_path
    from svgpathtools.path import transform as transform_path
except ImportError:  # pragma: no cover
    parse_path = None
    transform_path = None


def is_available() -> bool:
    return pyclipper is not None and parse_path is not None and transform_path is not None


def _flatten_to_rings(
    d: str, xf: tuple[float, float, float, float, float, float] | None = None
) -> list[list[tuple[int, int]]] | None:
    """Flatten already-linear closed paths for the private boolean backend."""
    try:
        rings: list[list[tuple[int, int]]] = []
        for sub in parse_path(d).continuous_subpaths():
            try:
                if not sub.isclosed():
                    return None
                length = sub.length()
            except Exception:  # noqa: BLE001
                return None
            n = int(max(8, min(4000, (length or 8) / _FLATTEN_STEP)))
            ring = []
            for i in range(n):
                point = sub.point(i / n)
                x, y = point.real, point.imag
                if xf is not None:
                    a, b, c, dd, e, f = xf
                    x, y = a * x + c * y + e, b * x + dd * y + f
                ring.append((int(round(x * _SCALE)), int(round(y * _SCALE))))
            dedup = [ring[0]]
            for point in ring[1:]:
                if point != dedup[-1]:
                    dedup.append(point)
            if len(dedup) >= 3:
                rings.append(dedup)
        return rings if rings else None
    except Exception:  # noqa: BLE001
        return None


def _rings_to_d(rings: list[list[tuple[int, int]]]) -> str:
    parts: list[str] = []
    for ring in rings:
        points = [(x / _SCALE, y / _SCALE) for x, y in ring]
        parts.append(f"M {points[0][0]:.2f} {points[0][1]:.2f}")
        parts.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
        parts.append("Z")
    return " ".join(parts)


def _union(subject: list, clip: list) -> list:
    if not subject:
        return [list(ring) for ring in clip]
    if not clip:
        return subject
    operation = pyclipper.Pyclipper()
    operation.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    operation.AddPaths(clip, pyclipper.PT_CLIP, True)
    return operation.Execute(
        pyclipper.CT_UNION,
        pyclipper.PFT_NONZERO,
        pyclipper.PFT_NONZERO,
    )


def _difference(subject: list, clip: list) -> list:
    if not clip:
        return subject
    operation = pyclipper.Pyclipper()
    operation.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    operation.AddPaths(clip, pyclipper.PT_CLIP, True)
    return operation.Execute(
        pyclipper.CT_DIFFERENCE,
        pyclipper.PFT_EVENODD,
        pyclipper.PFT_NONZERO,
    )


def _inset(rings: list, delta_px: float) -> list:
    if not rings:
        return rings
    offset = pyclipper.PyclipperOffset()
    offset.AddPaths(rings, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(-delta_px * _SCALE)


def _convert_svg_to_cutouts_polygonal(svg_path: Path) -> dict[str, Any]:
    """Private boolean implementation; caller has already rejected curves."""
    if not is_available():
        return {"status": "skipped", "error": "pyclipper/svgpathtools yok"}
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}

    items: list[tuple[Any, Any, list | None]] = []
    for parent in root.iter():
        parent_has_transform = parent.get("transform") is not None
        for element in list(parent):
            if element.tag.split("}")[-1] != "path":
                continue
            d = element.get("d")
            if parent_has_transform:
                items.append((parent, element, None))
                continue
            rings = _flatten_to_rings(d) if d else None
            items.append((parent, element, rings))
    if len(items) < 2:
        return {"status": "no_change", "paths": len(items)}

    try:
        upper_union: list = []
        removed = 0
        changed = 0
        for parent, element, rings in reversed(items):
            if rings is None:
                continue
            visible = _difference(rings, _inset(upper_union, _OVERLAP_PX))
            if not visible:
                parent.remove(element)
                removed += 1
            else:
                new_d = _rings_to_d(visible)
                if new_d:
                    element.set("d", new_d)
                    element.set("fill-rule", "evenodd")
                    changed += 1
            upper_union = _union(upper_union, rings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cut-outs dönüşümü başarısız, stacked korunuyor: %s", exc)
        return {"status": "failed", "error": str(exc)}

    try:
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}
    return {"status": "completed", "paths_changed": changed, "paths_removed": removed}


def _normalize_polygon_translates(source: Path, destination: Path) -> bool | None:
    """Normalize only finite path-level translate transforms.

    Returns ``True`` when a normalized copy was written, ``False`` when no
    transforms exist, and ``None`` for unsupported/group/curved transforms.
    """
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(source))
        root = tree.getroot()
    except Exception:  # noqa: BLE001
        return None

    changed = False
    for element in root.iter():
        transform = (element.get("transform") or "").strip()
        if not transform:
            continue
        if element.tag.split("}")[-1] != "path":
            return None
        match = _TRANSLATE_RE.fullmatch(transform)
        d = element.get("d") or ""
        commands = re.findall(r"[A-Za-z]", d)
        if match is None or not commands or any(command not in _LINEAR_COMMANDS for command in commands):
            return None
        tx = float(match.group(1))
        ty = float(match.group(2) or 0.0)
        if not np.isfinite([tx, ty]).all():
            return None
        try:
            matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
            normalized = transform_path(parse_path(d), matrix)
        except Exception:  # noqa: BLE001
            return None
        element.set("d", normalized.d())
        del element.attrib["transform"]
        changed = True

    if not changed:
        return False
    tree.write(str(destination), encoding="utf-8", xml_declaration=True)
    return True


def convert_svg_to_cutouts(svg_path: Path) -> dict[str, Any]:
    """Production cutout entry point with exact-byte stacked fallback."""
    from app.safe_cutouts import build_safe_cutout_candidate  # noqa: PLC0415
    from app.transform_journal import _atomic_write_bytes  # noqa: PLC0415

    svg_path = Path(svg_path)
    original_bytes = svg_path.read_bytes()
    original_sha256 = sha256(original_bytes).hexdigest()
    candidate_path = svg_path.with_name(f".{svg_path.name}.curve-safe.candidate.svg")
    normalized_path: Path | None = None
    candidate_path.unlink(missing_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=svg_path.parent,
            prefix=f".{svg_path.name}.",
            suffix=".translated.svg",
            delete=False,
        ) as handle:
            possible_normalized = Path(handle.name)
        normalized = _normalize_polygon_translates(svg_path, possible_normalized)
        if normalized is True:
            normalized_path = possible_normalized
            gate_source = normalized_path
        else:
            possible_normalized.unlink(missing_ok=True)
            gate_source = svg_path

        report = build_safe_cutout_candidate(
            gate_source,
            candidate_path,
            _convert_svg_to_cutouts_polygonal,
        )
        report = {
            **report,
            "input_source_sha256": original_sha256,
            "translate_normalized": normalized is True,
        }
        if report.get("status") not in {"completed", "no_change"}:
            if sha256(svg_path.read_bytes()).hexdigest() != original_sha256:
                _atomic_write_bytes(svg_path, original_bytes)
            return report

        expected_sha256 = str(report.get("published_sha256") or "")
        try:
            os.replace(candidate_path, svg_path)
        except OSError as exc:
            _atomic_write_bytes(svg_path, original_bytes)
            return {
                **report,
                "status": "failed",
                "reason": "atomic_publish_failed",
                "reason_codes": ["atomic_publish_failed"],
                "error": str(exc),
                "fallback": "stacked",
            }

        actual_sha256 = sha256(svg_path.read_bytes()).hexdigest()
        if not expected_sha256 or actual_sha256 != expected_sha256:
            _atomic_write_bytes(svg_path, original_bytes)
            return {
                **report,
                "status": "failed",
                "reason": "published_digest_mismatch",
                "reason_codes": ["published_digest_mismatch"],
                "fallback": "stacked",
            }
        return {**report, "final_sha256": actual_sha256}
    finally:
        candidate_path.unlink(missing_ok=True)
        if normalized_path is not None:
            normalized_path.unlink(missing_ok=True)


__all__ = ["convert_svg_to_cutouts", "is_available"]
