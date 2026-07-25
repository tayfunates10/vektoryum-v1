"""Select support-paint geometry without closing comparison-canvas holes.

The opaque trace canvas is renderer-proven before this module is used. Paint
whose solid fill matches that canvas represents comparison-background-colored
negative space or interior detail. Expanding those paths closes holes and adds
seams. Only paint that measurably contrasts with the proven canvas receives the
same-color support stroke.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from app.alpha_candidate_knockout import _local_name
from app.alpha_candidate_support import (
    _ALPHA_STYLE_NAMES,
    _GEOMETRY_TAGS,
    _style_declarations,
    _write_style,
)

_RGB_FUNCTION = re.compile(
    r"^rgb\(\s*([0-9.]+%?)\s*[, ]\s*([0-9.]+%?)\s*[, ]\s*([0-9.]+%?)\s*\)$",
    re.IGNORECASE,
)
_NAMED_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
}
_CANVAS_COLOR_TOLERANCE = 8


def _parse_channel(value: str) -> int | None:
    try:
        if value.endswith("%"):
            parsed = round(float(value[:-1]) * 2.55)
        else:
            parsed = round(float(value))
    except ValueError:
        return None
    return int(min(255, max(0, parsed)))


def parse_solid_rgb(value: str | None) -> tuple[int, int, int] | None:
    """Parse deterministic solid CSS colors used by production trace output."""
    raw = str(value or "").strip().lower()
    if not raw or raw in {"none", "transparent"} or raw.startswith("url("):
        return None
    if raw in _NAMED_RGB:
        return _NAMED_RGB[raw]
    if raw.startswith("#"):
        token = raw[1:]
        if len(token) in {3, 4}:
            token = "".join(character * 2 for character in token[:3])
        elif len(token) in {6, 8}:
            token = token[:6]
        else:
            return None
        try:
            return tuple(int(token[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            return None
    match = _RGB_FUNCTION.fullmatch(raw)
    if match is None:
        return None
    channels = tuple(_parse_channel(value) for value in match.groups())
    if any(channel is None for channel in channels):
        return None
    return channels  # type: ignore[return-value]


def _resolved_fill(node: ET.Element, inherited_fill: str | None = None) -> str | None:
    declarations = _style_declarations(node.get("style"))
    return declarations.get("fill", node.get("fill", inherited_fill))


def comparison_canvas_rgb(canvas: ET.Element) -> tuple[int, int, int] | None:
    """Return one unambiguous solid fill for the renderer-proven canvas."""
    fills: set[tuple[int, int, int]] = set()

    def visit(node: ET.Element, inherited_fill: str | None = None) -> None:
        fill = _resolved_fill(node, inherited_fill)
        local = _local_name(str(node.tag)).lower()
        if local in _GEOMETRY_TAGS:
            parsed = parse_solid_rgb(fill)
            if parsed is not None:
                fills.add(parsed)
        for child in list(node):
            visit(child, fill)

    visit(canvas)
    return next(iter(fills)) if len(fills) == 1 else None


def _matches_canvas(
    fill: str | None,
    canvas_rgb: tuple[int, int, int] | None,
) -> bool:
    if canvas_rgb is None:
        return False
    parsed = parse_solid_rgb(fill)
    if parsed is None:
        return False
    return max(abs(int(left) - int(right)) for left, right in zip(parsed, canvas_rgb)) <= _CANVAS_COLOR_TOLERANCE


def expand_non_canvas_paint(
    node: ET.Element,
    stroke_width: float,
    canvas: ET.Element,
) -> tuple[int, int, tuple[int, int, int] | None]:
    """Stroke only solid fills contrasting with the proven comparison canvas.

    Returns ``(expanded_count, canvas_matched_skip_count, canvas_rgb)``.
    Existing path data and path-command counts are never changed.
    """
    canvas_rgb = comparison_canvas_rgb(canvas)

    def visit(element: ET.Element, inherited_fill: str | None = None) -> tuple[int, int]:
        declarations = _style_declarations(element.get("style"))
        fill = declarations.get("fill", element.get("fill", inherited_fill))
        if fill is None:
            fill = "#000000"
        local = _local_name(str(element.tag)).lower()
        expanded = skipped = 0
        paintable = local in _GEOMETRY_TAGS and str(fill).strip().lower() not in {
            "none",
            "transparent",
            "rgba(0,0,0,0)",
        }
        if paintable and _matches_canvas(fill, canvas_rgb):
            skipped = 1
        elif paintable:
            for key in (
                "stroke",
                "stroke-width",
                "stroke-linejoin",
                "stroke-linecap",
                "stroke-dasharray",
                "stroke-dashoffset",
                "stroke-opacity",
                "paint-order",
            ):
                declarations.pop(key, None)
                element.attrib.pop(key, None)
            for key in _ALPHA_STYLE_NAMES:
                declarations.pop(key, None)
                element.attrib.pop(key, None)
            _write_style(element, declarations)
            element.set("stroke", fill)
            element.set(
                "stroke-width",
                f"{stroke_width:.8f}".rstrip("0").rstrip("."),
            )
            element.set("stroke-linejoin", "round")
            element.set("stroke-linecap", "round")
            element.set("paint-order", "stroke fill markers")
            expanded = 1
        for child in list(element):
            child_expanded, child_skipped = visit(child, fill)
            expanded += child_expanded
            skipped += child_skipped
        return expanded, skipped

    expanded_count, skipped_count = visit(node)
    return expanded_count, skipped_count, canvas_rgb
