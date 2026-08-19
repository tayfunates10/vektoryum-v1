"""Byte-minimal SVG lifecycle repairs applied before candidate scoring.

SVG fill semantics implicitly close open subpaths, while the production release
contract requires filled cycles to be explicitly/geometrically closed.  For
fill-only paths, inserting ``Z`` is render-equivalent and makes the structural
contract explicit.  Paths with a visible stroke are never touched because
closing those could change stroke rendering.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.component_quality import _filled_path_subpaths_closed

_PATH_TAG_RE = re.compile(r"<path\b[^>]*>", re.I | re.S)
_ATTR_RE_TEMPLATE = r"(?:^|\s){name}\s*=\s*([\"'])(.*?)\1"
_STYLE_PROP_TEMPLATE = r"(?:^|;)\s*{name}\s*:\s*([^;]+)"
_M_COMMAND_RE = re.compile(r"[Mm]")
_D_ATTR_RE = re.compile(r"(?P<prefix>(?:^|\s)d\s*=\s*)(?P<quote>[\"'])(?P<value>.*?)(?P=quote)", re.I | re.S)


def _attr(tag: str, name: str) -> str | None:
    match = re.search(_ATTR_RE_TEMPLATE.format(name=re.escape(name)), tag, re.I | re.S)
    return match.group(2).strip() if match else None


def _style_prop(tag: str, name: str) -> str | None:
    style = _attr(tag, "style")
    if not style:
        return None
    match = re.search(_STYLE_PROP_TEMPLATE.format(name=re.escape(name)), style, re.I)
    return match.group(1).strip() if match else None


def _paint_value(tag: str, name: str, *, default: str | None = None) -> str | None:
    value = _attr(tag, name)
    if value is None:
        value = _style_prop(tag, name)
    if value is None:
        value = default
    return value.lower() if isinstance(value, str) else value


def _split_subpaths(d: str) -> list[tuple[int, int]]:
    """Return text spans beginning at each explicit M/m command."""
    starts = [match.start() for match in _M_COMMAND_RE.finditer(d)]
    if not starts:
        return []
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(d))
        for index, start in enumerate(starts)
    ]


def _close_fill_subpaths(d: str) -> tuple[str, int]:
    spans = _split_subpaths(d)
    if not spans:
        return d, 0

    pieces: list[str] = []
    cursor = 0
    closed_count = 0
    for start, end in spans:
        pieces.append(d[cursor:start])
        segment = d[start:end]
        if _filled_path_subpaths_closed(segment):
            pieces.append(segment)
        else:
            trailing_len = len(segment) - len(segment.rstrip())
            if trailing_len:
                pieces.append(segment[:-trailing_len] + "Z" + segment[-trailing_len:])
            else:
                pieces.append(segment + "Z")
            closed_count += 1
        cursor = end
    pieces.append(d[cursor:])
    return "".join(pieces), closed_count


def close_implicit_fill_cycles(svg_path: Path) -> dict[str, Any]:
    """Explicitly close render-equivalent fill-only subpaths before scoring.

    The transform is deliberately byte-minimal: only ``Z`` bytes are inserted
    into existing ``d`` attributes.  No XML reserialization, palette change,
    geometry approximation, threshold change, or raster embedding occurs.
    """
    path = Path(svg_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"applied": False, "reason": f"read_error:{type(exc).__name__}", "paths_changed": 0, "subpaths_closed": 0}

    paths_changed = 0
    subpaths_closed = 0

    def _rewrite_tag(match: re.Match[str]) -> str:
        nonlocal paths_changed, subpaths_closed
        tag = match.group(0)
        fill = _paint_value(tag, "fill", default="black")
        stroke = _paint_value(tag, "stroke", default="none")
        if fill == "none" or stroke not in {None, "none", "transparent"}:
            return tag
        d_match = _D_ATTR_RE.search(tag)
        if not d_match:
            return tag
        old_d = d_match.group("value")
        new_d, count = _close_fill_subpaths(old_d)
        if count <= 0 or new_d == old_d:
            return tag
        paths_changed += 1
        subpaths_closed += count
        start, end = d_match.span("value")
        return tag[:start] + new_d + tag[end:]

    updated = _PATH_TAG_RE.sub(_rewrite_tag, text)
    if updated == text:
        return {"applied": False, "reason": "no_implicit_fill_cycles", "paths_changed": 0, "subpaths_closed": 0}
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return {"applied": False, "reason": f"write_error:{type(exc).__name__}", "paths_changed": 0, "subpaths_closed": 0}
    return {
        "applied": True,
        "reason": "explicit_fill_cycle_normalization",
        "paths_changed": paths_changed,
        "subpaths_closed": subpaths_closed,
    }


__all__ = ["close_implicit_fill_cycles"]
