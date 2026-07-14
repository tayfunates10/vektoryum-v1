"""Fail-closed SVG serializer for the deterministic centerline graph fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import cv2
import numpy as np

from app.centerline_graph import (
    skeletonize_binary,
    trace_skeleton_graph,
    validate_centerline_report,
)

SVG_NS = "http://www.w3.org/2000/svg"
REPORT_ID = "vektoryum-centerline-report"


def _number(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _path_data(path: dict[str, Any]) -> str:
    points = tuple(path.get("points") or ())
    if len(points) < 2:
        raise ValueError("centerline path must contain at least two points")
    commands = [f"M {_number(points[0][0])} {_number(points[0][1])}"]
    commands.extend(f"L {_number(x)} {_number(y)}" for x, y in points[1:])
    return " ".join(commands)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_centerline_report(svg_path: Path) -> dict[str, Any] | None:
    """Read the embedded graph report. Invalid JSON fails closed to ``None``."""
    try:
        root = ET.parse(str(svg_path)).getroot()
    except Exception:  # noqa: BLE001
        return None
    for element in root.iter():
        if element.tag.split("}")[-1] != "metadata":
            continue
        if element.get("id") != REPORT_ID or not element.text:
            continue
        try:
            report = json.loads(element.text)
        except (TypeError, ValueError):
            return None
        return report if isinstance(report, dict) else None
    return None


def vectorize_skeleton_graph_to_svg(
    input_path: Path,
    output_path: Path,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one exact SVG only after the centerline topology contract passes."""
    params = params or {}
    min_branch = params.get("min_branch", 6)
    stroke_width = params.get("stroke_width", 1.0)
    if isinstance(stroke_width, bool) or not isinstance(stroke_width, (int, float)):
        raise ValueError("stroke_width must be numeric")
    if not 0.1 <= float(stroke_width) <= 20.0:
        raise ValueError("stroke_width must be between 0.1 and 20")

    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Görsel okunamadı: {input_path}")
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    skeleton = skeletonize_binary(binary)
    traced = trace_skeleton_graph(skeleton, min_branch=min_branch)
    report = {
        **traced["report"],
        "source_ink_pixels": int(np.count_nonzero(binary)),
        "skeleton_pixel_count": int(np.count_nonzero(skeleton)),
    }
    valid, errors = validate_centerline_report(report)
    if not valid:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError("centerline topology contract failed: " + "; ".join(errors))

    metadata = json.dumps(
        report,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    paths = "".join(
        '<path fill="none" stroke="#000000" '
        f'stroke-width="{_number(float(stroke_width))}" '
        'stroke-linecap="round" stroke-linejoin="round" '
        f'd="{_path_data(path)}"/>'
        for path in traced["paths"]
    )
    height, width = image.shape
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        'data-vector-backend="opencv_skeleton_graph" '
        f'data-centerline-confidence="{_number(float(report["confidence"]))}">'
        f'<metadata id="{REPORT_ID}">{escape(metadata)}</metadata>'
        f'{paths}</svg>'
    )
    _atomic_write(Path(output_path), svg)

    embedded = read_centerline_report(Path(output_path))
    embedded_valid, embedded_errors = validate_centerline_report(embedded)
    if not embedded_valid:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(
            "embedded centerline report failed: " + "; ".join(embedded_errors)
        )
    return report


__all__ = ["read_centerline_report", "vectorize_skeleton_graph_to_svg"]
