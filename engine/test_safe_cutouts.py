from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

import app.cutouts as cutouts
from app.fidelity import render_svg_to_rgb
from app.safe_cutouts import build_safe_cutout_candidate


def _write(path: Path, body: str, *, width: int = 64, height: int = 40) -> bytes:
    text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )
    path.write_text(text, encoding="utf-8")
    return path.read_bytes()


def _polygon_fixture(path: Path) -> bytes:
    return _write(
        path,
        '<path fill="#ff0000" d="M 4 4 L 60 4 L 60 36 L 4 36 Z"/>'
        '<path fill="#0000ff" d="M 32 8 L 56 8 L 56 32 L 32 32 Z"/>',
    )


def test_curved_bezier_counter_falls_back_to_exact_stacked_bytes(tmp_path) -> None:
    path = tmp_path / "curved.svg"
    original = _write(
        path,
        '<path fill="#111111" d="M 4 4 C 24 0 40 0 60 4 L 60 36 L 4 36 Z"/>'
        '<path fill="#ffffff" d="M 20 12 C 28 8 36 8 44 12 L 44 28 L 20 28 Z"/>',
    )
    report = cutouts.convert_svg_to_cutouts(path)
    assert report["status"] == "skipped"
    assert "curve_preservation_unavailable" in report["reason_codes"]
    assert report["fallback"] == "stacked"
    assert path.read_bytes() == original


def test_arc_and_nested_hole_geometry_is_never_polygon_flattened(tmp_path) -> None:
    path = tmp_path / "arc.svg"
    original = _write(
        path,
        '<path fill="#000000" fill-rule="evenodd" '
        'd="M 32 3 A 17 17 0 1 1 31.99 3 Z M 32 13 A 7 7 0 1 0 32.01 13 Z"/>'
        '<path fill="#ff0000" d="M 28 18 L 36 18 L 36 26 L 28 26 Z"/>',
    )
    report = cutouts.convert_svg_to_cutouts(path)
    assert report["status"] == "skipped"
    assert report["source_contract"]["curve_commands"] == ["A"]
    assert path.read_bytes() == original


def test_unsupported_rotation_falls_back_without_partial_mutation(tmp_path) -> None:
    path = tmp_path / "transform.svg"
    original = _write(
        path,
        '<path fill="#ff0000" transform="rotate(15 32 20)" '
        'd="M 4 4 L 60 4 L 60 36 L 4 36 Z"/>'
        '<path fill="#0000ff" d="M 20 10 L 44 10 L 44 30 L 20 30 Z"/>',
    )
    report = cutouts.convert_svg_to_cutouts(path)
    assert report["status"] == "skipped"
    assert "unsupported_transform" in report["reason_codes"]
    assert path.read_bytes() == original


def test_dependency_unavailable_keeps_exact_stacked_bytes(monkeypatch, tmp_path) -> None:
    path = tmp_path / "dependency.svg"
    original = _polygon_fixture(path)
    monkeypatch.setattr(cutouts, "pyclipper", None)
    report = cutouts.convert_svg_to_cutouts(path)
    assert report["status"] == "skipped"
    assert report["reason"] == "dependency_unavailable"
    assert path.read_bytes() == original


def test_converter_exception_or_partial_write_cannot_escape_transaction(tmp_path) -> None:
    source = tmp_path / "source.svg"
    destination = tmp_path / "destination.svg"
    original = _polygon_fixture(source)

    def broken(candidate: Path) -> dict:
        candidate.write_text("<svg><broken", encoding="utf-8")
        raise RuntimeError("boom")

    report = build_safe_cutout_candidate(source, destination, broken)
    assert report["status"] == "failed"
    assert report["reason"] == "converter_exception"
    assert source.read_bytes() == original
    assert destination.exists() is False
    assert list(tmp_path.glob("*.candidate.svg")) == []


def test_missing_path_coverage_is_rejected_before_publish(tmp_path) -> None:
    source = tmp_path / "source.svg"
    destination = tmp_path / "destination.svg"
    original = _polygon_fixture(source)

    def removes_everything(candidate: Path) -> dict:
        _write(candidate, "")
        return {"status": "completed"}

    report = build_safe_cutout_candidate(source, destination, removes_everything)
    assert report["status"] == "failed"
    assert "path_coverage_mismatch" in report["reason_codes"]
    assert source.read_bytes() == original
    assert destination.exists() is False


def test_no_change_requires_and_publishes_exact_digest(tmp_path) -> None:
    source = tmp_path / "source.svg"
    destination = tmp_path / "destination.svg"
    original = _polygon_fixture(source)
    report = build_safe_cutout_candidate(
        source,
        destination,
        lambda _candidate: {"status": "no_change"},
    )
    assert report["status"] == "no_change"
    assert destination.read_bytes() == original
    assert report["published_sha256"] == sha256(original).hexdigest()


def test_polygonal_adjacent_colors_pass_visual_and_growth_contracts(tmp_path) -> None:
    path = tmp_path / "polygon.svg"
    _polygon_fixture(path)
    before = render_svg_to_rgb(path, 64, 40)
    assert before is not None

    report = cutouts.convert_svg_to_cutouts(path)
    after = render_svg_to_rgb(path, 64, 40)

    assert report["status"] == "completed"
    assert report["topology_valid"] is True
    assert report["coverage_complete"] is True
    assert report["curve_preserved"] is True
    assert report["command_count_after"] <= report["command_count_limit"]
    assert after is not None

    pixel_error = np.max(np.abs(before.astype(np.int16) - after.astype(np.int16)), axis=2)
    seam_ratio = float(np.mean(pixel_error > 24))
    halo_ratio = float(np.mean(pixel_error > 12))
    assert seam_ratio <= 0.002
    assert halo_ratio <= 0.02

    path_commands: list[str] = []
    for element in ET.parse(path).getroot().iter():
        if element.tag.split("}")[-1] == "path":
            path_commands.extend(re.findall(r"[A-Za-z]", element.get("d") or ""))
    assert not set(path_commands).intersection(set("CcQqAaSsTt"))
    assert not list(tmp_path.glob("*.curve-safe.candidate.svg"))
