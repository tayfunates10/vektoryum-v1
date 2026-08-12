"""AI-2 Issue #137 native-256 acceptance evidence.

The named QA fixtures in #137 are not stored as repository assets.  This module
therefore defines deterministic *canonical reproducers* with those names.  They
exercise the same production palette failure path in isolation: identical SVG
geometry is consolidated once with legacy CANONICAL_BWR and once with the
palette-preserving canonical=None path.  The source raster is rendered from the
unmodified truth SVG at native 256px so the comparison isolates palette
canonicalization rather than trace geometry.

The P0-B2b routing contract is tested separately in test_ai2_neutral_palette.py.
For the metric table here, every after artifact intentionally enters the
geometric palette-preserving family; ``auto_recommended_mode`` is reported as a
diagnostic rather than being confused with the isolated stage's explicit mode.

Run:
    python ai2_acceptance_metrics.py --output ai2_acceptance_metrics.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.analyzer import analyze_image_from_mem
from app.component_quality import measure_component_integrity_arrays
from app.fidelity import render_svg_to_rgb
from app.final_artifact_evaluator import _lab, ciede2000
from app.geometry_cleanup import consolidate_svg_palette
from app.neutral_palette import detect_neutral_luminance_bands
from app.pipeline import CANONICAL_BWR

SIZE = 256


def _svg(paths: list[tuple[str, str]]) -> str:
    body = "".join(f'<path d="{d}" fill="{fill}"/>' for d, fill in paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}">{body}</svg>'
    )


def _fixtures() -> dict[str, dict[str, Any]]:
    return {
        "qa-lowres-badge": {
            "phase": "P0-B2a",
            "expected_mode": "geometric_logo",
            "paths": [
                ("M0 0H256V256H0Z", "#f4f4f4"),
                ("M24 24H232V232H24Z", "#d0d0d0"),
                ("M42 42H214V214H42Z", "#888888"),
                ("M62 62H194V194H62Z", "#202020"),
                ("M84 84H172V172H84Z", "#f4f4f4"),
            ],
        },
        "qa-gray-border-counter": {
            "phase": "P0-B2b",
            "expected_mode": "geometric_logo",
            "require_auto_geometric": True,
            "paths": [
                ("M0 0H256V256H0Z", "#f7f7f7"),
                ("M20 20H236V236H20Z", "#b8b8b8"),
                ("M42 42H214V214H42Z", "#242424"),
                ("M90 90H166V166H90Z", "#f7f7f7"),
            ],
        },
        "neutral-tone-steps": {
            "phase": "P0-B2a",
            "expected_mode": "geometric_logo",
            # Every visible band is separated by far more than the existing
            # merge_tol=12.  There is deliberately no hidden near-duplicate
            # background fill: the five stripes cover the complete canvas.
            "paths": [
                ("M0 0H52V256H0Z", "#202020"),
                ("M52 0H104V256H52Z", "#606060"),
                ("M104 0H156V256H104Z", "#989898"),
                ("M156 0H208V256H156Z", "#c8c8c8"),
                ("M208 0H256V256H208Z", "#f0f0f0"),
            ],
        },
    }


def _render(path: Path) -> np.ndarray:
    rendered = render_svg_to_rgb(path, SIZE, SIZE)
    if rendered is None:
        raise RuntimeError(f"render unavailable: {path}")
    return rendered


def _boundary_p95(source: np.ndarray, rendered: np.ndarray) -> float:
    src_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    rnd_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    src_edge = cv2.Canny(src_gray, 20, 60) > 0
    rnd_edge = cv2.Canny(rnd_gray, 20, 60) > 0
    if not src_edge.any() and not rnd_edge.any():
        return 0.0
    if not src_edge.any() or not rnd_edge.any():
        return float("inf")
    dt_r = cv2.distanceTransform((~rnd_edge).astype(np.uint8), cv2.DIST_L2, 5)
    dt_s = cv2.distanceTransform((~src_edge).astype(np.uint8), cv2.DIST_L2, 5)
    values = np.concatenate([dt_r[src_edge], dt_s[rnd_edge]])
    return float(np.percentile(values, 95))


def _metrics(source: np.ndarray, rendered: np.ndarray, k: int) -> dict[str, Any]:
    diff = np.max(np.abs(source.astype(np.int16) - rendered.astype(np.int16)), axis=2)
    visible_residual = float(np.mean(diff > 2))
    de00 = ciede2000(_lab(source), _lab(rendered)).reshape(-1)
    cc = measure_component_integrity_arrays(source, rendered, k=k)
    return {
        "visible_residual": round(visible_residual, 6),
        "de00_p95": round(float(np.percentile(de00, 95)), 6),
        "source_cc_recall": cc.get("source_cc_recall"),
        "render_cc_precision": cc.get("render_cc_precision"),
        "min_true_cc_iou": cc.get("min_true_cc_iou"),
        "component_status": cc.get("status"),
        "boundary_p95_px": round(_boundary_p95(source, rendered), 6),
    }


def _evaluate_case(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    truth = root / f"{name}-truth.svg"
    before = root / f"{name}-legacy-bwr.svg"
    after = root / f"{name}-palette-preserving.svg"
    source_path = root / f"{name}.png"

    truth.write_text(_svg(spec["paths"]), encoding="utf-8")
    source = _render(truth)
    Image.fromarray(source).save(source_path)

    shutil.copyfile(truth, before)
    shutil.copyfile(truth, after)
    # Historical failure path isolated to the relevant variable: flat-mode
    # consolidation against B/W/R.  No trace/path/node budget is changed.
    consolidate_svg_palette(before, max_colors=6, canonical=CANONICAL_BWR, merge_tol=12.0)
    # Fixed P0-B2a behavior: layered-neutral geometric input keeps its source
    # palette family rather than canonicalizing to B/W/R.
    consolidate_svg_palette(after, max_colors=6, canonical=None, merge_tol=12.0)

    before_rgb = _render(before)
    after_rgb = _render(after)
    neutral = detect_neutral_luminance_bands(source_path)
    analysis = analyze_image_from_mem(Image.fromarray(source))
    band_count = int(neutral.get("band_count", 0))
    k = max(2, min(6, band_count))

    return {
        "fixture_kind": "canonical_reproducer_not_historical_asset",
        "phase": spec["phase"],
        "native_size": [SIZE, SIZE],
        "neutral_band_count": band_count,
        "neutral_band_centers": list(neutral.get("band_centers") or []),
        "auto_recommended_mode": str(analysis.get("recommended_mode")),
        "auto_route_required": bool(spec.get("require_auto_geometric")),
        "before": {
            "mode": "geometric_logo",
            "winner": "isolated_legacy_CANONICAL_BWR_stage",
            **_metrics(source, before_rgb, k=k),
        },
        "after": {
            "mode": spec["expected_mode"],
            "winner": "isolated_palette_preserving_geometric_stage",
            **_metrics(source, after_rgb, k=k),
        },
    }


def build_acceptance_report(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cases = {name: _evaluate_case(root, name, spec) for name, spec in _fixtures().items()}
    return {
        "schema_version": "ai2-issue-137-native256-v2",
        "note": (
            "Named QA assets were absent; deterministic canonical reproducers are used and explicitly marked. "
            "Metrics isolate the palette canonicalization variable; P0-B2b auto-routing is asserted separately."
        ),
        "acceptance": {
            "source_cc_recall": 1.0,
            "render_cc_precision": 1.0,
            "min_true_cc_iou": 0.95,
            "visible_residual": 0.01,
            "de00_p95": 1.0,
            "boundary_p95_px": 0.75,
        },
        "cases": cases,
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, case in report["cases"].items():
        after = case["after"]
        if case["neutral_band_count"] < 3:
            errors.append(f"{name}: neutral_band_count < 3")
        if after["mode"] != "geometric_logo":
            errors.append(f"{name}: after mode {after['mode']} != geometric_logo")
        if case.get("auto_route_required") and case.get("auto_recommended_mode") != "geometric_logo":
            errors.append(
                f"{name}: auto route {case.get('auto_recommended_mode')} != geometric_logo"
            )
        if after["source_cc_recall"] != 1.0:
            errors.append(f"{name}: source_cc_recall != 1.0")
        if after["render_cc_precision"] != 1.0:
            errors.append(f"{name}: render_cc_precision != 1.0")
        if float(after["min_true_cc_iou"] or 0.0) < 0.95:
            errors.append(f"{name}: min_true_cc_iou < 0.95")
        if float(after["visible_residual"]) > 0.01:
            errors.append(f"{name}: visible_residual > 1%")
        if float(after["de00_p95"]) > 1.0:
            errors.append(f"{name}: de00_p95 > 1.0")
        if float(after["boundary_p95_px"]) > 0.75:
            errors.append(f"{name}: boundary_p95 > 0.75px")
        # This evidence must prove an actual palette regression, not merely
        # generate a passing after image.
        if float(case["before"]["visible_residual"]) <= float(after["visible_residual"]):
            errors.append(f"{name}: legacy BWR residual is not worse than after")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ai2_acceptance_metrics.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ai2-137-") as tmp:
        report = build_acceptance_report(Path(tmp))
    errors = validate_report(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
