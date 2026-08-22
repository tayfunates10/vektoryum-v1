from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from engine.regression.public15_contour_compositing_probe import _capture_q32, _find_mask_application


def _edge(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel) > 0


def _distance_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if not np.any(a):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    values = dt[a]
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _measure(path: Path, source_rgba: np.ndarray, side: int) -> dict[str, Any]:
    h0, w0 = source_rgba.shape[:2]
    scale = min(1.0, float(side) / float(max(h0, w0)))
    w = max(1, int(round(w0 * scale)))
    h = max(1, int(round(h0 * scale)))
    src = resize_rgba(source_rgba, w, h)
    rnd = render_svg_to_rgba(path, w, h)
    if rnd is None:
        raise RuntimeError(f"render failed for {path}")
    if rnd.shape[:2] != (h, w):
        rnd = resize_rgba(rnd, w, h)

    sa = src[:, :, 3].astype(np.float32) / 255.0
    ra = rnd[:, :, 3].astype(np.float32) / 255.0
    src_edge = _edge(sa > 0.001)
    rnd_edge = _edge(ra > 0.001)
    boundary = cv2.dilate((src_edge | rnd_edge).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    src_rgb = src[:, :, :3].astype(np.float32) / 255.0
    rnd_rgb = rnd[:, :, :3].astype(np.float32) / 255.0
    src_pm = src_rgb * sa[:, :, None]
    rnd_pm = rnd_rgb * ra[:, :, None]
    pm_err = np.mean(np.abs(src_pm - rnd_pm), axis=2)
    straight_err = np.mean(np.abs(src_rgb - rnd_rgb), axis=2)

    src_comp = composite_rgba(src, 255).astype(np.float32) / 255.0
    rnd_comp = composite_rgba(rnd, 255).astype(np.float32) / 255.0
    comp_err = np.mean(np.abs(src_comp - rnd_comp), axis=2)

    transition = (sa > 0.001) & (sa < 0.999)
    return {
        "width": w,
        "height": h,
        "source_edge_pixels": int(np.count_nonzero(src_edge)),
        "render_edge_pixels": int(np.count_nonzero(rnd_edge)),
        "source_to_render_edge_distance": _distance_summary(src_edge, rnd_edge),
        "render_to_source_edge_distance": _distance_summary(rnd_edge, src_edge),
        "source_transition_pixels": int(np.count_nonzero(transition)),
        "boundary_pixels": int(np.count_nonzero(boundary)),
        "boundary_premultiplied_rgb_mae": float(np.mean(pm_err[boundary])) if np.any(boundary) else 0.0,
        "boundary_straight_rgb_mae": float(np.mean(straight_err[boundary])) if np.any(boundary) else 0.0,
        "boundary_white_composite_mae": float(np.mean(comp_err[boundary])) if np.any(boundary) else 0.0,
        "transition_premultiplied_rgb_mae": float(np.mean(pm_err[transition])) if np.any(transition) else 0.0,
        "transition_alpha_mae": float(np.mean(np.abs(sa[transition] - ra[transition]))) if np.any(transition) else 0.0,
        "transition_render_alpha_mean": float(np.mean(ra[transition])) if np.any(transition) else 0.0,
        "transition_source_alpha_mean": float(np.mean(sa[transition])) if np.any(transition) else 0.0,
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, _parent, source_rgba = _capture_q32(corpus, out, engine_version)
    root = ET.parse(baseline).getroot()
    application, mask_id = _find_mask_application(root)
    evidence = {
        "schema": "vektoryum-public15-boundary-rgb-residual-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "baseline_bytes": int(baseline.stat().st_size),
        "mask_id": mask_id,
        "mask_application_tag": application.tag.rsplit("}", 1)[-1],
        "scales": {
            "512": _measure(baseline, source_rgba, 512),
            "1024": _measure(baseline, source_rgba, 1024),
        },
        "interpretation_contract": {
            "edge_distances_near_zero_but_premultiplied_rgb_residual_high": "geometry aligns; color/premultiplied compositing at shared alpha boundary is the remaining mechanism",
            "edge_distances_high": "mask/source edge geometry misalignment remains the dominant mechanism",
            "premultiplied_residual_low_but_composite_residual_high": "straight-alpha/compositor conversion is suspect rather than mask geometry",
        },
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed": False,
            "raster_embedding": False,
            "fixture_bypass": False,
        },
    }
    (out / "public15-boundary-rgb-residual-proof.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    evidence = run_probe(Path(os.environ["RFV_CORPUS"]), Path(os.environ["PROOF_OUT"]), os.environ["ENGINE_VERSION"])
    print("PUBLIC15_BOUNDARY_RGB_RESIDUAL=" + json.dumps(evidence["scales"], sort_keys=True))


if __name__ == "__main__":
    main()
