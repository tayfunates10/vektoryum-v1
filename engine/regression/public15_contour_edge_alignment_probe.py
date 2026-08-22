from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from engine.regression.public15_contour_compositing_probe import _capture_q32, _write_variants


def _resize(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape[:2] == (height, width):
        return arr
    return resize_rgba(arr, width, height)


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    # Diagnostic-only, deterministic edge localization. These values do not gate acceptance.
    return cv2.Canny(gray, 64, 128) > 0


def _transition(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.int16)
    dx = np.zeros_like(a, dtype=bool)
    dy = np.zeros_like(a, dtype=bool)
    dx[:, 1:] = a[:, 1:] != a[:, :-1]
    dy[1:, :] = a[1:, :] != a[:-1, :]
    t = dx | dy
    return cv2.dilate(t.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0


def _distance_stats(edge_a: np.ndarray, edge_b: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(edge_a, dtype=bool)
    b = np.asarray(edge_b, dtype=bool)
    if not a.any() or not b.any():
        return {"a_pixels": int(a.sum()), "b_pixels": int(b.sum()), "median_px": None, "p95_px": None, "within_1px": None, "within_2px": None}
    distance = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[a]
    return {
        "a_pixels": int(a.sum()),
        "b_pixels": int(b.sum()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "within_1px": float(np.mean(values <= 1.0)),
        "within_2px": float(np.mean(values <= 2.0)),
    }


def _residual_summary(source_rgb: np.ndarray, render_rgb: np.ndarray, mask_transition: np.ndarray, source_transition: np.ndarray) -> dict[str, Any]:
    residual = np.mean(np.abs(source_rgb.astype(np.int16) - render_rgb.astype(np.int16)), axis=2).astype(np.float32)
    total = float(np.sum(residual))
    def band(mask: np.ndarray) -> dict[str, float | int]:
        m = np.asarray(mask, dtype=bool)
        vals = residual[m]
        return {
            "pixels": int(m.sum()),
            "mae": float(np.mean(vals)) if vals.size else 0.0,
            "residual_share": float(np.sum(vals) / total) if total > 0 else 0.0,
        }
    return {
        "global_mae": float(np.mean(residual)),
        "global_p95": float(np.percentile(residual, 95)),
        "mask_transition_band": band(mask_transition),
        "source_alpha_transition_band": band(source_transition),
        "outside_mask_transition": band(~mask_transition),
    }


def _alpha_bins(mask_alpha: np.ndarray, source_rgb: np.ndarray, render_rgb: np.ndarray) -> list[dict[str, float | int]]:
    residual = np.mean(np.abs(source_rgb.astype(np.int16) - render_rgb.astype(np.int16)), axis=2).astype(np.float32)
    bins = ((0, 0), (1, 63), (64, 127), (128, 191), (192, 254), (255, 255))
    rows: list[dict[str, float | int]] = []
    for lo, hi in bins:
        selected = (mask_alpha >= lo) & (mask_alpha <= hi)
        vals = residual[selected]
        rows.append({
            "alpha_lo": lo,
            "alpha_hi": hi,
            "pixels": int(selected.sum()),
            "rgb_mae": float(np.mean(vals)) if vals.size else 0.0,
        })
    return rows


def _measure_scale(baseline: Path, mask_on_solid: Path, source_rgba_full: np.ndarray, max_side: int) -> dict[str, Any]:
    h0, w0 = source_rgba_full.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    width = max(1, int(round(w0 * scale)))
    height = max(1, int(round(h0 * scale)))
    source_rgba = _resize(source_rgba_full, width, height)
    candidate = render_svg_to_rgba(baseline, width, height)
    mask_render = render_svg_to_rgba(mask_on_solid, width, height)
    if candidate is None or mask_render is None:
        return {"status": "render_failed", "width": width, "height": height}
    candidate = _resize(candidate, width, height)
    mask_render = _resize(mask_render, width, height)
    source_rgb = composite_rgba(source_rgba, 255)
    render_rgb = composite_rgba(candidate, 255)
    source_edges = _edge_map(source_rgb)
    render_edges = _edge_map(render_rgb)
    mask_alpha = mask_render[:, :, 3]
    mask_transition = _transition(mask_alpha)
    source_alpha_transition = _transition(source_rgba[:, :, 3])
    return {
        "status": "measured",
        "width": width,
        "height": height,
        "mask_alpha_distinct": int(len(np.unique(mask_alpha))),
        "mask_alpha_min": int(mask_alpha.min()),
        "mask_alpha_max": int(mask_alpha.max()),
        "mask_transition_pixels": int(mask_transition.sum()),
        "source_alpha_transition_pixels": int(source_alpha_transition.sum()),
        "mask_transition_to_source_rgb_edge": _distance_stats(mask_transition, source_edges),
        "source_rgb_edge_to_mask_transition": _distance_stats(source_edges, mask_transition),
        "render_rgb_edge_to_source_rgb_edge": _distance_stats(render_edges, source_edges),
        "source_alpha_transition_to_mask_transition": _distance_stats(source_alpha_transition, mask_transition),
        "residual": _residual_summary(source_rgb, render_rgb, mask_transition, source_alpha_transition),
        "residual_by_mask_alpha": _alpha_bins(mask_alpha, source_rgb, render_rgb),
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, _parent, source_rgba = _capture_q32(corpus, out, engine_version)
    variants = _write_variants(baseline, out)
    mask_on_solid = variants["mask_on_solid"]
    evidence = {
        "schema": "vektoryum-public15-contour-edge-alignment-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "baseline_bytes": int(baseline.stat().st_size),
        "scale_512": _measure_scale(baseline, mask_on_solid, source_rgba, 512),
        "scale_1024": _measure_scale(baseline, mask_on_solid, source_rgba, 1024),
        "interpretation_contract": {
            "mask_transition_residual_concentrated_and_edges_misaligned": "mask/source-RGB edge alignment is the seam mechanism candidate",
            "mask_transition_residual_not_concentrated": "mask/source-RGB edge alignment is insufficient to explain the seam",
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
    (out / "public15-contour-edge-alignment-proof.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    evidence = run_probe(Path(os.environ["RFV_CORPUS"]), Path(os.environ["PROOF_OUT"]), os.environ["ENGINE_VERSION"])
    print("PUBLIC15_CONTOUR_EDGE_ALIGNMENT=" + json.dumps({
        "baseline_bytes": evidence["baseline_bytes"],
        "scale_512": evidence["scale_512"],
        "scale_1024": evidence["scale_1024"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
