from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_contour_compositing_probe import _capture_q32


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "mae": 0.0, "p95_abs": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
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
    alpha_error = ra - sa
    transition = (sa > 0.001) & (sa < 0.999)

    src_rgb = src[:, :, :3].astype(np.float32) / 255.0
    rnd_rgb = rnd[:, :, :3].astype(np.float32) / 255.0
    src_pm = src_rgb * sa[:, :, None]
    rnd_pm = rnd_rgb * ra[:, :, None]
    pm_error = np.mean(np.abs(rnd_pm - src_pm), axis=2)

    bins: dict[str, Any] = {}
    edges = np.linspace(0.0, 1.0, 9)
    for i in range(8):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = transition & (sa >= lo) & (sa < hi if i < 7 else sa <= hi)
        err = alpha_error[mask]
        bins[f"{lo:.3f}-{hi:.3f}"] = {
            **_summary(err),
            "source_alpha_mean": float(np.mean(sa[mask])) if np.any(mask) else 0.0,
            "render_alpha_mean": float(np.mean(ra[mask])) if np.any(mask) else 0.0,
            "premultiplied_rgb_mae": float(np.mean(pm_error[mask])) if np.any(mask) else 0.0,
        }

    # Signed distance to the binary source boundary, in pixels. This distinguishes
    # edge displacement from same-edge coverage/phase mismatch.
    inside = sa > 0.001
    dist_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
    signed = dist_in - dist_out
    phase_bins: dict[str, Any] = {}
    for lo, hi in [(-2.0, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 4.5)]:
        mask = transition & (signed >= lo) & (signed < hi)
        phase_bins[f"{lo:g}:{hi:g}"] = {
            **_summary(alpha_error[mask]),
            "premultiplied_rgb_mae": float(np.mean(pm_error[mask])) if np.any(mask) else 0.0,
        }

    ae = alpha_error[transition]
    pe = pm_error[transition]
    corr = 0.0
    if ae.size > 1 and np.std(np.abs(ae)) > 1e-12 and np.std(pe) > 1e-12:
        corr = float(np.corrcoef(np.abs(ae), pe)[0, 1])

    return {
        "width": w,
        "height": h,
        "transition_pixels": int(np.count_nonzero(transition)),
        "transition_alpha_error": _summary(ae),
        "transition_pm_rgb_mae": float(np.mean(pe)) if pe.size else 0.0,
        "abs_alpha_error_to_pm_rgb_error_correlation": corr,
        "source_alpha_bins": bins,
        "signed_boundary_distance_bins": phase_bins,
        "source_transition_unique_alpha": int(np.unique(src[:, :, 3][transition]).size),
        "render_transition_unique_alpha": int(np.unique(rnd[:, :, 3][transition]).size),
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, _parent, source_rgba = _capture_q32(corpus, out, engine_version)
    evidence = {
        "schema": "vektoryum-public15-boundary-phase-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "baseline_bytes": int(baseline.stat().st_size),
        "scales": {
            "512": _measure(baseline, source_rgba, 512),
            "1024": _measure(baseline, source_rgba, 1024),
        },
        "decision_contract": {
            "same_edge_wrong_phase": "edge-distance proof remains subpixel while alpha error changes sign/magnitude by source-alpha or signed-distance bin and premultiplied RGB residual tracks absolute alpha error",
            "geometry_shift": "signed-distance bins show one-sided error consistent with a displaced boundary",
            "not_proven": "no single structured relationship dominates",
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
    (out / "public15-boundary-phase-proof.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    evidence = run_probe(Path(os.environ["RFV_CORPUS"]), Path(os.environ["PROOF_OUT"]), os.environ["ENGINE_VERSION"])
    print("PUBLIC15_BOUNDARY_PHASE=" + json.dumps(evidence["scales"], sort_keys=True))


if __name__ == "__main__":
    main()
