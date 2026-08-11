"""Connected-component integrity metrics for winner safety (AI-2 P1-A).

Global perceptual scores are area weighted and can hide a missing ® mark, a
split micro shape, or a newly invented island.  This module measures those
failure modes independently and exposes a hard *selection* gate without
changing any existing SSIM/DeltaE/edge-F1 weights or release thresholds.

The gate is intentionally applicable only to palette-like, non-photo,
non-gradient inputs.  A non-applicable input is not a failure.  An applicable
input whose component measurement cannot be produced is ``needs_review`` and
is never treated as a quality pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Preserve the production component-support floor already used by fidelity.py.
_MIN_COMPONENT_PIXELS = 48
_MIN_COMPONENT_FRACTION = 0.0001
_MAX_SOURCE_PALETTE_MEDIAN_RESIDUAL = 10.0
_KMEANS_K = 6
_KMEANS_SEED = 7

# Issue #137 acceptance invariants.  These are a new independent selection
# contract, not relaxations/changes to existing release-quality thresholds.
_REQUIRED_SOURCE_CC_RECALL = 1.0
_REQUIRED_RENDER_CC_PRECISION = 1.0
_REQUIRED_MIN_TRUE_CC_IOU = 0.95

# Failed candidates must live below every measured-pass candidate, but if ALL
# candidates fail the new gate their *legacy ordering* must remain unchanged.
# A constant offset preserves every old score delta / near-score heuristic while
# still ensuring a safe candidate (normal non-negative score range) always wins.
_FAIL_SCORE_OFFSET = 1000.0
_UNMEASURED_SCORE_OFFSET = 2000.0


@dataclass(frozen=True)
class _Component:
    class_index: int
    label_index: int
    area: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def component_gate_applicability(mode: str, analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Return explicit applicability for the production CC gate."""
    analysis = analysis or {}
    if mode == "photo_poster" or bool(analysis.get("likely_photo_or_complex")):
        return {"applicable": False, "reason": "photo_or_complex_input"}
    if bool(analysis.get("has_gradient")) or bool(analysis.get("tonal_gradient_foreground")):
        return {"applicable": False, "reason": "gradient_input"}
    return {"applicable": True, "reason": "palette_like_input"}


def _source_palette_classes(
    original_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    k: int = _KMEANS_K,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Classify source/render against deterministic centers learned from source."""
    if original_rgb.shape != rendered_rgb.shape:
        h, w = original_rgb.shape[:2]
        rendered_rgb = cv2.resize(rendered_rgb, (w, h), interpolation=cv2.INTER_AREA)

    lab_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab_o.reshape(-1, 3)
    step = max(1, samples.shape[0] // 40000)
    sub = samples[::step]

    # Do not request more clusters than distinct sampled LAB tuples.  This also
    # avoids OpenCV kmeans assertions for degenerate/flat fixtures.
    distinct = int(np.unique(sub.astype(np.uint8), axis=0).shape[0])
    k_eff = max(1, min(int(k), distinct, int(sub.shape[0])))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(_KMEANS_SEED)
    _compact, _labels, centers = cv2.kmeans(
        sub, k_eff, None, crit, 1, cv2.KMEANS_PP_CENTERS
    )

    from app.palette_ops import classify_features  # noqa: PLC0415

    cls_o = classify_features(lab_o, centers)
    cls_r = classify_features(lab_r, centers)
    reconstructed = np.take(centers, cls_o.reshape(-1), axis=0).reshape(lab_o.shape)
    median_residual = float(np.median(np.linalg.norm(lab_o - reconstructed, axis=2)))
    return cls_o, cls_r, median_residual


def _components(cls_map: np.ndarray, class_index: int, min_area: int) -> list[_Component]:
    mask = (cls_map == class_index).astype(np.uint8)
    if int(mask.sum()) < min_area:
        return []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    out: list[_Component] = []
    for label_index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label_index])
        if area < min_area:
            continue
        out.append(
            _Component(
                class_index=class_index,
                label_index=label_index,
                area=area,
                bbox=(x, y, w, h),
                mask=labels == label_index,
            )
        )
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(a & b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(intersection) / float(max(1, union))


def _greedy_one_to_one_matches(
    sources: list[_Component],
    renders: list[_Component],
) -> tuple[list[tuple[int, int, float]], list[float]]:
    """Deterministic maximum-IoU greedy matching within one palette class.

    Each render component can satisfy at most one source component.  Therefore a
    split source leaves extra render pieces (precision loss), and a missing
    source leaves an unmatched source (recall loss).  A fake island remains an
    unmatched render component.  Ties are stable by source/render list index.
    """
    pairs: list[tuple[float, int, int]] = []
    best_per_source = [0.0 for _ in sources]
    for si, source in enumerate(sources):
        for ri, render in enumerate(renders):
            value = _iou(source.mask, render.mask)
            best_per_source[si] = max(best_per_source[si], value)
            if value > 0.0:
                pairs.append((value, si, ri))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_s: set[int] = set()
    used_r: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, si, ri in pairs:
        if si in used_s or ri in used_r:
            continue
        used_s.add(si)
        used_r.add(ri)
        matches.append((si, ri, value))
    return matches, best_per_source


def measure_component_integrity_arrays(
    original_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    *,
    k: int = _KMEANS_K,
) -> dict[str, Any]:
    """Measure true source-CC recall, render-CC precision and min source IoU."""
    try:
        cls_o, cls_r, palette_residual = _source_palette_classes(original_rgb, rendered_rgb, k=k)
    except Exception as exc:  # noqa: BLE001
        return {
            "measured": False,
            "status": "needs_review",
            "reason": f"classification_error:{type(exc).__name__}",
        }

    if palette_residual > _MAX_SOURCE_PALETTE_MEDIAN_RESIDUAL:
        return {
            "measured": False,
            "status": "not_applicable",
            "reason": "continuous_tone_source",
            "source_palette_median_residual": round(palette_residual, 4),
        }

    h, w = cls_o.shape[:2]
    min_area = max(_MIN_COMPONENT_PIXELS, int(_MIN_COMPONENT_FRACTION * h * w))
    class_count = max(int(cls_o.max(initial=0)), int(cls_r.max(initial=0))) + 1

    source_total = 0
    render_total = 0
    matched_source = 0
    matched_render = 0
    source_best_ious: list[float] = []
    class_reports: list[dict[str, Any]] = []

    for class_index in range(class_count):
        sources = _components(cls_o, class_index, min_area)
        renders = _components(cls_r, class_index, min_area)
        if not sources and not renders:
            continue
        matches, best_per_source = _greedy_one_to_one_matches(sources, renders)
        matched_positive = [(si, ri, value) for si, ri, value in matches if value > 0.0]
        source_total += len(sources)
        render_total += len(renders)
        matched_source += len(matched_positive)
        matched_render += len(matched_positive)
        source_best_ious.extend(best_per_source)
        class_reports.append({
            "class_index": class_index,
            "source_components": len(sources),
            "render_components": len(renders),
            "matched_components": len(matched_positive),
            "min_source_best_iou": round(min(best_per_source), 4) if best_per_source else None,
        })

    if source_total == 0:
        return {
            "measured": False,
            "status": "needs_review",
            "reason": "no_supported_source_components",
            "source_palette_median_residual": round(palette_residual, 4),
            "min_component_area": min_area,
        }

    source_recall = float(matched_source) / float(source_total)
    render_precision = (
        float(matched_render) / float(render_total) if render_total > 0 else 0.0
    )
    min_true_iou = min(source_best_ious) if source_best_ious else 0.0
    passed = (
        source_recall >= _REQUIRED_SOURCE_CC_RECALL
        and render_precision >= _REQUIRED_RENDER_CC_PRECISION
        and min_true_iou >= _REQUIRED_MIN_TRUE_CC_IOU
    )
    return {
        "measured": True,
        "status": "pass" if passed else "fail",
        "reason": "component_integrity_pass" if passed else "component_integrity_fail",
        "source_cc_recall": round(source_recall, 4),
        "render_cc_precision": round(render_precision, 4),
        "min_true_cc_iou": round(float(min_true_iou), 4),
        "source_components": source_total,
        "render_components": render_total,
        "matched_components": matched_source,
        "min_component_area": min_area,
        "source_palette_median_residual": round(palette_residual, 4),
        "class_reports": class_reports,
    }


def score_svg_component_integrity(
    svg_path: Path,
    original_path: Path,
    *,
    mode: str,
    analysis: dict[str, Any] | None,
    max_side: int = 512,
) -> dict[str, Any]:
    """Production-safe SVG CC report with explicit applicability/fail-closed state."""
    applicability = component_gate_applicability(mode, analysis)
    if not applicability["applicable"]:
        return {
            "applicable": False,
            "measured": False,
            "status": "not_applicable",
            "reason": applicability["reason"],
        }

    try:
        from app.fidelity import load_reference_rgb, render_svg_to_rgb  # noqa: PLC0415

        reference, (w, h) = load_reference_rgb(Path(original_path), max_side=max_side)
        rendered = render_svg_to_rgb(Path(svg_path), w, h)
        if rendered is None:
            return {
                "applicable": True,
                "measured": False,
                "status": "needs_review",
                "reason": "render_unavailable",
            }
        report = measure_component_integrity_arrays(reference, rendered)
    except Exception as exc:  # noqa: BLE001
        return {
            "applicable": True,
            "measured": False,
            "status": "needs_review",
            "reason": f"measurement_error:{type(exc).__name__}",
        }

    if report.get("status") == "not_applicable":
        # Source itself proved continuous-tone.  This is a legitimate
        # applicability outcome, not a missing measurement.
        return {"applicable": False, **report}
    return {"applicable": True, **report}


def gate_candidate_scores(
    total_score: float,
    fidelity_score: float | None,
    component_report: dict[str, Any],
) -> dict[str, Any]:
    """Apply the independent CC eligibility gate without re-ranking failures.

    A passing/non-applicable candidate keeps its exact legacy score.  A measured
    failure gets a constant negative offset, so any passing candidate outranks
    it but all old score differences and geometric near-score heuristics are
    preserved if every candidate fails.  Missing applicable measurement is a
    separate, lower fail-closed band and carries ``needs_review``.
    """
    status = str(component_report.get("status") or "needs_review")
    applicable = bool(component_report.get("applicable"))
    result = {
        "total_score": float(total_score),
        "fidelity_score": fidelity_score,
        "selection_disqualified": False,
        "component_quality_status": status,
    }
    if not applicable or status == "pass":
        return result

    result["selection_disqualified"] = True
    offset = (
        _UNMEASURED_SCORE_OFFSET
        if status == "needs_review" or not component_report.get("measured")
        else _FAIL_SCORE_OFFSET
    )
    result["total_score"] = float(total_score) - offset
    result["fidelity_score"] = (
        float(fidelity_score) - offset if fidelity_score is not None else None
    )
    return result
