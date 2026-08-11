"""Connected-component integrity metrics for winner safety (AI-2 P1-A).

Global perceptual scores are area weighted and can hide a missing ® mark, a
split micro shape, or a newly invented island.  This module measures those
failure modes independently and exposes a hard *selection* gate without
changing any existing SSIM/DeltaE/edge-F1 weights or release thresholds.

The gate is intentionally applicable only to palette-like, non-photo,
non-gradient inputs.  A non-applicable input is not a failure.  An applicable
input whose component measurement cannot be produced is ``needs_review`` and
is never treated as a quality pass.

Candidate eligibility also preserves the repository's pre-existing release
invariant for filled geometry: a filled subpath must be geometrically closed.
A raster-exact but structurally open filled candidate sits below ordinary
component failures so the CC guard cannot promote an artifact the existing
release contract already rejects.  This is not a new quality threshold; it
mirrors the existing ``core_release_runner`` structural invariant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from defusedxml import ElementTree as SafeET

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

# Selection-only bands.  Equal-status candidates retain their legacy score
# differences.  Structural-invalid filled paths are deliberately below ordinary
# CC failures because the repository release contract already rejects them.
_FAIL_SCORE_OFFSET = 1000.0
_STRUCTURAL_FAIL_SCORE_OFFSET = 1500.0
_UNMEASURED_SCORE_OFFSET = 2000.0

_STYLE_FILL = re.compile(r"(?:^|;)\s*fill\s*:\s*([^;]+)", re.I)
_PATH_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_PATH_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")
_PATH_GROUP = {"L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7}
_CLOSE_EPS = 1e-6


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


def _path_fill(element: Any) -> str:
    direct = element.attrib.get("fill")
    if direct is not None:
        return str(direct).strip().lower()
    style = str(element.attrib.get("style") or "")
    match = _STYLE_FILL.search(style)
    return match.group(1).strip().lower() if match else "black"


def _filled_path_subpaths_closed(d: str) -> bool:
    """Dependency-free geometric closure check for SVG path subpaths.

    The repository has workflows that deliberately install only a minimal
    measurement dependency set.  Winner scoring therefore cannot import the
    optional ``svgpathtools`` package at module import time.  For the release
    invariant we only need endpoint semantics, so this parser tracks the SVG
    command endpoint for every subpath, supports relative commands, explicit Z,
    and the tracer pattern where the final curve endpoint returns to the M point
    without emitting Z.
    """
    tokens = list(_PATH_CMD_RE.finditer(d or ""))
    if not tokens:
        return False

    cur = (0.0, 0.0)
    start: tuple[float, float] | None = None
    explicit_closed = False
    saw_segment = False
    closed: list[bool] = []

    def _finish() -> None:
        nonlocal start, explicit_closed, saw_segment
        if start is None:
            return
        endpoint_closed = (
            abs(cur[0] - start[0]) <= _CLOSE_EPS
            and abs(cur[1] - start[1]) <= _CLOSE_EPS
        )
        closed.append(bool(saw_segment and (explicit_closed or endpoint_closed)))
        start = None
        explicit_closed = False
        saw_segment = False

    for match in tokens:
        cmd = match.group(1)
        c = cmd.upper()
        rel = cmd.islower()
        try:
            nums = [float(value) for value in _PATH_NUM_RE.findall(match.group(2))]
        except ValueError:
            return False

        if c == "M":
            if len(nums) < 2 or len(nums) % 2:
                return False
            _finish()
            for index in range(0, len(nums), 2):
                x, y = nums[index], nums[index + 1]
                if rel:
                    x += cur[0]
                    y += cur[1]
                cur = (x, y)
                if index == 0:
                    start = cur
                else:
                    saw_segment = True  # extra moveto pairs are implicit lineto
            continue

        if start is None:
            return False
        if c == "Z":
            if nums:
                return False
            explicit_closed = True
            saw_segment = True
            cur = start
            continue

        group = _PATH_GROUP.get(c)
        if group is None or not nums or len(nums) % group:
            return False
        for index in range(0, len(nums), group):
            values = nums[index:index + group]
            x, y = cur
            if c == "H":
                x = values[0] + (cur[0] if rel else 0.0)
            elif c == "V":
                y = values[0] + (cur[1] if rel else 0.0)
            else:
                # L/T endpoints use their final pair; C/S/Q/A likewise encode
                # their endpoint in the final pair of each segment group.
                x, y = values[-2], values[-1]
                if rel:
                    x += cur[0]
                    y += cur[1]
            cur = (x, y)
            saw_segment = True

    _finish()
    return bool(closed) and all(closed)


def has_open_required_cycle(svg_path: Path) -> bool:
    """Mirror the existing release invariant for geometrically open fills.

    A literal ``Z`` is not mandatory: production tracers can serialize a closed
    curve by returning its final endpoint to the first point.  Open stroke-only
    paths remain valid.  Parsing errors fail closed.
    """
    try:
        root = SafeET.parse(str(svg_path)).getroot()
    except Exception:
        return True
    for element in root.iter():
        if str(element.tag).split("}")[-1].lower() != "path":
            continue
        if _path_fill(element) == "none":
            continue
        d = str(element.attrib.get("d") or "")
        if not _filled_path_subpaths_closed(d):
            return True
    return False


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
                "open_required_cycle": has_open_required_cycle(Path(svg_path)),
            }
        report = measure_component_integrity_arrays(reference, rendered)
    except Exception as exc:  # noqa: BLE001
        return {
            "applicable": True,
            "measured": False,
            "status": "needs_review",
            "reason": f"measurement_error:{type(exc).__name__}",
            "open_required_cycle": has_open_required_cycle(Path(svg_path)),
        }

    if report.get("status") == "not_applicable":
        # Source itself proved continuous-tone.  This is a legitimate
        # applicability outcome, not a missing measurement.
        return {"applicable": False, **report}

    open_cycle = has_open_required_cycle(Path(svg_path))
    result = {
        "applicable": True,
        **report,
        "component_measurement_status": report.get("status"),
        "open_required_cycle": open_cycle,
    }
    if open_cycle:
        # Preserve the pre-existing release structural invariant.  Keep all CC
        # metrics for diagnostics, but this artifact cannot be promoted by the
        # new winner gate simply because its raster happens to match perfectly.
        result["status"] = "fail"
        result["reason"] = "open_required_cycle"
    return result


def gate_candidate_scores(
    total_score: float,
    fidelity_score: float | None,
    component_report: dict[str, Any],
) -> dict[str, Any]:
    """Apply the independent CC/structure eligibility bands for selection.

    Passing/non-applicable candidates keep their exact legacy score.  Ordinary
    measured CC failures share a constant band, preserving their old relative
    ordering.  A pre-existing release-structural failure (open filled cycle) is
    lower than that band, and missing applicable measurement is lowest and
    carries ``needs_review``.  No existing fidelity/release threshold changes.
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
    if status == "needs_review" or not component_report.get("measured"):
        offset = _UNMEASURED_SCORE_OFFSET
    elif component_report.get("reason") == "open_required_cycle" or component_report.get("open_required_cycle"):
        offset = _STRUCTURAL_FAIL_SCORE_OFFSET
    else:
        offset = _FAIL_SCORE_OFFSET
    result["total_score"] = float(total_score) - offset
    result["fidelity_score"] = (
        float(fidelity_score) - offset if fidelity_score is not None else None
    )
    return result
