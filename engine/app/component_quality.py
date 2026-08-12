"""Connected-component integrity metrics for AI-2 winner safety.

The component gate is independent from perceptual fidelity.  It never rewrites
``total_score``/``fidelity_score``: selection uses an explicit safe-candidate
pool, so benchmark/evaluator metrics remain the real measured values.

For palette-like inputs, source/render components are paired with a deterministic
maximum-weight one-to-one assignment.  This prevents greedy ordering from hiding
micro-component loss, fragmentation, or invented islands.  Photo/gradient inputs
remain explicitly outside the gate.  Missing applicable measurement is
``needs_review`` and never counts as a safe pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from defusedxml import ElementTree as SafeET

_MIN_COMPONENT_PIXELS = 48
_MIN_COMPONENT_FRACTION = 0.0001
_MAX_SOURCE_PALETTE_MEDIAN_RESIDUAL = 10.0
_KMEANS_K = 6
_KMEANS_SEED = 7

_REQUIRED_SOURCE_CC_RECALL = 1.0
_REQUIRED_RENDER_CC_PRECISION = 1.0
_REQUIRED_MIN_TRUE_CC_IOU = 0.95

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
                    saw_segment = True
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
                x, y = values[-2], values[-1]
                if rel:
                    x += cur[0]
                    y += cur[1]
            cur = (x, y)
            saw_segment = True

    _finish()
    return bool(closed) and all(closed)


def has_open_required_cycle(svg_path: Path) -> bool:
    """Diagnostic mirror of the repository's explicit/endpoint closure invariant."""
    try:
        root = SafeET.parse(str(svg_path)).getroot()
    except Exception:
        return True
    for element in root.iter():
        if str(element.tag).split("}")[-1].lower() != "path":
            continue
        if _path_fill(element) == "none":
            continue
        if not _filled_path_subpaths_closed(str(element.attrib.get("d") or "")):
            return True
    return False


def _source_palette_classes(
    original_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    k: int = _KMEANS_K,
) -> tuple[np.ndarray, np.ndarray, float]:
    if original_rgb.shape != rendered_rgb.shape:
        h, w = original_rgb.shape[:2]
        rendered_rgb = cv2.resize(rendered_rgb, (w, h), interpolation=cv2.INTER_AREA)

    lab_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab_o.reshape(-1, 3)
    step = max(1, samples.shape[0] // 40000)
    sub = samples[::step]
    distinct = int(np.unique(sub.astype(np.uint8), axis=0).shape[0])
    k_eff = max(1, min(int(k), distinct, int(sub.shape[0])))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(_KMEANS_SEED)
    _compact, _labels, centers = cv2.kmeans(sub, k_eff, None, crit, 1, cv2.KMEANS_PP_CENTERS)

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
        out.append(_Component(class_index, label_index, area, (x, y, w, h), labels == label_index))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(a & b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(intersection) / float(max(1, union))


def _hungarian_minimize(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Deterministic rectangular Hungarian assignment, O(n^2 m), no optional deps."""
    if not cost or not cost[0]:
        return []
    rows = len(cost)
    cols = len(cost[0])
    transposed = rows > cols
    matrix = [list(row) for row in cost]
    if transposed:
        matrix = [[cost[i][j] for i in range(rows)] for j in range(cols)]
        rows, cols = cols, rows

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, cols + 1):
        if p[j] == 0:
            continue
        r, c = p[j] - 1, j - 1
        pairs.append((c, r) if transposed else (r, c))
    pairs.sort()
    return pairs


def _optimal_one_to_one_matches(
    sources: list[_Component],
    renders: list[_Component],
) -> tuple[list[tuple[int, int, float]], list[float]]:
    """Maximum-total-IoU one-to-one assignment within one palette class."""
    if not sources:
        return [], []
    weights = [[_iou(source.mask, render.mask) for render in renders] for source in sources]
    if not renders:
        return [], [0.0 for _ in sources]
    max_weight = max((max(row) for row in weights), default=0.0)
    cost = [[max_weight - value for value in row] for row in weights]
    assigned = _hungarian_minimize(cost)
    matches: list[tuple[int, int, float]] = []
    matched_iou = [0.0 for _ in sources]
    for si, ri in assigned:
        value = float(weights[si][ri])
        if value <= 0.0:
            continue
        matches.append((si, ri, value))
        matched_iou[si] = value
    return matches, matched_iou


def measure_component_integrity_arrays(
    original_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    *,
    k: int = _KMEANS_K,
) -> dict[str, Any]:
    try:
        cls_o, cls_r, palette_residual = _source_palette_classes(original_rgb, rendered_rgb, k=k)
    except Exception as exc:  # noqa: BLE001
        return {"measured": False, "status": "needs_review", "reason": f"classification_error:{type(exc).__name__}"}

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
    source_total = render_total = matched_source = matched_render = 0
    source_match_ious: list[float] = []
    class_reports: list[dict[str, Any]] = []

    for class_index in range(class_count):
        sources = _components(cls_o, class_index, min_area)
        renders = _components(cls_r, class_index, min_area)
        if not sources and not renders:
            continue
        matches, matched_iou = _optimal_one_to_one_matches(sources, renders)
        source_total += len(sources)
        render_total += len(renders)
        matched_source += len(matches)
        matched_render += len(matches)
        source_match_ious.extend(matched_iou)
        class_reports.append({
            "class_index": class_index,
            "source_components": len(sources),
            "render_components": len(renders),
            "matched_components": len(matches),
            "min_source_matched_iou": round(min(matched_iou), 4) if matched_iou else None,
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
    render_precision = float(matched_render) / float(render_total) if render_total > 0 else 0.0
    min_true_iou = min(source_match_ious) if source_match_ious else 0.0
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
    applicability = component_gate_applicability(mode, analysis)
    if not applicability["applicable"]:
        return {"applicable": False, "measured": False, "status": "not_applicable", "reason": applicability["reason"]}

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
        return {"applicable": False, **report}

    return {
        "applicable": True,
        **report,
        "component_measurement_status": report.get("status"),
        "open_required_cycle": has_open_required_cycle(Path(svg_path)),
    }


def gate_candidate_scores(
    total_score: float,
    fidelity_score: float | None,
    component_report: dict[str, Any],
) -> dict[str, Any]:
    """Attach selection eligibility without mutating measured score values."""
    status = str(component_report.get("status") or "needs_review")
    applicable = bool(component_report.get("applicable"))
    safe = (not applicable) or status == "pass"
    return {
        "total_score": float(total_score),
        "fidelity_score": fidelity_score,
        "selection_disqualified": not safe,
        "selection_safe": safe,
        "component_quality_status": status,
    }
