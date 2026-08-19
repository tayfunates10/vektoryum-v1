"""Vektoryum tanıları için deterministik residual-hata metrikleri.

Global SSIM/fidelity skorları alan ağırlıklıdır. Tek piksellik bir marka noktası
kaybolduğu hâlde global skor kusursuz görünebilir. Bu modül production
evaluator'ı değiştirmeden kalan hataya yalnız tanı amaçlı ve fail-closed bir
görünüm ekler:

* sınır ve iç bölgeye ayrılmış algısal residual alan (CIEDE2000);
* açık küçük-detay recall'u ile palet-duyarlı bağlı-bileşen eşleştirmesi;
* piksel cinsinden renk-sınırı yer değiştirmesi;
* transparanlık varsa alpha-düzlemi sadakati;
* ayrı engelleri ortalamayla gizlenemeyen sonlu near-zero sözleşmesi.

Sözleşme, "ölçülen corpus üzerinde ilan edilmiş toleransların dışında tespit
edilebilir hata kalmaması" anlamına gelir. Keyfi raster girdinin matematiksel
kayıpsız vektör olarak yeniden kurulabileceği iddiası değildir.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from app.final_artifact_evaluator import ciede2000
from app.graph_source import canonical_segmentation
from app.palette_ops import classify_rgb
from app.source_truth import alpha_plane_metrics, composite_rgba, render_svg_to_rgba

POLICY_VERSION = "vektoryum-near-zero-residual-v1"
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)

# Bunlar tanısal hedeflerdir; production release eşikleri değildir. İleride
# release kapısına yükseltilmeleri için nitelikli gerçek-dünya kanıtı ve ayrıca
# incelenmiş bir politika değişikliği gerekir.
NEAR_ZERO_TARGETS: dict[str, tuple[str, float]] = {
    "de00_p95": ("max", 2.3),
    "visible_error_ratio": ("max", 0.01),
    "boundary_p95_px": ("max", 0.75),
    "source_component_recall": ("min", 1.0),
    "render_component_precision": ("min", 1.0),
    "small_component_recall": ("min", 1.0),
    "min_component_iou": ("min", 0.95),
}


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        np.asarray(rgb, dtype=np.float32) / 255.0,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape[:2] == (height, width):
        return arr.copy()
    interpolation = cv2.INTER_AREA if width < arr.shape[1] else cv2.INTER_LANCZOS4
    return cv2.resize(arr, (int(width), int(height)), interpolation=interpolation)


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    lab = np.asarray(labels)
    edges = np.zeros(lab.shape, dtype=bool)
    edges[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edges[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    edges[1:, :] |= lab[1:, :] != lab[:-1, :]
    edges[:-1, :] |= lab[1:, :] != lab[:-1, :]
    return edges


def _boundary_distance(source_edges: np.ndarray, render_edges: np.ndarray) -> dict[str, float | None]:
    src = np.asarray(source_edges, dtype=bool)
    rnd = np.asarray(render_edges, dtype=bool)
    if not src.any() and not rnd.any():
        return {
            "boundary_mean_px": 0.0,
            "boundary_p95_px": 0.0,
            "boundary_max_px": 0.0,
        }
    if not src.any() or not rnd.any():
        return {
            "boundary_mean_px": None,
            "boundary_p95_px": None,
            "boundary_max_px": None,
        }
    to_render = cv2.distanceTransform((~rnd).astype(np.uint8), cv2.DIST_L2, 5)
    to_source = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([to_render[src], to_source[rnd]]).astype(np.float64)
    return {
        "boundary_mean_px": float(distances.mean()),
        "boundary_p95_px": float(np.percentile(distances, 95)),
        "boundary_max_px": float(distances.max(initial=0.0)),
    }


def _corner_background_label(labels: np.ndarray) -> int:
    h, w = labels.shape
    span_y = max(1, min(h, round(h * 0.04)))
    span_x = max(1, min(w, round(w * 0.04)))
    corners = np.concatenate(
        [
            labels[:span_y, :span_x].reshape(-1),
            labels[:span_y, -span_x:].reshape(-1),
            labels[-span_y:, :span_x].reshape(-1),
            labels[-span_y:, -span_x:].reshape(-1),
        ]
    )
    counts = np.bincount(corners.astype(np.int64))
    return int(np.argmax(counts))


def _components(labels: np.ndarray, class_id: int, min_area: int) -> list[dict[str, Any]]:
    mask = (labels == int(class_id)).astype(np.uint8)
    count, component_map, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    result: list[dict[str, Any]] = []
    for component_id in range(1, count):
        x, y, width, height, area = map(int, stats[component_id])
        if area < min_area:
            continue
        result.append(
            {
                "class_id": int(class_id),
                "component_id": int(component_id),
                "area": area,
                "bbox": [x, y, width, height],
                "map": component_map,
            }
        )
    return result


def _component_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly, lw, lh = left["bbox"]
    rx, ry, rw, rh = right["bbox"]
    x0, y0 = max(lx, rx), max(ly, ry)
    x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = 0
    if x1 > x0 and y1 > y0:
        left_mask = left["map"][y0:y1, x0:x1] == left["component_id"]
        right_mask = right["map"][y0:y1, x0:x1] == right["component_id"]
        intersection = int((left_mask & right_mask).sum())
    union = int(left["area"]) + int(right["area"]) - intersection
    return 1.0 if union == 0 else float(intersection / union)


def connected_component_fidelity(
    source_labels: np.ndarray,
    render_labels: np.ndarray,
    *,
    min_area: int | None = None,
    match_iou: float = 0.50,
    small_area: int | None = None,
) -> dict[str, Any]:
    """Gerçek bağlı bileşenleri palet sınıfında Hungarian IoU ile eşleştir.

    Her kaynak bileşen ayrı katılır. Böylece büyük bir şekil, aynı rengi kullanan
    kayıp bir noktayı gizleyemez.
    """
    source = np.asarray(source_labels, dtype=np.uint8)
    rendered = np.asarray(render_labels, dtype=np.uint8)
    if source.shape != rendered.shape:
        raise ValueError("kaynak/render etiket boyutları eşleşmeli")
    h, w = source.shape
    min_area = int(min_area or max(2, round(h * w * 0.000015)))
    small_area = int(small_area or max(64, round(h * w * 0.002)))
    background = _corner_background_label(source)
    classes = sorted(set(np.unique(source).tolist()) | set(np.unique(rendered).tolist()))

    source_count = render_count = 0
    matched_source = matched_render = 0
    source_ious: list[float] = []
    weighted_iou_sum = 0.0
    weighted_area = 0
    small_count = small_matched = 0
    worst: dict[str, Any] | None = None
    unmatched_source_area = 0
    unmatched_render_area = 0

    for class_id in classes:
        if int(class_id) == background:
            continue
        src_components = _components(source, int(class_id), min_area)
        rnd_components = _components(rendered, int(class_id), min_area)
        source_count += len(src_components)
        render_count += len(rnd_components)
        matrix = np.zeros((len(src_components), len(rnd_components)), dtype=np.float64)
        for source_index, src_component in enumerate(src_components):
            for render_index, rnd_component in enumerate(rnd_components):
                matrix[source_index, render_index] = _component_iou(src_component, rnd_component)

        assignments: dict[int, tuple[int, float]] = {}
        assigned_render: set[int] = set()
        if matrix.size:
            rows, cols = linear_sum_assignment(1.0 - matrix)
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                assignments[int(row)] = (int(col), float(matrix[row, col]))
                if float(matrix[row, col]) >= match_iou:
                    assigned_render.add(int(col))

        for source_index, component in enumerate(src_components):
            _render_index, iou = assignments.get(source_index, (-1, 0.0))
            source_ious.append(iou)
            weighted_iou_sum += iou * int(component["area"])
            weighted_area += int(component["area"])
            passes = iou >= match_iou
            matched_source += int(passes)
            if not passes:
                unmatched_source_area += int(component["area"])
            is_small = int(component["area"]) <= small_area
            if is_small:
                small_count += 1
                small_matched += int(passes)
            if worst is None or iou < float(worst["iou"]):
                worst = {
                    "class_id": int(class_id),
                    "area_px": int(component["area"]),
                    "bbox": list(component["bbox"]),
                    "iou": float(iou),
                    "small": bool(is_small),
                }

        matched_render += len(assigned_render)
        for render_index, component in enumerate(rnd_components):
            if render_index not in assigned_render:
                unmatched_render_area += int(component["area"])

    source_recall = 1.0 if source_count == 0 else matched_source / source_count
    render_precision = 1.0 if render_count == 0 else matched_render / render_count
    small_recall = 1.0 if small_count == 0 else small_matched / small_count
    return {
        "background_class_id": background,
        "min_component_area_px": min_area,
        "small_component_area_max_px": small_area,
        "source_component_count": source_count,
        "render_component_count": render_count,
        "matched_source_count": matched_source,
        "matched_render_count": matched_render,
        "unmatched_source_count": source_count - matched_source,
        "unmatched_render_count": render_count - matched_render,
        "unmatched_source_area_ratio": float(unmatched_source_area / max(1, h * w)),
        "unmatched_render_area_ratio": float(unmatched_render_area / max(1, h * w)),
        "source_component_recall": float(source_recall),
        "render_component_precision": float(render_precision),
        "small_component_count": small_count,
        "small_component_recall": float(small_recall),
        "min_component_iou": float(min(source_ious)) if source_ious else 1.0,
        "mean_component_iou": float(np.mean(source_ious)) if source_ious else 1.0,
        "area_weighted_component_iou": float(weighted_iou_sum / max(1, weighted_area)),
        "worst_component": worst,
    }


def _residual_regions(mask: np.ndarray, min_area: int) -> dict[str, Any]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8),
        connectivity=8,
    )
    areas = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
        reverse=True,
    )
    retained = [area for area in areas if area >= min_area]
    total = max(1, int(mask.size))
    return {
        "residual_region_count": len(retained),
        "largest_residual_region_ratio": float(retained[0] / total) if retained else 0.0,
        "top_residual_region_area_px": retained[:8],
    }


def measure_residual_error(
    source_rgba: np.ndarray,
    render_rgba: np.ndarray,
    *,
    palette_size: int = 8,
) -> dict[str, Any]:
    source = np.asarray(source_rgba, dtype=np.uint8)
    rendered = np.asarray(render_rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("kaynak RGBA olmalı")
    if rendered.ndim != 3 or rendered.shape[2] != 4:
        raise ValueError("render RGBA olmalı")
    if rendered.shape[:2] != source.shape[:2]:
        rendered = cv2.resize(
            rendered,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    source_rgb = composite_rgba(source, 255)
    render_rgb = composite_rgba(rendered, 255)
    unique_count = int(np.unique(source_rgb.reshape(-1, 3), axis=0).shape[0])
    k = max(1, min(int(palette_size), unique_count))
    source_labels, palette = canonical_segmentation(source_rgb, k=k)
    render_labels = classify_rgb(render_rgb, palette.astype(np.float32)).astype(np.uint8)

    delta = ciede2000(_lab(source_rgb), _lab(render_rgb)).astype(np.float32)
    visible = delta > 2.3
    severe = delta > 5.0
    source_edges = _label_boundaries(source_labels)
    render_edges = _label_boundaries(render_labels)
    edge_band = cv2.dilate(
        source_edges.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)
    residual_total = int(visible.sum())
    edge_pixels = int(edge_band.sum())
    interior_pixels = int((~edge_band).sum())
    region_min = max(2, round(source.shape[0] * source.shape[1] * 0.00003))

    result: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "palette_size": int(len(palette)),
        "de00_mean": float(delta.mean()),
        "de00_p50": float(np.percentile(delta, 50)),
        "de00_p95": float(np.percentile(delta, 95)),
        "de00_p99": float(np.percentile(delta, 99)),
        "de00_max": float(delta.max(initial=0.0)),
        "perceptual_exact_ratio": float((delta <= 1.0).mean()),
        "visible_error_ratio": float(visible.mean()),
        "severe_error_ratio": float(severe.mean()),
        "edge_error_ratio": float((visible & edge_band).sum() / max(1, edge_pixels)),
        "interior_error_ratio": float((visible & ~edge_band).sum() / max(1, interior_pixels)),
        "residual_edge_share": float((visible & edge_band).sum() / max(1, residual_total)),
    }
    result.update(_residual_regions(visible, region_min))
    result.update(_boundary_distance(source_edges, render_edges))
    result.update(connected_component_fidelity(source_labels, render_labels))

    source_has_alpha = bool((source[:, :, 3] < 255).any())
    result["source_has_alpha"] = source_has_alpha
    result["alpha"] = (
        alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
        if source_has_alpha
        else None
    )
    return result


def _check(code: str, value: object, direction: str, target: float) -> dict[str, Any]:
    measured = _finite(value)
    numeric = float(value) if measured else None
    passed = bool(
        measured
        and ((direction == "min" and numeric >= target) or (direction == "max" and numeric <= target))
    )
    if not measured:
        gap = None
    elif direction == "min":
        gap = max(0.0, target - numeric)
    else:
        gap = max(0.0, numeric - target)
    return {
        "code": code,
        "measured": measured,
        "value": numeric,
        "direction": direction,
        "target": float(target),
        "gap": gap,
        "passed": passed,
    }


def build_near_zero_contract(
    residual: dict[str, Any] | None,
    *,
    deterministic: bool | None,
    structural_failures: Iterable[str],
    multi_scale: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structural = list(structural_failures)
    checks: list[dict[str, Any]] = [
        {
            "code": "structural_safety",
            "measured": True,
            "value": len(structural),
            "direction": "max",
            "target": 0,
            "gap": len(structural),
            "passed": not structural,
        },
        {
            "code": "byte_determinism",
            "measured": deterministic is not None,
            "value": deterministic,
            "direction": "equals",
            "target": True,
            "gap": None if deterministic is None else int(not deterministic),
            "passed": deterministic is True,
        },
    ]
    for code, (direction, target) in NEAR_ZERO_TARGETS.items():
        checks.append(_check(code, (residual or {}).get(code), direction, target))

    alpha = (residual or {}).get("alpha")
    if (residual or {}).get("source_has_alpha"):
        checks.append(_check("alpha_iou", (alpha or {}).get("alpha_iou"), "min", 0.995))
        checks.append(_check("alpha_mae", (alpha or {}).get("alpha_mae"), "max", 0.005))

    if multi_scale is not None:
        checks.extend(
            [
                {
                    "code": "multi_scale_complete",
                    "measured": isinstance(multi_scale.get("all_scales_measured"), bool),
                    "value": multi_scale.get("all_scales_measured"),
                    "direction": "equals",
                    "target": True,
                    "gap": (
                        None
                        if not isinstance(multi_scale.get("all_scales_measured"), bool)
                        else int(not multi_scale["all_scales_measured"])
                    ),
                    "passed": multi_scale.get("all_scales_measured") is True,
                },
                _check(
                    "multi_scale_component_recall",
                    multi_scale.get("min_source_component_recall"),
                    "min",
                    1.0,
                ),
                _check(
                    "multi_scale_small_component_recall",
                    multi_scale.get("min_small_component_recall"),
                    "min",
                    1.0,
                ),
                _check(
                    "multi_scale_boundary_excess",
                    multi_scale.get("max_normalized_boundary_excess_px"),
                    "max",
                    0.25,
                ),
            ]
        )

    blockers = [str(item["code"]) for item in checks if not item["passed"]]
    return {
        "policy_version": POLICY_VERSION,
        "definition": "no_detectable_error_outside_finite_tolerances",
        "production_gate": False,
        "ready": not blockers,
        "blocker_codes": blockers,
        "checks": checks,
    }


def measure_multiscale_svg(
    svg_path: Any,
    source_builder: Callable[[int], Image.Image],
    *,
    base_size: int = 256,
    scales: Iterable[float] = SCALES,
) -> dict[str, Any]:
    requested_scales = tuple(float(scale) for scale in scales)
    levels: list[dict[str, Any]] = []
    missing: list[str] = []
    for scale in requested_scales:
        size = max(16, int(round(base_size * float(scale))))
        source = np.asarray(source_builder(size).convert("RGBA"), dtype=np.uint8).copy()
        rendered = render_svg_to_rgba(svg_path, size, size)
        label = f"{float(scale):g}x"
        if rendered is None:
            missing.append(label)
            continue
        metrics = measure_residual_error(source, rendered)
        boundary = metrics.get("boundary_p95_px")
        quantization_allowance = max(math.sqrt(2.0), 0.75 * float(scale))
        levels.append(
            {
                "scale": label,
                "size": size,
                "de00_p95": metrics["de00_p95"],
                "visible_error_ratio": metrics["visible_error_ratio"],
                "boundary_p95_px": boundary,
                "boundary_quantization_allowance_px": quantization_allowance,
                "normalized_boundary_excess_px": (
                    max(0.0, float(boundary) - quantization_allowance) / float(scale)
                    if _finite(boundary)
                    else None
                ),
                "source_component_recall": metrics["source_component_recall"],
                "render_component_precision": metrics["render_component_precision"],
                "small_component_recall": metrics["small_component_recall"],
                "min_component_iou": metrics["min_component_iou"],
                "unmatched_source_count": metrics["unmatched_source_count"],
                "unmatched_render_count": metrics["unmatched_render_count"],
            }
        )

    def finite_values(key: str) -> list[float]:
        return [float(level[key]) for level in levels if _finite(level.get(key))]

    component_recall = finite_values("source_component_recall")
    small_recall = finite_values("small_component_recall")
    normalized_boundary_excess = finite_values("normalized_boundary_excess_px")
    visible_ratio = finite_values("visible_error_ratio")
    return {
        "scales_requested": [f"{scale:g}x" for scale in requested_scales],
        "missing_scales": missing,
        "levels": levels,
        "min_source_component_recall": min(component_recall) if component_recall else None,
        "min_small_component_recall": min(small_recall) if small_recall else None,
        "max_normalized_boundary_excess_px": (
            max(normalized_boundary_excess) if normalized_boundary_excess else None
        ),
        "max_visible_error_ratio": max(visible_ratio) if visible_ratio else None,
        "all_scales_measured": not missing and len(levels) == len(requested_scales),
    }


def blocker_histogram(contracts: Iterable[dict[str, Any] | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for contract in contracts:
        counts.update((contract or {}).get("blocker_codes") or [])
    return dict(sorted(counts.items()))
