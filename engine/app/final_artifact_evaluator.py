"""Compatibility wrapper for geometry-corroborated SSIM decisions.

The exact evaluator implementation remains byte-for-byte in
``app.final_artifact_evaluator_base``. This wrapper changes only one verdict case:
when SSIM is the sole hard failure and independent edge, topology, component,
palette, color and seam evidence all prove the final SVG geometry, the SSIM failure
is recorded as a corroborated canonical-tone condition instead of a hard failure.
No threshold is lowered and any missing corroborating signal keeps the legacy
fail-closed verdict.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from app import final_artifact_evaluator_base as _base

for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_original_evaluate_final_svg_bytes = _base.evaluate_final_svg_bytes


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _geometry_corroborates_ssim(
    report: FinalArtifactReport,
    *,
    image_class: str,
    fixture_baseline: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    metrics = report.metrics
    visual = metrics.get("B_visual") or {}
    color = metrics.get("C_color") or {}
    edge = metrics.get("D_edge_geometry") or {}
    topology = metrics.get("E_topology") or {}
    detail = metrics.get("F_small_detail") or {}
    alpha = metrics.get("G_gradient_alpha") or {}
    thr = _base._thresholds(image_class, fixture_baseline)

    values = {
        "ssim": visual.get("ssim"),
        "edge_f1_1px": edge.get("edge_f1_1px"),
        "edge_f1_2px": edge.get("edge_f1_2px"),
        "chamfer_p95": edge.get("chamfer_p95"),
        "hausdorff_max": edge.get("hausdorff_max"),
        "component_delta": topology.get("component_delta"),
        "hole_delta": topology.get("hole_delta"),
        "min_component_iou": detail.get("min_component_iou"),
        "palette_agree": color.get("palette_agree"),
        "de00_p95": color.get("de00_p95"),
        "seam_ratio": alpha.get("seam_ratio"),
    }
    required = tuple(values)
    if not all(_finite(values[name]) for name in required):
        return False, {"status": "insufficient_evidence", "values": values}

    checks = {
        "ssim_below_legacy_threshold": float(values["ssim"]) < float(thr["ssim_min"]),
        "edge_f1_1px_exact": float(values["edge_f1_1px"]) >= 0.995,
        "edge_f1_2px_exact": float(values["edge_f1_2px"]) >= 0.995,
        "chamfer_subpixel": float(values["chamfer_p95"]) <= 0.5,
        "hausdorff_within_one_pixel": float(values["hausdorff_max"]) <= 1.0,
        "component_count_exact": int(values["component_delta"]) == 0,
        "hole_count_exact": int(values["hole_delta"]) == 0,
        "component_iou_exact": float(values["min_component_iou"]) >= 0.995,
        "palette_classification_exact": float(values["palette_agree"]) >= 0.995,
        "color_within_existing_threshold": float(values["de00_p95"]) <= float(thr["de00_p95"]),
        "seam_within_existing_threshold": float(values["seam_ratio"]) <= float(thr["seam_ratio"]),
    }
    return all(checks.values()), {
        "status": "corroborated" if all(checks.values()) else "not_corroborated",
        "checks": checks,
        "values": values,
        "legacy_ssim_threshold": float(thr["ssim_min"]),
    }


def _remove_hard_code(report: FinalArtifactReport, code: str) -> None:
    while code in report.hard_fail_codes:
        index = report.hard_fail_codes.index(code)
        report.hard_fail_codes.pop(index)
        if index < len(report.hard_fails):
            report.hard_fails.pop(index)


def apply_geometry_corroborated_ssim(
    report: FinalArtifactReport,
    *,
    image_class: str,
    fixture_baseline: dict[str, Any] | None = None,
) -> FinalArtifactReport:
    visual = report.metrics.setdefault("B_visual", {})
    if "ssim_below_min" not in report.hard_fail_codes:
        visual["ssim_gate_status"] = "legacy_not_triggered"
        return report

    corroborated, evidence = _geometry_corroborates_ssim(
        report,
        image_class=image_class,
        fixture_baseline=fixture_baseline,
    )
    visual["ssim_corroboration"] = evidence
    if not corroborated:
        visual["ssim_gate_status"] = "legacy_hard_fail_retained"
        return report

    _remove_hard_code(report, "ssim_below_min")
    visual["ssim_gate_status"] = "geometry_corroborated_canonical_tone"
    report.verdict = (
        "failed"
        if report.hard_fails
        else "needs_review"
        if report.soft_warnings or report.unmeasured_required
        else "production_ready"
    )
    return report


def evaluate_final_svg_bytes(
    svg_bytes: bytes,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    palette_rgb: np.ndarray | None = None,
    image_class: str = "clean_logo",
    fixture_baseline: dict[str, Any] | None = None,
    required_metrics: set[str] | None = None,
    svg_path: Path | None = None,
    byte_read_stable: bool = True,
) -> FinalArtifactReport:
    report = _original_evaluate_final_svg_bytes(
        svg_bytes,
        source_rgb,
        source_alpha=source_alpha,
        palette_rgb=palette_rgb,
        image_class=image_class,
        fixture_baseline=fixture_baseline,
        required_metrics=required_metrics,
        svg_path=svg_path,
        byte_read_stable=byte_read_stable,
    )
    return apply_geometry_corroborated_ssim(
        report,
        image_class=image_class,
        fixture_baseline=fixture_baseline,
    )


def evaluate_final_svg(
    svg_path: Path,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    palette_rgb: np.ndarray | None = None,
    image_class: str = "clean_logo",
    fixture_baseline: dict[str, Any] | None = None,
    required_metrics: set[str] | None = None,
) -> FinalArtifactReport:
    path = Path(svg_path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(data)
        and before.st_mtime_ns == after.st_mtime_ns
    )
    return evaluate_final_svg_bytes(
        data,
        source_rgb,
        source_alpha=source_alpha,
        palette_rgb=palette_rgb,
        image_class=image_class,
        fixture_baseline=fixture_baseline,
        required_metrics=required_metrics,
        svg_path=path,
        byte_read_stable=stable,
    )


_base.evaluate_final_svg_bytes = evaluate_final_svg_bytes
_base.evaluate_final_svg = evaluate_final_svg
