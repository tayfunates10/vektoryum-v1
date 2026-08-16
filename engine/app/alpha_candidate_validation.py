"""Composite-aware validation for renderer-native source-alpha reconstruction.

Alpha reconstruction owns both the source alpha plane and the visible RGBA
result produced when that alpha is composited onto real backgrounds.  The
existing alpha IoU/MAE thresholds remain unchanged.  In addition, the worst
normalized RGB mean-absolute residual across white, black and checker
backgrounds must not exceed one percent.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _source_rgb_on_white
from app.source_truth import (
    alpha_plane_metrics,
    multibackground_pairs,
    render_svg_to_rgba,
    resize_rgba,
)

_ALPHA_PLANE_FAILURE_CODES = {
    "alpha_iou_below_min",
    "alpha_mae_above_max",
}
_ALPHA_APPEARANCE_PREFIXES = (
    "alpha_white_",
    "alpha_black_",
    "alpha_checker_",
)
_VISIBLE_COMPOSITE_RESIDUAL_MAX = 0.01


def _release_transient_memory() -> None:
    """Best-effort release between repeated evaluator passes in one worker."""
    gc.collect()
    try:
        import ctypes

        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)
    except (AttributeError, OSError):
        pass


def visible_composite_residual_metrics(
    source_rgba: np.ndarray,
    rendered_rgba: np.ndarray,
) -> dict[str, Any]:
    """Measure visible RGB residual after compositing onto three backgrounds."""
    backgrounds: dict[str, dict[str, float]] = {}
    for name, (source_rgb, rendered_rgb) in multibackground_pairs(
        source_rgba, rendered_rgba
    ).items():
        diff = np.abs(
            source_rgb.astype(np.float32) - rendered_rgb.astype(np.float32)
        ) / 255.0
        backgrounds[name] = {
            "rgb_mae": float(diff.mean()),
            "rgb_p95": float(np.percentile(diff, 95)),
            "rgb_max": float(diff.max(initial=0.0)),
        }

    return {
        "visible_composite_residual": float(
            max((item["rgb_mae"] for item in backgrounds.values()), default=0.0)
        ),
        "visible_composite_residual_p95": float(
            max((item["rgb_p95"] for item in backgrounds.values()), default=0.0)
        ),
        "visible_composite_residual_max": float(
            max((item["rgb_max"] for item in backgrounds.values()), default=0.0)
        ),
        "visible_composite_backgrounds": backgrounds,
    }


def validate_alpha_reconstruction_contract(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
) -> dict[str, Any]:
    """Validate alpha-plane and visible-composite fidelity without budget changes."""
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import (  # noqa: PLC0415
        _boundary_offsets,
        _color_metrics,
        _structure_check,
        _thresholds,
        evaluate_final_svg,
    )

    _release_transient_memory()

    source_height, source_width = source_rgba_full.shape[:2]
    eval_scale = min(1.0, 512.0 / float(max(source_width, source_height)))
    eval_width = max(1, int(round(source_width * eval_scale)))
    eval_height = max(1, int(round(source_height * eval_scale)))
    source_eval = resize_rgba(source_rgba_full, eval_width, eval_height)
    rendered = render_svg_to_rgba(candidate_path, eval_width, eval_height)
    if rendered is None:
        raise RuntimeError("source_alpha_candidate_knockout_render_unmeasured")
    if rendered.shape[:2] != (eval_height, eval_width):
        rendered = resize_rgba(rendered, eval_width, eval_height)

    direct_metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered[:, :, 3])
    direct_values = {
        "alpha_iou": float(direct_metrics["alpha_iou"]),
        "alpha_mae": float(direct_metrics["alpha_mae"]),
        "alpha_p95": float(direct_metrics["alpha_p95"]),
        "alpha_max": float(direct_metrics["alpha_max"]),
        "source_coverage": float(direct_metrics["source_coverage"]),
        "render_coverage": float(direct_metrics["render_coverage"]),
    }
    composite_values = visible_composite_residual_metrics(source_eval, rendered)

    composite_de00_p95 = 0.0
    for source_rgb, rendered_rgb in multibackground_pairs(
        source_eval, rendered
    ).values():
        composite_de00_p95 = max(
            composite_de00_p95,
            float(_color_metrics(source_rgb, rendered_rgb)["de00_p95"]),
        )

    boundary = _boundary_offsets(
        np.asarray(source_eval[:, :, 3] > 0, dtype=bool),
        np.asarray(rendered[:, :, 3] > 0, dtype=bool),
    )
    boundary_p95 = (
        None if boundary is None else float(boundary["hausdorff_p95"])
    )

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    if direct_values["alpha_iou"] < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_iou_gate_failed:"
            f"{direct_values['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if direct_values["alpha_mae"] > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_mae_gate_failed:"
            f"{direct_values['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )
    if (
        float(composite_values["visible_composite_residual"])
        > _VISIBLE_COMPOSITE_RESIDUAL_MAX
    ):
        raise RuntimeError(
            "source_alpha_candidate_knockout_visible_composite_residual_failed:"
            f"{composite_values['visible_composite_residual']:.6f}>"
            f"{_VISIBLE_COMPOSITE_RESIDUAL_MAX:.6f}"
        )

    source_rgb = _source_rgb_on_white(source_eval)
    source_alpha = np.ascontiguousarray(source_eval[:, :, 3])
    del rendered, direct_metrics
    _release_transient_memory()

    report = evaluate_final_svg(
        candidate_path,
        source_rgb,
        source_alpha=source_alpha,
        image_class=image_class,
        required_metrics={"alpha_fidelity"},
    )
    alpha_group = report.metrics.get("G_gradient_alpha") or {}
    evaluator_alpha_iou = alpha_group.get("alpha_iou")
    evaluator_alpha_mae = alpha_group.get("alpha_mae")
    if evaluator_alpha_iou is None or evaluator_alpha_mae is None:
        raise RuntimeError(
            "source_alpha_candidate_knockout_evaluator_rejected:"
            "alpha_plane_unmeasured"
        )

    plane_failure_codes = [
        code for code in report.hard_fail_codes
        if code in _ALPHA_PLANE_FAILURE_CODES
    ]
    if float(evaluator_alpha_iou) < float(thresholds["alpha_iou_min"]):
        if "alpha_iou_below_min" not in plane_failure_codes:
            plane_failure_codes.append("alpha_iou_below_min")
    if float(evaluator_alpha_mae) > float(thresholds["alpha_mae_max"]):
        if "alpha_mae_above_max" not in plane_failure_codes:
            plane_failure_codes.append("alpha_mae_above_max")
    if plane_failure_codes:
        raise RuntimeError(
            "source_alpha_candidate_knockout_evaluator_rejected:"
            + ",".join(plane_failure_codes)
        )

    structure, _messages, structure_codes, root = _structure_check(
        Path(candidate_path).read_bytes()
    )
    if structure_codes or root is None:
        raise RuntimeError(
            "source_alpha_candidate_knockout_structure_failed:"
            + ",".join(structure_codes or ["parse_failed"])
        )
    after_counts = (
        int(structure.get("path_count") or 0),
        int(structure.get("node_count") or 0),
    )
    if after_counts != parent_counts:
        raise RuntimeError(
            "source_alpha_candidate_knockout_candidate_geometry_changed:"
            f"{parent_counts[0]}/{parent_counts[1]}->"
            f"{after_counts[0]}/{after_counts[1]}"
        )

    appearance_codes = [
        code for code in report.hard_fail_codes
        if code.startswith(_ALPHA_APPEARANCE_PREFIXES)
    ]
    other_hard_codes = [
        code for code in report.hard_fail_codes
        if code not in _ALPHA_PLANE_FAILURE_CODES
        and not code.startswith(_ALPHA_APPEARANCE_PREFIXES)
    ]
    result = {
        "source_truth_alpha_iou": direct_values["alpha_iou"],
        "source_truth_alpha_mae": direct_values["alpha_mae"],
        "source_truth_alpha_p95": direct_values["alpha_p95"],
        "source_truth_alpha_max": direct_values["alpha_max"],
        "source_truth_source_coverage": direct_values["source_coverage"],
        "source_truth_render_coverage": direct_values["render_coverage"],
        **composite_values,
        "visible_composite_residual_limit": _VISIBLE_COMPOSITE_RESIDUAL_MAX,
        "visible_composite_de00_p95": float(composite_de00_p95),
        "alpha_boundary_p95_px": boundary_p95,
        "visible_composite_gate_status": "passed",
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": float(evaluator_alpha_iou),
        "final_evaluator_alpha_mae": float(evaluator_alpha_mae),
        "final_evaluator_alpha_plane_hard_fail_codes": [],
        "final_evaluator_alpha_appearance_codes": appearance_codes,
        "final_evaluator_other_hard_fail_codes": other_hard_codes,
        "appearance_regression_authority": "transform_journal_parent_delta",
        "visible_composite_authority": "alpha_reconstruction_direct_composite_gate",
        "non_alpha_regression_authority": "transform_journal_parent_delta",
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }
    del report, root, source_rgb, source_alpha, source_eval
    _release_transient_memory()
    return result
