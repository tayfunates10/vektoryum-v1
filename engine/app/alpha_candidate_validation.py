"""Dual evaluator contract for renderer-native source-alpha reconstruction.

The alpha transform owns source-alpha plane fidelity. It therefore requires the
unchanged direct alpha IoU/MAE gates and independently confirms the same two
metrics through FinalArtifactEvaluator on the bounded comparison grid.

FinalArtifactEvaluator also reports white, black and checker appearance under
``alpha_*`` codes. Those are RGB/color-composite judgements rather than alpha
plane measurements. They are recorded here but remain fail-closed in the
immediately following real TransformJournal parent-to-candidate comparison,
together with absolute SSIM, topology, seam, color and complexity policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _source_rgb_on_white
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba

_ALPHA_PLANE_FAILURE_CODES = {
    "alpha_iou_below_min",
    "alpha_mae_above_max",
}
_ALPHA_APPEARANCE_PREFIXES = (
    "alpha_white_",
    "alpha_black_",
    "alpha_checker_",
)


def validate_alpha_reconstruction_contract(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
) -> dict[str, Any]:
    """Validate owned alpha-plane metrics twice and preserve candidate geometry."""
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import (  # noqa: PLC0415
        _structure_check,
        _thresholds,
        evaluate_final_svg,
    )

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

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    if float(direct_metrics["alpha_iou"]) < float(thresholds["alpha_iou_min"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_iou_gate_failed:"
            f"{direct_metrics['alpha_iou']:.6f}<{thresholds['alpha_iou_min']}"
        )
    if float(direct_metrics["alpha_mae"]) > float(thresholds["alpha_mae_max"]):
        raise RuntimeError(
            "source_alpha_candidate_knockout_mae_gate_failed:"
            f"{direct_metrics['alpha_mae']:.6f}>{thresholds['alpha_mae_max']}"
        )

    # Bind the second alpha-plane measurement to the same bounded truth. The
    # following TransformJournal remains authoritative for RGB appearance and
    # every parent-relative structural/visual/topology/seam/complexity regression.
    report = evaluate_final_svg(
        candidate_path,
        _source_rgb_on_white(source_eval),
        source_alpha=source_eval[:, :, 3],
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
    return {
        "source_truth_alpha_iou": float(direct_metrics["alpha_iou"]),
        "source_truth_alpha_mae": float(direct_metrics["alpha_mae"]),
        "source_truth_source_coverage": float(direct_metrics["source_coverage"]),
        "source_truth_render_coverage": float(direct_metrics["render_coverage"]),
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": float(evaluator_alpha_iou),
        "final_evaluator_alpha_mae": float(evaluator_alpha_mae),
        "final_evaluator_alpha_plane_hard_fail_codes": [],
        "final_evaluator_alpha_appearance_codes": appearance_codes,
        "final_evaluator_other_hard_fail_codes": other_hard_codes,
        "appearance_regression_authority": "transform_journal_parent_delta",
        "non_alpha_regression_authority": "transform_journal_parent_delta",
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }
