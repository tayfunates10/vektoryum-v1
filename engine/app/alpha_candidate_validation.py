"""Dual evaluator contract for renderer-native source-alpha reconstruction.

The alpha transform owns source-alpha fidelity. It therefore requires the
unchanged direct alpha IoU/MAE gates and the FinalArtifactEvaluator alpha group
on the same bounded evaluation grid. Absolute source-vs-vector SSIM, topology,
seam and color verdicts are not reinterpreted here: immediately after this
transaction the real TransformJournal compares the exact parent candidate with
the exact transformed candidate and rejects any regression under its unchanged
structural, visual, topology, seam and complexity policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.alpha_candidate_knockout import _source_rgb_on_white
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba


def validate_alpha_reconstruction_contract(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
) -> dict[str, Any]:
    """Validate owned alpha metrics twice and preserve exact candidate geometry."""
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

    # Bind the second evaluator to the same bounded alpha truth used by the
    # production alpha hard gate. The following TransformJournal stage remains
    # authoritative for parent-relative SSIM, topology, seam and complexity.
    report = evaluate_final_svg(
        candidate_path,
        _source_rgb_on_white(source_eval),
        source_alpha=source_eval[:, :, 3],
        image_class=image_class,
        required_metrics={"alpha_fidelity"},
    )
    alpha_group = report.metrics.get("G_gradient_alpha") or {}
    alpha_failure_codes = [
        code for code in report.hard_fail_codes if code.startswith("alpha_")
    ]
    if (
        alpha_group.get("alpha_fidelity_status") != "passed"
        or alpha_failure_codes
    ):
        codes = ",".join(alpha_failure_codes or ["alpha_fidelity_not_passed"])
        raise RuntimeError(
            f"source_alpha_candidate_knockout_evaluator_rejected:{codes}"
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

    non_alpha_hard_codes = [
        code for code in report.hard_fail_codes if not code.startswith("alpha_")
    ]
    return {
        "source_truth_alpha_iou": float(direct_metrics["alpha_iou"]),
        "source_truth_alpha_mae": float(direct_metrics["alpha_mae"]),
        "source_truth_source_coverage": float(direct_metrics["source_coverage"]),
        "source_truth_render_coverage": float(direct_metrics["render_coverage"]),
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_iou": float(alpha_group["alpha_iou"]),
        "final_evaluator_alpha_mae": float(alpha_group["alpha_mae"]),
        "final_evaluator_alpha_hard_fail_codes": list(alpha_failure_codes),
        "final_evaluator_non_alpha_hard_fail_codes": non_alpha_hard_codes,
        "non_alpha_regression_authority": "transform_journal_parent_delta",
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }
