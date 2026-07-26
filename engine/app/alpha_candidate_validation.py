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

import gc
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
_ALPHA_BACKGROUND_NAMES = ("white", "black", "checker")
_ALPHA_BACKGROUND_HIGHER_IS_BETTER = ("ssim", "ms_ssim")
_ALPHA_BACKGROUND_LOWER_IS_BETTER = ("rgb_mae", "rgb_p95")


def alpha_background_non_regression(
    parent_report: Any,
    candidate_report: Any,
) -> dict[str, Any]:
    """Prove an alpha transform does not degrade any existing background metric.

    No new threshold is introduced.  The exact FinalArtifactEvaluator metrics from
    the accepted parent and the alpha candidate are compared on white, black and
    checker backgrounds.  Every higher-is-better metric must be >= the parent and
    every lower-is-better metric must be <= the parent.  Missing/non-finite evidence
    is fail-closed.
    """
    import math

    def _backgrounds(report: Any) -> dict[str, Any]:
        metrics = getattr(report, "metrics", None)
        if not isinstance(metrics, dict):
            raise RuntimeError("source_alpha_background_report_unmeasured")
        group = metrics.get("G_gradient_alpha") or {}
        backgrounds = group.get("backgrounds") or {}
        if not isinstance(backgrounds, dict):
            raise RuntimeError("source_alpha_background_report_unmeasured")
        return backgrounds

    parent_backgrounds = _backgrounds(parent_report)
    candidate_backgrounds = _backgrounds(candidate_report)
    comparisons: dict[str, Any] = {}
    failure_codes: list[str] = []
    for background in _ALPHA_BACKGROUND_NAMES:
        parent = parent_backgrounds.get(background) or {}
        candidate = candidate_backgrounds.get(background) or {}
        values: dict[str, Any] = {}
        for metric in (
            *_ALPHA_BACKGROUND_HIGHER_IS_BETTER,
            *_ALPHA_BACKGROUND_LOWER_IS_BETTER,
        ):
            parent_value = parent.get(metric)
            candidate_value = candidate.get(metric)
            if not isinstance(parent_value, (int, float)) or isinstance(parent_value, bool):
                raise RuntimeError(
                    f"source_alpha_background_metric_unmeasured:{background}:{metric}:parent"
                )
            if not isinstance(candidate_value, (int, float)) or isinstance(candidate_value, bool):
                raise RuntimeError(
                    f"source_alpha_background_metric_unmeasured:{background}:{metric}:candidate"
                )
            parent_number = float(parent_value)
            candidate_number = float(candidate_value)
            if not math.isfinite(parent_number) or not math.isfinite(candidate_number):
                raise RuntimeError(
                    f"source_alpha_background_metric_nonfinite:{background}:{metric}"
                )
            if metric in _ALPHA_BACKGROUND_HIGHER_IS_BETTER:
                passed = candidate_number >= parent_number
            else:
                passed = candidate_number <= parent_number
            values[metric] = {
                "parent": parent_number,
                "candidate": candidate_number,
                "non_regressing": bool(passed),
            }
            if not passed:
                failure_codes.append(
                    f"source_alpha_{background}_{metric}_regression"
                )
        comparisons[background] = values
    return {
        "source_alpha_background_non_regression": not failure_codes,
        "source_alpha_background_failure_codes": failure_codes,
        "source_alpha_background_comparison": comparisons,
        "source_alpha_background_authority": (
            "final_artifact_evaluator_parent_candidate_exact_non_regression"
        ),
    }


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


def validate_alpha_reconstruction_contract(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
    parent_path: Path | None = None,
) -> dict[str, Any]:
    """Validate owned alpha-plane metrics twice and preserve candidate geometry."""
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import (  # noqa: PLC0415
        _structure_check,
        _thresholds,
        evaluate_final_svg,
    )

    # A production request may invoke this contract more than once while trying a
    # bounded fallback and then compacting the accepted artifact. Release completed
    # renderer/evaluator allocations before the next exact pass; gates are unchanged.
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
        "source_coverage": float(direct_metrics["source_coverage"]),
        "render_coverage": float(direct_metrics["render_coverage"]),
    }

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

    # Bind the second alpha-plane measurement to the same bounded truth. Build
    # compact evaluator inputs, then release the first renderer result before the
    # heavier multi-background evaluator pass.
    source_rgb = _source_rgb_on_white(source_eval)
    source_alpha = np.ascontiguousarray(source_eval[:, :, 3])
    del rendered, direct_metrics, source_eval
    _release_transient_memory()

    # The following TransformJournal remains authoritative for RGB appearance and
    # every parent-relative structural/visual/topology/seam/complexity regression.
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

    background_proof: dict[str, Any] = {
        "source_alpha_background_non_regression": False,
        "source_alpha_background_failure_codes": [
            "source_alpha_parent_background_evidence_missing"
        ],
        "source_alpha_background_comparison": {},
        "source_alpha_background_authority": "unmeasured",
    }
    if parent_path is not None:
        parent_report = evaluate_final_svg(
            Path(parent_path),
            source_rgb,
            source_alpha=source_alpha,
            image_class=image_class,
            required_metrics={"alpha_fidelity"},
        )
        background_proof = alpha_background_non_regression(
            parent_report, report
        )
        if not background_proof["source_alpha_background_non_regression"]:
            raise RuntimeError(
                "source_alpha_candidate_background_regression:"
                + ",".join(
                    background_proof["source_alpha_background_failure_codes"]
                )
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
        "source_truth_source_coverage": direct_values["source_coverage"],
        "source_truth_render_coverage": direct_values["render_coverage"],
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": float(evaluator_alpha_iou),
        "final_evaluator_alpha_mae": float(evaluator_alpha_mae),
        "final_evaluator_alpha_plane_hard_fail_codes": [],
        "final_evaluator_alpha_appearance_codes": appearance_codes,
        "final_evaluator_other_hard_fail_codes": other_hard_codes,
        "appearance_regression_authority": (
            "final_artifact_evaluator_parent_candidate_exact_non_regression"
            if background_proof["source_alpha_background_non_regression"]
            else "transform_journal_parent_delta"
        ),
        "non_alpha_regression_authority": (
            "transform_journal_verified_source_alpha_contract"
            if background_proof["source_alpha_background_non_regression"]
            else "transform_journal_parent_delta"
        ),
        **background_proof,
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
    }
    if parent_path is not None:
        del parent_report
    del report, root, source_rgb, source_alpha
    _release_transient_memory()
    return result
