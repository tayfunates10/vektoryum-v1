"""Terminal evaluator-grid retry for painter alpha reconstruction.

This module is deliberately isolated from the existing painter tournament.  It is
called only after every existing candidate family has failed, and only when an
existing paint-deficit candidate reached FinalArtifactEvaluator and was rejected
solely by unchanged alpha-plane quality gates.  It never changes evaluator or
TransformJournal policy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.alpha_artwork_identity import (
    alpha_transaction_id,
    artwork_fingerprint,
)

_FINAL_EVALUATOR_MAX_SIDE = 1024.0
_ALPHA_PLANE_FAILURE_CODES = {"alpha_iou_below_min", "alpha_mae_above_max"}

# Match the already-existing paint-deficit ladder.  The retry does not invent a
# coarser quality path: it only replays labels that already reached evaluator and
# failed there on the legacy painter grid.
_PAINT_DEFICIT_RETRY_SPECS: tuple[tuple[str, str, int], ...] = (
    ("paint-deficit-q24", "polygon", 24),
    ("paint-deficit-cumulative", "cumulative", 24),
    ("paint-deficit-cumulative-q20", "cumulative", 20),
    ("paint-deficit-cumulative-q16", "cumulative", 16),
    ("paint-deficit-cumulative-q12", "cumulative", 12),
    ("paint-deficit-cumulative-q8", "cumulative", 8),
)


def evaluator_aligned_source_rgba(source_rgba_full: np.ndarray) -> np.ndarray:
    """Sample source RGBA on FinalArtifactEvaluator's fresh-render grid.

    FinalArtifactEvaluator compares alpha after reducing the source to a
    source-aspect raster whose longest side is 1024, with standalone alpha
    resized by INTER_AREA, then freshly rasterizes the SVG at that exact size.
    The returned array mirrors that source-side sampling without changing any
    acceptance threshold or evaluator code.
    """
    source = np.asarray(source_rgba_full, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise RuntimeError("source_alpha_candidate_painter_evaluator_grid_invalid_rgba")

    source_height, source_width = source.shape[:2]
    scale = min(
        1.0,
        _FINAL_EVALUATOR_MAX_SIDE / float(max(source_width, source_height)),
    )
    eval_width = max(1, int(round(source_width * scale)))
    eval_height = max(1, int(round(source_height * scale)))
    if (eval_height, eval_width) == (source_height, source_width):
        return source.copy()

    aligned = cv2.resize(
        source,
        (eval_width, eval_height),
        interpolation=cv2.INTER_AREA,
    )
    aligned = np.asarray(aligned, dtype=np.uint8).copy()
    aligned[:, :, 3] = cv2.resize(
        source[:, :, 3],
        (eval_width, eval_height),
        interpolation=cv2.INTER_AREA,
    )
    return aligned


def alpha_only_evaluator_retry_labels(
    attempts: list[dict[str, Any]],
) -> set[str]:
    """Return legacy paint-deficit labels eligible for evaluator-grid replay."""
    prefix = "source_alpha_candidate_painter_evaluator_rejected:"
    allowed_labels = {label for label, _encoding, _levels in _PAINT_DEFICIT_RETRY_SPECS}
    labels: set[str] = set()
    for entry in attempts:
        if entry.get("status") != "evaluator_rejected":
            continue
        if entry.get("validation_stage") != "evaluator":
            continue
        label = str(entry.get("encoding_label") or "")
        if label not in allowed_labels:
            continue
        error_code = str(entry.get("exact_error_code") or "")
        if not error_code.startswith(prefix):
            continue
        reasons = {
            reason
            for reason in error_code[len(prefix):].split(",")
            if reason
        }
        if reasons and reasons.issubset(_ALPHA_PLANE_FAILURE_CODES):
            labels.add(label)
    return labels


def eligible_retry_specs(
    attempts: list[dict[str, Any]],
) -> tuple[tuple[str, str, int], ...]:
    """Return existing ladder entries that may be replayed, in legacy order."""
    labels = alpha_only_evaluator_retry_labels(attempts)
    return tuple(spec for spec in _PAINT_DEFICIT_RETRY_SPECS if spec[0] in labels)


def evaluate_evaluator_aligned_paint_deficit(
    *,
    attempts: list[dict[str, Any]],
    original_root: Any,
    canvas: Any,
    source_rgba_full: np.ndarray,
    parent_sha256: str,
    mode: str,
    parent_counts: tuple[int, int],
    limits: dict[str, Any],
    target: Path,
    excluded_from_parent: tuple[Any, ...],
    parent_journal_path: Path,
    journal_source_rgb: np.ndarray,
    journal_image_class: str,
    write_tree_to_temp: Callable[..., Path],
    assess_candidate: Callable[..., dict[str, Any]],
    run_geometry_journal: Callable[..., tuple[bool, list[str]]],
) -> list[Any] | None:
    """Replay evaluator-alpha-only failures on evaluator's own raster grid.

    Existing candidates and order are untouched.  This function is terminal and
    additive: it only sees labels that already failed the final evaluator on alpha
    IoU/MAE, rebuilds those same encodings on the final evaluator comparison grid,
    and sends every retry through the unchanged byte/native/bounded/evaluator,
    identity and TransformJournal gates.
    """
    specs = eligible_retry_specs(attempts)
    if not specs:
        return None

    from app.alpha_candidate_paint_deficit import (  # noqa: PLC0415
        build_paint_deficit_reconstruction_tree,
    )
    from app.alpha_svg_mask import _quantize_alpha  # noqa: PLC0415

    aligned_rgba = evaluator_aligned_source_rgba(source_rgba_full)
    aligned_alpha = np.asarray(aligned_rgba[:, :, 3], dtype=np.uint8).copy()
    aligned_alpha_sha256 = hashlib.sha256(
        np.ascontiguousarray(aligned_alpha).tobytes()
    ).hexdigest()
    _quantized, aligned_opacity = _quantize_alpha(aligned_alpha)
    aligned_source_level_count = len(
        [level for level in aligned_opacity if int(level) > 0]
    )
    byte_limit = int(limits["byte_limit"])

    for order, (base_label, mask_encoding, levels) in enumerate(specs):
        label = f"{base_label}-evaluator-aligned"
        txn = alpha_transaction_id(
            parent_sha256,
            aligned_alpha_sha256,
            mode,
            label,
        )
        entry: dict[str, Any] = {
            "stroke_width": 0.0,
            "encoding_label": label,
            "encoding_family": "paint_deficit",
            "exact_or_quantized": "evaluator_aligned",
            "source_alpha_level_count": int(aligned_source_level_count),
            "encoded_alpha_level_count": int(levels),
            "actual_serialized_bytes": None,
            "byte_limit": byte_limit,
            "projected_path_count": int(parent_counts[0]),
            "actual_path_count": None,
            "path_limit": int(limits["path_limit"]),
            "projected_node_count": int(parent_counts[1]),
            "actual_node_count": None,
            "node_limit": int(limits["node_limit"]),
            "preflight_status": None,
            "validation_started": False,
            "validation_stage": None,
            "status": None,
            "exact_error_code": "",
            "native_alpha_iou": None,
            "native_alpha_mae": None,
            "bounded_alpha_iou": None,
            "bounded_alpha_mae": None,
            "evaluator_alpha_iou": None,
            "evaluator_alpha_mae": None,
            "artwork_fingerprint_match": None,
            "journal_gate_started": False,
            "journal_passed": None,
            "journal_reason_codes": [],
            "retry_alignment": "final_evaluator_comparison_grid",
            "retry_grid_width": int(aligned_alpha.shape[1]),
            "retry_grid_height": int(aligned_alpha.shape[0]),
        }

        try:
            probe_root, probe_geometry = build_paint_deficit_reconstruction_tree(
                original_root,
                canvas,
                aligned_rgba,
                txn,
                mask_encoding=mask_encoding,
                levels=levels,
            )
        except RuntimeError as exc:
            entry["preflight_status"] = "not_constructed"
            entry["validation_stage"] = "paint_deficit_geometry"
            entry["status"] = "geometry_rejected"
            entry["exact_error_code"] = str(exc)
            attempts.append(entry)
            continue

        probe_temp = write_tree_to_temp(probe_root, target)
        probe_size = int(probe_temp.stat().st_size)
        entry["actual_serialized_bytes"] = probe_size
        entry["encoded_alpha_level_count"] = int(
            probe_geometry.get("encoded_alpha_level_count", levels)
        )
        for stat_key in (
            "paint_deficit_pixel_count",
            "source_component_count",
            "anchored_source_component_count",
            "detached_source_component_count",
        ):
            if stat_key in probe_geometry:
                entry[stat_key] = int(probe_geometry[stat_key])

        if probe_size > byte_limit:
            entry["preflight_status"] = "over_budget"
            entry["status"] = "byte_rejected"
            entry["exact_error_code"] = (
                "source_alpha_candidate_painter_byte_budget_rejected:"
                f"{label}:{probe_size}>{byte_limit}"
            )
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            continue

        entry["preflight_status"] = "within_budget"
        entry["validation_started"] = True
        parent_artwork_fp = artwork_fingerprint(
            original_root,
            txn,
            excluded_from_parent,
        )
        assessment = assess_candidate(
            probe_temp,
            source_rgba_full,
            aligned_alpha,
            mode,
            parent_counts,
            transaction_id=txn,
            parent_artwork_fingerprint=parent_artwork_fp,
        )
        for field in (
            "validation_stage",
            "status",
            "exact_error_code",
            "native_alpha_iou",
            "native_alpha_mae",
            "bounded_alpha_iou",
            "bounded_alpha_mae",
            "evaluator_alpha_iou",
            "evaluator_alpha_mae",
            "artwork_fingerprint_match",
            "actual_path_count",
            "actual_node_count",
        ):
            entry[field] = assessment[field]

        if assessment["status"] != "accepted":
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            continue

        entry["journal_gate_started"] = True
        journal_passed, journal_codes = run_geometry_journal(
            parent_journal_path,
            probe_temp,
            journal_source_rgb,
            journal_image_class,
            assessment["report"],
        )
        entry["journal_passed"] = bool(journal_passed)
        entry["journal_reason_codes"] = list(journal_codes)
        if not journal_passed:
            entry["status"] = "geometry_rejected"
            entry["validation_stage"] = "journal_geometry"
            entry["exact_error_code"] = (
                "source_alpha_candidate_painter_journal_geometry_rejected:"
                + ",".join(journal_codes)
            )
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            continue

        attempts.append(entry)
        geometry = dict(probe_geometry)
        geometry.update(
            {
                "evaluator_aligned_retry": True,
                "evaluator_aligned_grid_width": int(aligned_alpha.shape[1]),
                "evaluator_aligned_grid_height": int(aligned_alpha.shape[0]),
                "evaluator_aligned_legacy_label": base_label,
            }
        )
        report = dict(assessment["report"] or {})
        report["painter_retry_source_alignment"] = (
            "final_evaluator_comparison_grid"
        )
        return [
            probe_size,
            int(assessment["actual_path_count"] or 0),
            int(assessment["actual_node_count"] or 0),
            order,
            probe_temp,
            geometry,
            report,
            label,
            0.0,
        ]

    return None
