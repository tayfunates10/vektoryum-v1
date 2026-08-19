"""Fail-closed preflight for pathological fragmented source-alpha contours.

The legacy contour fallback intentionally re-expresses highly fragmented alpha
fields as exact one-cell SVG subpaths. That representation is safe but can be
provably impossible under the unchanged TransformJournal node budget. In that
case materializing millions of path commands only burns CPU/RAM before the same
budget rejection.

This module keeps the existing admissible contour behavior byte-for-byte. It
adds a cheap, exact command-count lower bound only for the legacy fragmented
one-cell branch and lets the already-existing candidate-knockout/support
fallback chain continue when that contour retry is structurally impossible.
No quality threshold or path/node/byte budget is changed.
"""
from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

_FRAGMENTATION_BUDGET_PREFIX = (
    "source_alpha_mask_contour_fragmentation_budget_rejected:"
)
_INSTALLED = False


def _fragmented_one_cell_stats(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    *,
    max_compact_contours: int,
) -> dict[str, int | bool]:
    """Return the exact one-cell command cost if legacy would enter that branch.

    ``_build_contour_plan`` enters its expensive exact-cell branch when no
    non-degenerate compact contour exists but degenerate islands do, or when the
    number of non-degenerate contours exceeds ``_MAX_COMPACT_CONTOURS``. We can
    decide that condition without constructing any SVG path strings. A one-cell
    subpath is exactly ``M h v h Z`` = five counted path commands.
    """
    compact_contours = 0
    pruned_contours = 0
    has_compact_layer = False

    for level in sorted(opacity_by_level):
        mask = (np.asarray(quantized) == int(level)).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            # _canonical_contour returns None iff fewer than three points exist.
            if int(len(contour)) < 3:
                pruned_contours += 1
                continue
            compact_contours += 1
            has_compact_layer = True
            if compact_contours > int(max_compact_contours):
                break
        if compact_contours > int(max_compact_contours):
            break

    fragmented = bool(
        (not has_compact_layer and pruned_contours > 0)
        or compact_contours > int(max_compact_contours)
    )
    if not fragmented:
        return {
            "fragmented": False,
            "compact_contour_count": int(compact_contours),
            "pruned_contour_count": int(pruned_contours),
            "cell_count": 0,
            "command_count": 0,
        }

    cell_count = int(np.count_nonzero(np.asarray(quantized) > 0))
    return {
        "fragmented": True,
        "compact_contour_count": int(compact_contours),
        "pruned_contour_count": int(pruned_contours),
        "cell_count": int(cell_count),
        "command_count": int(cell_count * 5),
    }


def _preflight_fragmented_contour_nodes(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    *,
    available_nodes: int,
    max_compact_contours: int,
) -> dict[str, int | bool]:
    """Reject only when the legacy exact-cell branch cannot fit the node budget."""
    stats = _fragmented_one_cell_stats(
        quantized,
        opacity_by_level,
        max_compact_contours=max_compact_contours,
    )
    if bool(stats["fragmented"]) and int(stats["command_count"]) > int(
        available_nodes
    ):
        raise RuntimeError(
            _FRAGMENTATION_BUDGET_PREFIX
            + f"path_nodes={int(stats['command_count'])}/{int(available_nodes)},"
            f"alpha_cells={int(stats['cell_count'])},"
            f"compact_contours={int(stats['compact_contour_count'])},"
            f"pruned_contours={int(stats['pruned_contour_count'])}"
        )
    return stats


def install_contour_budget_guard() -> None:
    """Install the guard before runtime alpha wrappers are constructed."""
    global _INSTALLED
    if _INSTALLED:
        return

    from defusedxml import ElementTree as SafeET
    from app import alpha_mask_adaptive as adaptive_module
    from app import alpha_mask_budget as budget_module
    from app.alpha_preprocess import _rgba_from_source_at_size

    original_contour_fallback_plan = adaptive_module._contour_fallback_plan
    original_factory = adaptive_module.make_rect_fidelity_fallback

    if not getattr(
        original_contour_fallback_plan,
        "__vektoryum_fragmentation_budget_guard__",
        False,
    ):
        @wraps(original_contour_fallback_plan)
        def guarded_contour_fallback_plan(
            svg_path: Path,
            source_path: Path,
        ) -> tuple[dict[str, Any], dict[str, int]]:
            target = Path(svg_path)
            before_size = target.stat().st_size
            root = SafeET.fromstring(target.read_bytes())
            limits = budget_module._journal_limits(root, before_size)
            width, height = budget_module._viewbox_size(root)
            scale = min(
                1.0,
                budget_module._MAX_MASK_SIDE / float(max(width, height)),
            )
            raster_width = max(1, int(round(width * scale)))
            raster_height = max(1, int(round(height * scale)))
            rgba = _rgba_from_source_at_size(
                Path(source_path),
                (raster_width, raster_height),
            )
            quantized, opacity_by_level = budget_module._quantize_alpha(
                np.asarray(rgba[:, :, 3], dtype=np.uint8)
            )
            available_nodes = max(
                0,
                int(limits["node_limit"]) - int(limits["parent_node_count"]),
            )
            _preflight_fragmented_contour_nodes(
                quantized,
                opacity_by_level,
                available_nodes=available_nodes,
                max_compact_contours=budget_module._MAX_COMPACT_CONTOURS,
            )
            # Admissible/non-fragmented cases keep the legacy implementation and
            # therefore retain its exact path construction and telemetry.
            return original_contour_fallback_plan(target, Path(source_path))

        guarded_contour_fallback_plan.__vektoryum_fragmentation_budget_guard__ = True
        adaptive_module._contour_fallback_plan = guarded_contour_fallback_plan

    if not getattr(
        original_factory,
        "__vektoryum_fragmentation_passthrough_factory__",
        False,
    ):
        def guarded_factory(
            guarded_builder: Callable[[Path, Path, str], dict[str, Any]],
        ) -> Callable[[Path, Path, str], dict[str, Any]]:
            """Preserve the original alpha error when contour retry is impossible.

            Outer candidate-knockout/support fallbacks are keyed to the original
            exact alpha-gate prefix. A dead contour retry must not replace that
            signal with an internal complexity error and terminate the chain.
            """
            if getattr(
                guarded_builder,
                "__vektoryum_rect_fidelity_fallback__",
                False,
            ):
                return guarded_builder
            base_builder = inspect.unwrap(guarded_builder)

            @wraps(guarded_builder)
            def fallback(
                svg_path: Path,
                source_path: Path,
                mode: str,
            ) -> dict[str, Any]:
                target = Path(svg_path)
                source = Path(source_path)
                original_error: RuntimeError | None = None
                try:
                    return guarded_builder(target, source, mode)
                except RuntimeError as first_error:
                    trigger = str(first_error)
                    if not trigger.startswith((
                        "source_alpha_mask_iou_gate_failed:",
                        "source_alpha_mask_mae_gate_failed:",
                    )):
                        raise
                    original_error = first_error

                from app.alpha_mask_budget import (
                    _ALPHA_MASK_ENCODING,
                    _ALPHA_MASK_PLAN,
                    _create_atomic_backup,
                    _restore_atomic_backup,
                )

                try:
                    plan, measurements = adaptive_module._contour_fallback_plan(
                        target,
                        source,
                    )
                except RuntimeError as contour_error:
                    if str(contour_error).startswith(_FRAGMENTATION_BUDGET_PREFIX):
                        assert original_error is not None
                        raise original_error from contour_error
                    raise

                encoding_token = _ALPHA_MASK_ENCODING.set("path")
                plan_token = _ALPHA_MASK_PLAN.set(plan)
                backup = _create_atomic_backup(target)
                try:
                    report = base_builder(target, source, mode)
                except BaseException:
                    _restore_atomic_backup(backup, target)
                    raise
                else:
                    backup.unlink(missing_ok=True)
                finally:
                    _ALPHA_MASK_PLAN.reset(plan_token)
                    _ALPHA_MASK_ENCODING.reset(encoding_token)

                report.update(measurements)
                report["mask_encoding"] = "path"
                report["preflight_mask_encoding"] = "rect"
                report["mask_fallback_reason"] = "rect_exact_alpha_gate_failure"
                report["mask_fallback_trigger"] = trigger
                report["rollback_guard"] = "armed_and_committed"
                return report

            fallback.__vektoryum_rect_fidelity_fallback__ = True
            return fallback

        guarded_factory.__vektoryum_fragmentation_passthrough_factory__ = True
        adaptive_module.make_rect_fidelity_fallback = guarded_factory

    _INSTALLED = True
