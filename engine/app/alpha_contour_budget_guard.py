"""Fail-closed preflight for pathological fragmented source-alpha contours.

The legacy contour fallback intentionally re-expresses highly fragmented alpha
fields as exact one-cell SVG subpaths. That representation is safe but can be
provably impossible under the unchanged TransformJournal node budget. In that
case materializing millions of path commands only burns CPU/RAM before the same
budget rejection.

This module keeps admissible contour behavior unchanged. It adds a cheap, exact
command-count lower bound only for the legacy fragmented one-cell branch. The
preflight is bound to both production routes that can reach ``_build_contour_plan``:
``alpha_mask_budget._preflight`` and the later adaptive rect-fidelity fallback.
When that branch cannot fit the existing node budget, the dedicated rejection is
made eligible for the already-existing candidate-geometry knockout fallback.
No quality threshold or path/node/byte budget is changed.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_FRAGMENTATION_BUDGET_PREFIX = (
    "source_alpha_mask_contour_fragmentation_budget_rejected:"
)
_INSTALLED = False
_BUDGET_PREFLIGHT_AVAILABLE_NODES: ContextVar[int | None] = ContextVar(
    "vektoryum_fragmented_contour_available_nodes",
    default=None,
)


def _fragmented_one_cell_stats(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    *,
    max_compact_contours: int,
) -> dict[str, int | bool]:
    """Return exact one-cell command cost if legacy enters that branch.

    ``_build_contour_plan`` switches to exact one-cell subpaths when every
    contour is degenerate, or when non-degenerate contour count exceeds
    ``_MAX_COMPACT_CONTOURS``. The switch can be determined without building SVG
    strings. Each one-cell path is ``M h v h Z``: exactly five commands under
    the unchanged journal command counter.
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
            # _canonical_contour returns None exactly for fewer than 3 points.
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
    """Reject only when legacy exact-cell geometry cannot fit node budget."""
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
    """Install all production node-preflight routes and the existing fallback hook."""
    global _INSTALLED
    if _INSTALLED:
        return

    from defusedxml import ElementTree as SafeET
    from app import alpha_candidate_knockout as knockout_module
    from app import alpha_mask_adaptive as adaptive_module
    from app import alpha_mask_budget as budget_module
    from app.alpha_preprocess import _rgba_from_source_at_size

    # The normal production alpha-mask transaction reaches
    # alpha_mask_budget._preflight first. That function decides whether verbose
    # rect geometry fits and calls the module-global _build_contour_plan only when
    # it does not. Bind the already-computed parent node capacity through a
    # ContextVar so the contour builder can reject the pathological one-cell branch
    # exactly at that call site, before any SVG path strings are allocated.
    original_budget_preflight = budget_module._preflight
    if not getattr(
        original_budget_preflight,
        "__vektoryum_fragmentation_budget_context__",
        False,
    ):
        @wraps(original_budget_preflight)
        def guarded_budget_preflight(
            svg_path: Path,
            source_path: Path,
        ) -> dict[str, Any] | None:
            target = Path(svg_path)
            before_size = target.stat().st_size
            root = SafeET.fromstring(target.read_bytes())
            limits = budget_module._journal_limits(root, before_size)
            available_nodes = max(
                0,
                int(limits["node_limit"]) - int(limits["parent_node_count"]),
            )
            token = _BUDGET_PREFLIGHT_AVAILABLE_NODES.set(available_nodes)
            try:
                return original_budget_preflight(target, Path(source_path))
            finally:
                _BUDGET_PREFLIGHT_AVAILABLE_NODES.reset(token)

        guarded_budget_preflight.__vektoryum_fragmentation_budget_context__ = True
        budget_module._preflight = guarded_budget_preflight

    original_build_contour_plan = budget_module._build_contour_plan
    if not getattr(
        original_build_contour_plan,
        "__vektoryum_fragmentation_budget_guard__",
        False,
    ):
        @wraps(original_build_contour_plan)
        def guarded_build_contour_plan(
            quantized: np.ndarray,
            opacity_by_level: dict[int, float],
        ) -> dict[str, Any] | None:
            available_nodes = _BUDGET_PREFLIGHT_AVAILABLE_NODES.get()
            if available_nodes is not None:
                _preflight_fragmented_contour_nodes(
                    quantized,
                    opacity_by_level,
                    available_nodes=int(available_nodes),
                    max_compact_contours=budget_module._MAX_COMPACT_CONTOURS,
                )
            return original_build_contour_plan(quantized, opacity_by_level)

        guarded_build_contour_plan.__vektoryum_fragmentation_budget_guard__ = True
        budget_module._build_contour_plan = guarded_build_contour_plan

    # A rect mask can pass the budget preflight and fail only the exact alpha render
    # gate later. That established adaptive retry computes a fresh parent budget, so
    # retain a dedicated wrapper there as well. Its original function imports the
    # module-global (now guarded) builder only after this wrapper has independently
    # proven the node cost admissible, avoiding duplicate rejection semantics.
    original_contour_fallback_plan = adaptive_module._contour_fallback_plan
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
            # Admissible/non-fragmented cases retain the original construction,
            # serialization, validation and telemetry byte-for-byte.
            return original_contour_fallback_plan(target, Path(source_path))

        guarded_contour_fallback_plan.__vektoryum_fragmentation_budget_guard__ = True
        adaptive_module._contour_fallback_plan = guarded_contour_fallback_plan

    # Candidate-knockout is already the next fail-closed production stage after
    # exact source-alpha reconstruction fails. Mark only this structural
    # impossibility as retry-eligible; no broad exception class is opened.
    prefixes = tuple(knockout_module._ALPHA_FAILURE_PREFIXES)
    if _FRAGMENTATION_BUDGET_PREFIX not in prefixes:
        knockout_module._ALPHA_FAILURE_PREFIXES = (
            *prefixes,
            _FRAGMENTATION_BUDGET_PREFIX,
        )

    _INSTALLED = True
