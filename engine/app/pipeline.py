"""Pipeline public facade with AI-2 selection/lifecycle policy.

The established pipeline implementation lives in :mod:`app.pipeline_core`.
This facade keeps that implementation isolated while applying the small AI-2
policies at public extension points:

* connected-component safety filters candidate selection without rewriting
  measured total/fidelity scores;
* an exact source-palette path candidate is available only for fully opaque,
  low-color geometric/single-color artwork already inside the existing palette
  cap; it never embeds raster data or increases a shared budget;
* Pareto-dominated candidates cannot win inside the component-safe pool;
* neutral-palette repair is finalized after candidate generation and before its
  first score, so scoring itself is byte-pure;
* render-equivalent explicit fill-cycle closure is a final journaled lifecycle
  transform, followed by a final score of those exact artifact bytes.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app import pipeline_core as _core
from app.neutral_palette import (
    detect_neutral_luminance_bands,
    geometric_preserves_neutral_palette,
    restore_layered_neutral_svg_palette,
)
from app.source_palette_vector import (
    SourcePaletteNotApplicable,
    vectorize_source_palette_paths,
)
from app.svg_lifecycle import close_implicit_fill_cycles

for _name in dir(_core):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, getattr(_core, _name))

_base_select_best = _core.select_best
_base_produce_candidate = _core.produce_candidate
_base_produce_and_score_job = _core._produce_and_score_job
_base_apply_editability_preference = _core._apply_editability_preference
_base_refine_best = _core.refine_best
_base_refit_one = _core._refit_one
_base_apply_boundary_refit = _core._apply_boundary_refit
_base_build_vector_candidates = _core.build_vector_candidates
_base_run_pipeline = _core.run_pipeline

_SOURCE_PALETTE_MODES = {"geometric_logo", "single_color"}
_SOURCE_PALETTE_CANDIDATE = "source_palette_exact"


def _is_selection_safe(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get(
            "selection_safe",
            not candidate.get("selection_disqualified", False),
        )
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fidelity_axes(candidate: dict[str, Any]) -> dict[str, tuple[float, bool]] | None:
    """Return independent measured fidelity axes as ``value, higher_is_better``."""
    details = candidate.get("score_details") or {}
    component = candidate.get("component_quality") or {}
    axes: dict[str, tuple[float, bool]] = {}
    for name, value, higher in (
        ("fidelity", candidate.get("fidelity_score"), True),
        ("ssim", details.get("ssim"), True),
        ("mean_delta_e", details.get("mean_delta_e"), False),
        ("edge_f1", details.get("edge_f1"), True),
    ):
        measured = _finite(value)
        if measured is None:
            return None
        axes[name] = (measured, higher)

    if component.get("applicable"):
        for name in (
            "source_cc_recall",
            "render_cc_precision",
            "min_true_cc_iou",
        ):
            measured = _finite(component.get(name))
            if measured is None:
                return None
            axes[name] = (measured, True)
    return axes


def _dominates(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    """True iff candidate is no worse on every independent fidelity axis."""
    left = _fidelity_axes(candidate)
    right = _fidelity_axes(other)
    if left is None or right is None or set(left) != set(right):
        return False
    strictly_better = False
    for name in left:
        left_value, higher = left[name]
        right_value, _ = right[name]
        if higher:
            if left_value < right_value:
                return False
            strictly_better |= left_value > right_value
        else:
            if left_value > right_value:
                return False
            strictly_better |= left_value < right_value
    return bool(strictly_better)


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove candidates strictly dominated on independent fidelity axes."""
    if len(candidates) < 2:
        return candidates
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            other is not candidate and _dominates(other, candidate)
            for other in candidates
        ):
            continue
        front.append(candidate)
    return front or candidates


def _selection_pool(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer CC-safe Pareto candidates; preserve legacy ranking if none pass."""
    safe = [candidate for candidate in scored if _is_selection_safe(candidate)]
    if not safe:
        return scored
    return _pareto_front(safe)


def select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
    """Preserve a safe legacy winner; reroute only an actually unsafe winner."""
    legacy_chosen, legacy_raw, legacy_reason = _base_select_best(scored, mode)
    if _is_selection_safe(legacy_chosen):
        return legacy_chosen, legacy_raw, legacy_reason
    safe = [candidate for candidate in scored if _is_selection_safe(candidate)]
    if not safe:
        return legacy_chosen, legacy_raw, legacy_reason
    pool = _pareto_front(safe)
    chosen, _safe_raw, reason = _base_select_best(pool, mode)
    return chosen, legacy_raw, f"component_integrity_pareto_guard+{reason}"


def build_vector_candidates(mode: str) -> dict[str, dict[str, Any]]:
    """Add a narrow exact-palette candidate without changing existing families."""
    candidates = dict(_base_build_vector_candidates(mode))
    if mode in _SOURCE_PALETTE_MODES:
        candidates.setdefault(
            _SOURCE_PALETTE_CANDIDATE,
            {
                "engine": _SOURCE_PALETTE_CANDIDATE,
                "source_palette_exact": True,
            },
        )
    return candidates


def _source_palette_candidate(
    name: str,
    spec: dict[str, Any],
    mode: str,
    job_dir: Path,
    original_path: Path | None,
    palette_cap: int | None,
) -> dict[str, Any]:
    if original_path is None or mode not in _SOURCE_PALETTE_MODES:
        return {
            "name": name,
            "success": False,
            "error": "source_palette_not_applicable:mode_or_source",
            "engine": spec.get("engine"),
        }
    cap = palette_cap if palette_cap is not None else _core.PALETTE_CAP.get(mode)
    if not cap:
        return {
            "name": name,
            "success": False,
            "error": "source_palette_not_applicable:no_existing_palette_cap",
            "engine": spec.get("engine"),
        }
    svg_path = Path(job_dir) / f"{name}.svg"
    try:
        report = vectorize_source_palette_paths(
            Path(original_path),
            svg_path,
            max_colors=int(cap),
        )
    except SourcePaletteNotApplicable as exc:
        return {
            "name": name,
            "success": False,
            "error": f"source_palette_not_applicable:{exc}",
            "engine": spec.get("engine"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "success": False,
            "error": f"source_palette_candidate_failed:{type(exc).__name__}:{exc}",
            "engine": spec.get("engine"),
        }
    return {
        "name": name,
        "success": True,
        "engine": spec.get("engine"),
        "svg_path": str(svg_path),
        "source_palette_vector": report,
        "cleanup_report": {},
        "regularize_report": {},
        "palette_report": {
            "status": "preserved",
            "reason": "exact_source_palette_within_existing_cap",
            "palette_cap": int(cap),
        },
    }


def produce_candidate(
    name: str,
    spec: dict[str, Any],
    preprocessed_path: Path,
    mode: str,
    job_dir: Path,
    original_path: Path | None = None,
    palette_cap: int | None = None,
) -> dict[str, Any]:
    """Finalize narrow candidate construction before any candidate scoring."""
    if spec.get("engine") == _SOURCE_PALETTE_CANDIDATE:
        return _source_palette_candidate(
            name,
            spec,
            mode,
            job_dir,
            original_path,
            palette_cap,
        )

    result = _base_produce_candidate(
        name,
        spec,
        preprocessed_path,
        mode,
        job_dir,
        original_path=original_path,
        palette_cap=palette_cap,
    )
    result.setdefault(
        "neutral_palette_restore",
        {"applied": False, "reason": "not_layered_neutral_geometric"},
    )
    if not result.get("success") or original_path is None or mode != "geometric_logo":
        return result

    neutral_report = detect_neutral_luminance_bands(original_path)
    if not geometric_preserves_neutral_palette(mode, neutral_report):
        return result

    result["neutral_palette_restore"] = restore_layered_neutral_svg_palette(
        Path(result["svg_path"]), Path(original_path), mode=mode
    )
    return result


def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Picklable facade entrypoint so worker processes install the same policy."""
    return _base_produce_and_score_job(args)


def _apply_editability_preference(
    scored: list[dict[str, Any]], current_best: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    legacy_best, legacy_reason = _base_apply_editability_preference(scored, current_best)
    if _is_selection_safe(legacy_best):
        return legacy_best, legacy_reason
    safe = [candidate for candidate in scored if _is_selection_safe(candidate)]
    if not safe:
        return legacy_best, legacy_reason
    pool = _pareto_front(safe)
    safe_seed = current_best if current_best in pool else max(pool, key=_core._fidelity_rank_key)
    guarded_best, guarded_reason = _base_apply_editability_preference(pool, safe_seed)
    return guarded_best, f"component_integrity_pareto_guard+{guarded_reason}"


def refine_best(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    preprocessed_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    refined, info = _base_refine_best(
        best, mode, analysis, original_path, preprocessed_path, job_dir, scored
    )
    if _is_selection_safe(best) and not _is_selection_safe(refined):
        return best, {
            **info,
            "applied": False,
            "component_integrity_rejected": refined.get("name"),
        }
    if _is_selection_safe(best) and _dominates(best, refined):
        return best, {
            **info,
            "applied": False,
            "pareto_integrity_rejected": refined.get("name"),
        }
    return refined, info


def _refit_one(
    cand: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    job_dir: Path,
) -> dict[str, Any] | None:
    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)
    if (
        refined is not None
        and _is_selection_safe(cand)
        and not _is_selection_safe(refined)
    ):
        return None
    if refined is not None and _is_selection_safe(cand) and _dominates(cand, refined):
        return None
    return refined


def _apply_boundary_refit(
    best: dict[str, Any],
    mode: str,
    analysis: dict[str, Any],
    original_path: Path,
    job_dir: Path,
    scored: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate, info = _base_apply_boundary_refit(
        best, mode, analysis, original_path, job_dir, scored
    )
    if candidate is not best and _is_selection_safe(best):
        from app.component_quality import (
            gate_candidate_scores,
            score_svg_component_integrity,
            score_svg_discrete_rgb_palette_integrity,
        )
        strict_component = score_svg_component_integrity(
            Path(candidate["svg_path"]), Path(original_path),
            mode=mode, analysis=analysis, max_side=512,
            min_component_pixels=2, min_component_fraction=0.0,
        )
        rgb_component = score_svg_discrete_rgb_palette_integrity(
            Path(candidate["svg_path"]), Path(original_path),
            mode=mode, analysis=analysis, max_side=512,
            min_component_pixels=2,
        )
        post_refit_reports = [
            report for report in (strict_component, rgb_component)
            if report.get("applicable")
        ]
        for report in post_refit_reports:
            strict_gate = gate_candidate_scores(
                float(candidate.get("total_score", 0.0)),
                candidate.get("fidelity_score"), report,
            )
            if strict_gate.get("selection_safe") is not True:
                candidate["selection_safe"] = False
                candidate["selection_disqualified"] = True
                candidate["component_quality_status"] = report.get("status")
                scored[:] = [item for item in scored if item is not candidate]
                return best, {
                    **info, "applied": False,
                    "reason": "post_refit_micro_component_regression",
                    "post_refit_component_quality": strict_component,
                    "post_refit_rgb_palette_quality": rgb_component,
                }
        if strict_component.get("applicable"):
            candidate["component_quality"] = strict_component
        candidate["post_refit_rgb_palette_quality"] = rgb_component
    if _is_selection_safe(best) and not _is_selection_safe(candidate):
        return best, {
            **info,
            "applied": False,
            "reason": "component_integrity_regression",
        }
    if _is_selection_safe(best) and _dominates(best, candidate):
        return best, {
            **info,
            "applied": False,
            "reason": "pareto_integrity_regression",
        }
    return candidate, info


class _FillCycleJournal:
    """Factory namespace for the exact-render structural repair journal."""

    @staticmethod
    def build(
        baseline_path: Path,
        source_rgb: np.ndarray,
        *,
        image_class: str,
        required_metrics: set[str],
    ) -> Any:
        from app import transform_journal as _tj  # noqa: PLC0415

        class _ExactRenderRepairJournal(_tj.TransformJournal):
            """Accept only exact-render-preserving structural normalization."""

            def _measure(self, data: bytes) -> dict[str, Any]:
                sha = hashlib.sha256(data).hexdigest()
                measure_alpha = "alpha_fidelity" in self.required_metrics
                cache_key = f"{sha}:fill_cycle_exact:alpha={int(measure_alpha)}"
                if cache_key not in self._cache:
                    started = time.perf_counter()
                    try:
                        self._cache[cache_key] = _tj._measure_svg_bytes(
                            data,
                            self.source_rgb,
                            max_side=self.max_side,
                            required_metrics=self.required_metrics,
                            measure_alpha=measure_alpha,
                            capture_render=True,
                        )
                    finally:
                        self.evaluation_seconds += time.perf_counter() - started
                return self._cache[cache_key]

            def _decide(
                self,
                before: dict[str, Any],
                after: dict[str, Any],
                *,
                stage_id: str,
            ) -> list[str]:
                if stage_id == "explicit_fill_cycle_normalization":
                    required_ok = not after.get("required_unmeasured")
                    structural_ok = bool(after.get("structural_safe"))
                    gradient_ok = int(after.get("gradient_definition_count") or 0) >= int(
                        before.get("gradient_definition_count") or 0
                    )
                    rgb_before = before.get("render_rgb_sha256")
                    rgb_after = after.get("render_rgb_sha256")
                    rgb_exact = bool(rgb_before) and rgb_before == rgb_after
                    alpha_exact = True
                    if "alpha_fidelity" in self.required_metrics:
                        alpha_before = before.get("alpha_sha256")
                        alpha_after = after.get("alpha_sha256")
                        alpha_exact = bool(alpha_before) and alpha_before == alpha_after
                    if required_ok and structural_ok and gradient_ok and rgb_exact and alpha_exact:
                        return []
                return super()._decide(before, after, stage_id=stage_id)

        return _ExactRenderRepairJournal(
            baseline_path,
            source_rgb,
            image_class=image_class,
            required_metrics=required_metrics,
        )


def _finalize_fill_cycle_lifecycle(
    result: dict[str, Any],
    *,
    image: Image.Image,
    original_path: Path,
) -> dict[str, Any]:
    """Journal explicit fill closure on the final artifact, then rescore it."""
    best = result.get("best")
    if not best or not best.get("svg_path"):
        return result

    analysis = result.get("analysis") or {}
    mode = str(result.get("mode_used") or "")
    svg_path = Path(best["svg_path"])

    from app.transform_journal import merge_journal_reports  # noqa: PLC0415

    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        source_rgb = np.clip(
            rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        required_metrics = {"alpha_fidelity"}
    else:
        source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        required_metrics: set[str] = set()
    if analysis.get("has_gradient"):
        required_metrics.add("gradient_fidelity")

    journal = _FillCycleJournal.build(
        svg_path,
        source_rgb,
        image_class=_core._JOURNAL_IMAGE_CLASS.get(mode, "clean_logo"),
        required_metrics=required_metrics,
    )
    accepted, lifecycle_report, stage = journal.run_in_place(
        "explicit_fill_cycle_normalization",
        svg_path,
        close_implicit_fill_cycles,
    )
    result["transform_journal"] = merge_journal_reports(
        result.get("transform_journal"), journal.to_dict()
    )
    result["refit_info"] = {
        **(result.get("refit_info") or {}),
        "fill_cycle_normalization": {
            **(lifecycle_report or {}),
            "journal_status": stage["status"],
            "journal_reasons": stage["reason_codes"],
        },
    }

    if not accepted or not (lifecycle_report or {}).get("applied"):
        return result

    rescored = _core.score_candidate(best, Path(original_path), analysis, mode)
    if rescored is not None and rescored.get("rendered_ok"):
        result["best"] = rescored
        result["refit_info"]["final_rescore_after_fill_cycle"] = {
            "status": "measured",
            "fidelity_score": rescored.get("fidelity_score"),
        }
        if (
            mode != "photo_poster"
            and (analysis.get("background") or {}).get("is_uniform_background")
        ):
            result["structure_report"] = _core.score_structure_integrity(
                rescored["svg_path"], Path(original_path)
            )
    else:
        result["best"] = {
            **best,
            "rendered_ok": False,
            "fidelity_score": None,
            "final_rescore_error": "fill_cycle_measurement_unavailable",
        }
        result["refit_info"]["final_rescore_after_fill_cycle"] = {
            "status": "unmeasured",
            "error": "fill_cycle_measurement_unavailable",
        }
    return result


def run_pipeline(
    image: Image.Image,
    original_path: Path,
    trace_mode: str,
    job_dir: Path,
    refine: bool = True,
    edge_cleanup: bool = True,
) -> dict[str, Any]:
    result = _base_run_pipeline(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=refine,
        edge_cleanup=edge_cleanup,
    )
    return _finalize_fill_cycle_lifecycle(
        result,
        image=image,
        original_path=Path(original_path),
    )


_core.select_best = select_best
_core.build_vector_candidates = build_vector_candidates
_core.produce_candidate = produce_candidate
_core._produce_and_score_job = _produce_and_score_job
_core._apply_editability_preference = _apply_editability_preference
_core.refine_best = refine_best
_core._refit_one = _refit_one
_core._apply_boundary_refit = _apply_boundary_refit

WorkerFailure = _core.WorkerFailure


def __getattr__(name: str) -> Any:
    return getattr(_core, name)
