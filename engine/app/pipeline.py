"""Pipeline public facade with AI-2 selection/lifecycle policy.

The established pipeline implementation lives in :mod:`app.pipeline_core`.
This facade keeps that implementation byte-identical while applying the small
AI-2 manager-remediation policies at its public extension points:

* connected-component safety filters candidate selection without rewriting
  measured total/fidelity scores;
* neutral-palette repair is finalized after candidate generation and before its
  first score, so scoring itself is byte-pure;
* render-equivalent explicit fill-cycle closure is a final journaled lifecycle
  transform, followed by a final score of those exact artifact bytes.
"""

from __future__ import annotations

import hashlib
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
_base_run_pipeline = _core.run_pipeline


def _is_selection_safe(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get(
            "selection_safe",
            not candidate.get("selection_disqualified", False),
        )
    )


def _selection_pool(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer component-safe candidates; preserve legacy ranking if none pass."""
    safe = [candidate for candidate in scored if _is_selection_safe(candidate)]
    return safe or scored


def select_best(scored: list[dict[str, Any]], mode: str) -> tuple[dict, dict, str]:
    """Run the established selector inside the independent CC-safe pool."""
    legacy_chosen, legacy_raw, legacy_reason = _base_select_best(scored, mode)
    pool = _selection_pool(scored)
    if len(pool) == len(scored):
        return legacy_chosen, legacy_raw, legacy_reason
    chosen, _safe_raw, reason = _base_select_best(pool, mode)
    return chosen, legacy_raw, f"component_integrity_guard+{reason}"


def produce_candidate(
    name: str,
    spec: dict[str, Any],
    preprocessed_path: Path,
    mode: str,
    job_dir: Path,
    original_path: Path | None = None,
    palette_cap: int | None = None,
) -> dict[str, Any]:
    """Finalize the narrow neutral-palette repair before candidate scoring."""
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
    pool = _selection_pool(scored)
    if current_best not in pool:
        current_best = max(pool, key=_core._fidelity_rank_key)
    return _base_apply_editability_preference(pool, current_best)


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
    if _is_selection_safe(best) and not _is_selection_safe(candidate):
        return best, {
            **info,
            "applied": False,
            "reason": "component_integrity_regression",
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
_core.produce_candidate = produce_candidate
_core._produce_and_score_job = _produce_and_score_job
_core._apply_editability_preference = _apply_editability_preference
_core.refine_best = refine_best
_core._refit_one = _refit_one
_core._apply_boundary_refit = _apply_boundary_refit

WorkerFailure = _core.WorkerFailure


def __getattr__(name: str) -> Any:
    return getattr(_core, name)
