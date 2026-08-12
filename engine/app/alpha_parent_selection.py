"""Choose a leaner measured parent before final source-alpha masking."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_SUPPORTED_MODES = {
    "geometric_logo",
    "minimal_ai",
    "flat_logo",
    "logo_color",
    "photo_poster",
}
_MODE_IMAGE_CLASS = {
    "geometric_logo": "geometric",
    "minimal_ai": "clean_logo",
    "flat_logo": "clean_logo",
    "logo_color": "clean_logo",
    "photo_poster": "photo",
}
_MAX_ALPHA_LEVELS = 4
_MAX_TRIALS = 4
_FIDELITY_MARGIN = 0.15
_EDGE_MARGIN = 0.005


def _path(candidate: dict[str, Any]) -> Path | None:
    raw = candidate.get("svg_path")
    path = Path(raw) if raw else None
    return path if path is not None and path.is_file() else None


def _path_count(candidate: dict[str, Any]) -> int:
    return int((candidate.get("score_details") or {}).get("path_count") or 0)


def _edge_f1(candidate: dict[str, Any]) -> float | None:
    for source in (candidate, candidate.get("score_details") or {}):
        value = source.get("edge_f1")
        if value is not None:
            return float(value)
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _journal_source_rgb(source_path: Path) -> np.ndarray:
    with Image.open(Path(source_path)) as source:
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3].astype(np.float32)[:, :, None] / 255.0
    return np.clip(
        rgba[:, :, :3].astype(np.float32) * alpha
        + 255.0 * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def _shortlist(result: dict[str, Any]) -> list[dict[str, Any]]:
    best = result.get("best")
    if not isinstance(best, dict):
        return []
    pool = [best]
    pool.extend(
        item
        for item in result.get("scored") or []
        if isinstance(item, dict)
        and item.get("rendered_ok")
        and not (item.get("score_details") or {}).get("has_bitmap")
    )
    unique: dict[str, dict[str, Any]] = {}
    for item in pool:
        path = _path(item)
        if path is not None:
            unique.setdefault(str(path.resolve()), item)
    pool = list(unique.values())
    if len(pool) <= _MAX_TRIALS:
        return pool

    selected = [best]
    selected.append(
        min(pool, key=lambda item: (_path_count(item), _path(item).stat().st_size))
    )
    selected.append(
        max(
            pool,
            key=lambda item: (
                float(item.get("fidelity_score") or 0.0),
                -_path_count(item),
            ),
        )
    )
    refits = [item for item in pool if str(item.get("name") or "").endswith("_refit")]
    if refits:
        selected.append(
            max(refits, key=lambda item: float(item.get("fidelity_score") or 0.0))
        )

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in selected:
        path = _path(item)
        if path is None or str(path.resolve()) in seen:
            continue
        seen.add(str(path.resolve()))
        output.append(item)
        if len(output) == _MAX_TRIALS:
            break
    return output


def choose_alpha_parent_trial(
    trials: list[dict[str, Any]],
) -> dict[str, Any] | None:
    measured = [
        item
        for item in trials
        if item.get("rendered_ok")
        and item.get("fidelity_score") is not None
        and item.get("candidate") is not None
    ]
    if not measured:
        return None
    top_fidelity = max(float(item["fidelity_score"]) for item in measured)
    near_top = [
        item
        for item in measured
        if float(item["fidelity_score"]) >= top_fidelity - _FIDELITY_MARGIN
    ]
    edge_values = [
        float(item["edge_f1"])
        for item in near_top
        if item.get("edge_f1") is not None
    ]
    top_edge = max(edge_values) if edge_values else None
    eligible = [
        item
        for item in near_top
        if top_edge is None
        or item.get("edge_f1") is None
        or float(item["edge_f1"]) >= top_edge - _EDGE_MARGIN
    ]
    return min(
        eligible,
        key=lambda item: (
            int(item.get("path_count") or 0),
            int(item.get("byte_size") or 0),
            -float(item["fidelity_score"]),
        ),
    )


def _bind_parent_selection_journal(
    result: dict[str, Any],
    *,
    best_path: Path,
    selected_path: Path,
    source_path: Path,
    mode: str,
    trial: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Accept alternate-parent selection as an explicit journal boundary.

    Parent selection changes the artifact that the subsequent alpha journal uses as
    its baseline. Without this stage, the previous journal still ends at the old
    winner and the merged chain correctly reports both a stage-parent and journal-
    boundary mismatch. The alternate parent is therefore admitted only through the
    unchanged TransformJournal gates; rejection simply preserves the original best.
    """
    existing = result.get("transform_journal")
    best_sha = _sha256(best_path)
    if isinstance(existing, dict):
        existing_final = existing.get("final_accepted_sha256")
        if existing_final and existing_final != best_sha:
            return None, None, "existing_journal_best_mismatch"

    from app.transform_journal import TransformJournal, merge_journal_reports  # noqa: PLC0415

    journal = TransformJournal(
        best_path,
        _journal_source_rgb(source_path),
        image_class=_MODE_IMAGE_CLASS.get(mode, "clean_logo"),
        required_metrics=set(),
    )
    accepted_path, stage = journal.consider_candidate(
        "source_alpha_parent_selection",
        best_path,
        selected_path,
        transform_report={
            "schema": "vektoryum-alpha-parent-selection-v1",
            "selected_fidelity_score": float(trial["fidelity_score"]),
            "selected_edge_f1": trial.get("edge_f1"),
            "selected_path_count_after_alpha": int(trial["path_count"]),
            "selected_byte_size_after_alpha": int(trial["byte_size"]),
        },
    )
    if accepted_path != selected_path:
        reasons = ",".join(stage.get("reason_codes") or ["journal_rejected"])
        return None, stage, f"journal_rejected:{reasons}"

    merged = merge_journal_reports(
        existing if isinstance(existing, dict) else None,
        journal.to_dict(),
    )
    if not merged or not merged.get("chain_valid", False):
        codes = ",".join(
            (merged or {}).get("chain_failure_codes") or ["journal_merge_failed"]
        )
        return None, stage, f"journal_chain_invalid:{codes}"
    return merged, stage, None


def _select(result: dict[str, Any], source_path: Path, job_dir: Path) -> dict[str, Any]:
    from app.alpha_pipeline_retry import alpha_remediation_enabled  # noqa: PLC0415

    # The legacy-first pass must preserve the established winner and its bytes.
    # Alternate-parent trials are authorized only in the bounded remediation pass.
    if not alpha_remediation_enabled():
        return result

    best = result.get("best")
    mode = str(result.get("mode_used") or "")
    if not isinstance(best, dict) or mode not in _SUPPORTED_MODES:
        return result
    try:
        with Image.open(source_path) as source:
            levels = np.unique(np.asarray(source.convert("RGBA"))[:, :, 3])
    except (OSError, ValueError):
        return result
    # Preserve the proven bounded parent-selection contract. Dense/continuous
    # alpha stays on the dedicated alpha pipeline; broad parent probing changed
    # previously accepted knockout topology without cross-corpus proof.
    if len(levels) <= 1 or len(levels) > _MAX_ALPHA_LEVELS:
        return result

    shortlist = _shortlist(result)
    if len(shortlist) <= 1:
        return result

    from app import alpha_svg_mask  # noqa: PLC0415
    from app.pipeline import score_candidate  # noqa: PLC0415

    trials: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="alpha-parent-", dir=job_dir) as temporary:
        for index, candidate in enumerate(shortlist):
            parent = _path(candidate)
            if parent is None:
                continue
            trial_path = Path(temporary) / f"trial-{index}.svg"
            shutil.copy2(parent, trial_path)
            try:
                alpha_svg_mask.apply_source_alpha_mask(trial_path, source_path, mode)
                rescored = score_candidate(
                    {**candidate, "svg_path": trial_path},
                    source_path,
                    result.get("analysis") or {},
                    mode,
                )
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rescored, dict):
                continue
            trials.append(
                {
                    "candidate": candidate,
                    "rendered_ok": bool(rescored.get("rendered_ok")),
                    "fidelity_score": rescored.get("fidelity_score"),
                    "edge_f1": _edge_f1(rescored),
                    "path_count": _path_count(rescored),
                    "byte_size": trial_path.stat().st_size,
                }
            )

    chosen = choose_alpha_parent_trial(trials)
    if chosen is None:
        return result
    candidate = chosen["candidate"]
    selected_path = _path(candidate)
    best_path = _path(best)
    if selected_path is None or best_path is None or selected_path.resolve() == best_path.resolve():
        return result

    merged_journal, selection_stage, rejection = _bind_parent_selection_journal(
        result,
        best_path=best_path,
        selected_path=selected_path,
        source_path=source_path,
        mode=mode,
        trial=chosen,
    )
    if rejection is not None or merged_journal is None:
        result["alpha_parent_selection"] = {
            "status": "preserved_original",
            "reason": rejection or "journal_not_available",
            "source_alpha_level_count": int(len(levels)),
            "original_candidate_name": str(best.get("name") or best_path.stem),
            "proposed_candidate_name": str(candidate.get("name") or selected_path.stem),
            "trial_count": len(trials),
            "journal_stage": selection_stage,
        }
        return result

    result["best"] = candidate
    result["transform_journal"] = merged_journal
    result["selection_reason"] = (
        f"{result.get('selection_reason') or 'selected'}+alpha_parent_editability"
    )
    result["alpha_parent_selection"] = {
        "status": "selected",
        "source_alpha_level_count": int(len(levels)),
        "original_candidate_name": str(best.get("name") or best_path.stem),
        "selected_candidate_name": str(candidate.get("name") or selected_path.stem),
        "trial_count": len(trials),
        "selected_fidelity_score": float(chosen["fidelity_score"]),
        "selected_edge_f1": chosen.get("edge_f1"),
        "selected_path_count_after_alpha": int(chosen["path_count"]),
        "selected_byte_size_after_alpha": int(chosen["byte_size"]),
        "fidelity_margin": _FIDELITY_MARGIN,
        "edge_f1_margin": _EDGE_MARGIN,
        "journal_stage_status": str((selection_stage or {}).get("status") or "accepted"),
        "journal_chain_valid": True,
    }
    return result


def wrap_run_pipeline_selecting_alpha_parent(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_alpha_parent_selected__", False):
        return original

    @wraps(original)
    def wrapped(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ) -> dict[str, Any]:
        result = original(
            image,
            original_path,
            trace_mode,
            job_dir,
            refine=refine,
            edge_cleanup=edge_cleanup,
        )
        return _select(result, Path(original_path), Path(job_dir))

    wrapped.__vektoryum_alpha_parent_selected__ = True
    return wrapped
