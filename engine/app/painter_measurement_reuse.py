"""Reuse a painter candidate's request-scoped TransformJournal measurements.

The painter tournament already measures the selected parent/candidate pair through
unchanged production gates. The final source-alpha journal may reuse those exact
metrics only when bytes, source RGB, image class, metric scope, max-side and the
source-alpha proof all match. The original measurement is charged once; the
verified duplicate cache hit contributes zero evaluation seconds.
"""
from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

_INSTALLED = False
_PROOF_KEYS = (
    "schema", "status", "applied", "candidate_geometry_preserved",
    "candidate_path_data_preserved", "candidate_identity_preserved",
    "artwork_identity_preserved", "trace_rgb_bytes_preserved",
    "source_alpha_background_non_regression",
    "source_alpha_background_failure_codes", "source_alpha_background_comparison",
    "source_alpha_background_authority", "source_alpha_paint_support_verified",
    "source_alpha_paint_support_authority", "source_alpha_component_count",
    "source_alpha_unsupported_component_count",
    "source_alpha_non_canvas_renderable_count",
    "final_evaluator_alpha_plane_status",
    "final_evaluator_alpha_plane_hard_fail_codes", "source_truth_alpha_iou",
    "source_truth_alpha_mae", "final_evaluator_alpha_iou",
    "final_evaluator_alpha_mae",
)


@dataclass(frozen=True)
class _Proof:
    parent_sha: str
    candidate_sha: str
    source_sha: str
    image_class: str
    required_metrics: tuple[str, ...]
    max_side: int
    measured_seconds: float
    transform_sha: str
    cache: dict[str, Any]


_REGISTRY: ContextVar[dict[str, list[_Proof]]] = ContextVar(
    "vektoryum_painter_measurement_reuse", default={}
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _sha_source(source: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(source, dtype=np.uint8)).tobytes()
    ).hexdigest()


def _namespace(source: np.ndarray, max_side: int, metrics: set[str]) -> str:
    return f"{_sha_source(source)[:16]}:{int(max_side)}:{','.join(sorted(metrics))}|"


def _metric_key(namespace: str, sha: str) -> str:
    return f"{namespace}{sha}:alpha=0:render=0"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _transform_sha(report: Any) -> str | None:
    if not isinstance(report, dict):
        return None
    try:
        payload = json.dumps(
            {key: report.get(key) for key in _PROOF_KEYS},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
            default=_json_default,
        ).encode()
    except (TypeError, ValueError):
        return None
    return _sha_bytes(payload)


def _clear_scope() -> None:
    _REGISTRY.set({})


def _register(
    parent: Path,
    candidate: Path,
    source: np.ndarray,
    image_class: str,
    metrics: set[str],
    max_side: int,
    cache: dict[str, Any],
    report: Any,
) -> None:
    transform_sha = _transform_sha(report)
    if not isinstance(cache, dict) or transform_sha is None:
        return
    parent_sha, candidate_sha = _sha_path(parent), _sha_path(candidate)
    namespace = _namespace(source, max_side, metrics)
    seconds = 0.0
    for sha in (parent_sha, candidate_sha):
        key = _metric_key(namespace, sha)
        metric, elapsed = cache.get(key), cache.get(f"{key}\x00seconds")
        if not isinstance(metric, dict) or metric.get("sha256") != sha:
            return
        if not isinstance(elapsed, (int, float)) or float(elapsed) < 0:
            return
        seconds += float(elapsed)
    proof = _Proof(
        parent_sha, candidate_sha, _sha_source(source), str(image_class),
        tuple(sorted(metrics)), int(max_side), seconds, transform_sha, cache,
    )
    registry = dict(_REGISTRY.get())
    registry[candidate_sha] = (registry.get(candidate_sha, []) + [proof])[-8:]
    _REGISTRY.set(dict(list(registry.items())[-64:]))


def _take(journal: Any, parent: Path, candidate: Path, report: Any) -> _Proof | None:
    registry = dict(_REGISTRY.get())
    _REGISTRY.set({})
    try:
        parent_sha, candidate_sha = _sha_path(parent), _sha_path(candidate)
        source_sha, transform_sha = _sha_source(journal.source_rgb), _transform_sha(report)
    except Exception:
        return None
    metrics = tuple(sorted(str(value) for value in set(journal.required_metrics)))
    for proof in registry.get(candidate_sha, []):
        if not (
            proof.parent_sha == parent_sha
            and proof.source_sha == source_sha
            and proof.image_class == str(journal.image_class)
            and proof.required_metrics == metrics
            and proof.max_side == int(journal.max_side)
            and proof.transform_sha == transform_sha
        ):
            continue
        namespace = _namespace(journal.source_rgb, journal.max_side, set(metrics))
        if all(
            isinstance(proof.cache.get(_metric_key(namespace, sha)), dict)
            and proof.cache[_metric_key(namespace, sha)].get("sha256") == sha
            for sha in (parent_sha, candidate_sha)
        ):
            return proof
    return None


def _journal_key(journal: Any, data: bytes) -> str:
    capture = journal._measurement_stage_id == "restore_source_dimensions"
    return (
        f"{journal._cache_namespace}{_sha_bytes(data)}"
        f":alpha={int(capture)}:render={int(capture)}"
    )


def _wrap_measure(original: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_verified_reuse__", False):
        return original

    @wraps(original)
    def wrapped(journal: Any, data: bytes) -> dict[str, Any]:
        hit = bool(getattr(journal, "_vektoryum_reuse_bound", False)) and (
            _journal_key(journal, data) in journal._cache
        )
        before = float(journal.evaluation_seconds)
        result = original(journal, data)
        if hit:
            journal.evaluation_seconds = before
            journal._vektoryum_reuse_hits = int(
                getattr(journal, "_vektoryum_reuse_hits", 0)
            ) + 1
        return result

    wrapped.__vektoryum_verified_reuse__ = True
    return wrapped


def _wrap_painter_journal(original: Callable[..., tuple[bool, list[str]]]):
    if getattr(original, "__vektoryum_reuse_register__", False):
        return original

    @wraps(original)
    def wrapped(parent, candidate, source, image_class, report, measurement_cache=None):
        passed, reasons = original(
            parent, candidate, source, image_class, report,
            measurement_cache=measurement_cache,
        )
        if passed and isinstance(measurement_cache, dict):
            _register(
                Path(parent), Path(candidate), source, image_class, set(), 512,
                measurement_cache, report,
            )
        return bool(passed), list(reasons)

    wrapped.__vektoryum_reuse_register__ = True
    return wrapped


def _wrap_painter_apply(original: Callable[..., dict[str, Any]]):
    if getattr(original, "__vektoryum_reuse_scope__", False):
        return original

    @wraps(original)
    def wrapped(*args, **kwargs):
        _clear_scope()
        try:
            return original(*args, **kwargs)
        except BaseException:
            _clear_scope()
            raise

    wrapped.__vektoryum_reuse_scope__ = True
    return wrapped


def _wrap_consider(original: Callable[..., tuple[Path, dict[str, Any]]]):
    if getattr(original, "__vektoryum_reuse_consumer__", False):
        return original

    @wraps(original)
    def wrapped(journal, stage_id, parent, candidate, *, transform_report=None):
        proof = None
        if stage_id == "source_alpha_vector_mask":
            proof = _take(journal, Path(parent), Path(candidate), transform_report)
            if proof is not None:
                journal._cache = proof.cache
                journal._cache_namespace = _namespace(
                    journal.source_rgb, journal.max_side, set(journal.required_metrics)
                )
                journal._vektoryum_reuse_bound = True
        accepted, stage = original(
            journal, stage_id, parent, candidate, transform_report=transform_report
        )
        if proof is not None and isinstance(stage, dict):
            hits = int(getattr(journal, "_vektoryum_reuse_hits", 0))
            stage.update({
                "measurement_cache_reuse": "verified_request_scoped_exact_sha",
                "measurement_cache_hit_count": hits,
                "measurement_cache_original_seconds": round(proof.measured_seconds, 6),
            })
            if isinstance(transform_report, dict):
                transform_report.update({
                    "journal_measurement_cache_reuse": (
                        "verified_request_scoped_exact_sha"
                    ),
                    "journal_measurement_cache_hit_count": hits,
                    "journal_original_measurement_seconds": round(
                        proof.measured_seconds, 6
                    ),
                })
        return accepted, stage

    wrapped.__vektoryum_reuse_consumer__ = True
    return wrapped


def install_painter_measurement_reuse() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from app import alpha_candidate_painter
    from app.transform_journal import TransformJournal

    alpha_candidate_painter._run_painter_geometry_journal = _wrap_painter_journal(
        alpha_candidate_painter._run_painter_geometry_journal
    )
    alpha_candidate_painter.apply_candidate_painter_reconstruction = (
        _wrap_painter_apply(
            alpha_candidate_painter.apply_candidate_painter_reconstruction
        )
    )
    TransformJournal._measure = _wrap_measure(TransformJournal._measure)
    TransformJournal.consider_candidate = _wrap_consider(
        TransformJournal.consider_candidate
    )
    _INSTALLED = True
