"""Bound RFV-3D3 alpha retries and reuse already-verified painter evidence.

This adapter keeps three narrow production contracts together:

* every accepted source-alpha representation may seed the existing compact
  budget-only retry;
* a painter candidate that already passed the real TransformJournal may hand its
  exact request-scoped measurements to the immediately following duplicate final
  journal, but only through an identity-bound, one-shot proof;
* the direct-element processing cap counts children that are actually rendered,
  rather than rejecting a large binary SVG whose contrasting children require no
  per-child probe.

No byte, node, path, alpha, fidelity, journal or timeout threshold is increased.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from app import alpha_budget_retry as _retry
from app import alpha_svg_mask

_INSTALLED = False
_DIRECT_PROBE_COUNT: ContextVar[int | None] = ContextVar(
    "vektoryum_direct_alpha_probe_count", default=None
)


def _wrap_register_accepted_alpha(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Register any applied source-alpha result for a later budget-only retry."""
    if getattr(original, "__vektoryum_alpha_budget_retry_all_accepted__", False):
        return original

    @wraps(original)
    def wrapped(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        report = original(Path(svg_path), Path(source_path), mode)
        if (
            isinstance(report, dict)
            and report.get("status") == "accepted"
            and report.get("applied") is True
        ):
            _retry._register(
                Path(svg_path),
                _retry._PendingKnockout(Path(source_path), str(mode), report),
            )
        return report

    wrapped.__vektoryum_alpha_budget_retry_all_accepted__ = True
    return wrapped


def _direct_renderable_indexes(root: ET.Element) -> list[int]:
    """Return all direct paint children; the real render work is budgeted later.

    Binary alpha with a proven comparison canvas can assign opacity to contrasting
    artwork without rendering each child. Rejecting merely because the SVG has more
    than 512 direct elements therefore sent valid large artwork into a pathological
    contour fallback. The unchanged 512 cap is now enforced by
    ``_wrap_direct_alpha_mode`` only when a child is actually measured.
    """
    from app import alpha_candidate_direct as direct

    indexes = [
        index
        for index, child in enumerate(list(root))
        if direct._local_name(str(child.tag)).lower() in direct._RENDERABLE_TAGS
        and direct._local_name(str(child.tag)).lower()
        not in direct._PROTECTED_ROOT_TAGS
    ]
    if not indexes:
        raise RuntimeError("source_alpha_direct_no_paint_children")
    return indexes


def _wrap_direct_alpha_mode(original: Callable[..., int]) -> Callable[..., int]:
    """Charge the existing direct-child cap only for a real rendered probe."""
    if getattr(original, "__vektoryum_direct_probe_budget__", False):
        return original

    @wraps(original)
    def wrapped(*args, **kwargs) -> int:
        from app import alpha_candidate_direct as direct

        count = _DIRECT_PROBE_COUNT.get()
        if count is not None:
            count += 1
            _DIRECT_PROBE_COUNT.set(count)
            if count > int(direct._MAX_DIRECT_CHILDREN):
                raise RuntimeError(
                    "source_alpha_direct_probe_budget_exceeded:"
                    f"{count}>{int(direct._MAX_DIRECT_CHILDREN)}"
                )
        return int(original(*args, **kwargs))

    wrapped.__vektoryum_direct_probe_budget__ = True
    return wrapped


def _wrap_direct_candidate_build(original: Callable[..., Any]) -> Callable[..., Any]:
    """Give each direct candidate its own deterministic render-probe budget."""
    if getattr(original, "__vektoryum_direct_probe_scope__", False):
        return original

    @wraps(original)
    def wrapped(*args, **kwargs):
        token = _DIRECT_PROBE_COUNT.set(0)
        try:
            return original(*args, **kwargs)
        finally:
            _DIRECT_PROBE_COUNT.reset(token)

    wrapped.__vektoryum_direct_probe_scope__ = True
    return wrapped


def _handoff_id(
    *,
    parent_sha: str,
    candidate_sha: str,
    source_sha: str,
    image_class: str,
    metrics: tuple[str, ...],
    max_side: int,
    transform_sha: str,
) -> str:
    payload = {
        "schema": "vektoryum-painter-journal-handoff-v1",
        "parent_sha256": parent_sha,
        "candidate_sha256": candidate_sha,
        "source_rgb_sha256": source_sha,
        "image_class": image_class,
        "required_metrics": list(metrics),
        "max_side": int(max_side),
        "transform_proof_sha256": transform_sha,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _register_painter_handoff(
    parent: Path,
    candidate: Path,
    source,
    image_class: str,
    metrics: set[str],
    max_side: int,
    cache: dict[str, Any],
    report: Any,
) -> None:
    """Register one exact painter measurement and stamp its lineage on the report."""
    from app import painter_measurement_reuse as reuse

    transform_sha = reuse._transform_sha(report)
    if not isinstance(cache, dict) or transform_sha is None or not isinstance(report, dict):
        return
    try:
        parent_sha = reuse._sha_path(Path(parent))
        candidate_sha = reuse._sha_path(Path(candidate))
        source_sha = reuse._sha_source(source)
    except Exception:
        return
    metric_tuple = tuple(sorted(str(value) for value in set(metrics)))
    namespace = reuse._namespace(source, int(max_side), set(metric_tuple))
    seconds = 0.0
    for sha in (parent_sha, candidate_sha):
        key = reuse._metric_key(namespace, sha)
        metric = cache.get(key)
        elapsed = cache.get(f"{key}\x00seconds")
        if not isinstance(metric, dict) or metric.get("sha256") != sha:
            return
        if not isinstance(elapsed, (int, float)) or float(elapsed) < 0:
            return
        seconds += float(elapsed)

    handoff = _handoff_id(
        parent_sha=parent_sha,
        candidate_sha=candidate_sha,
        source_sha=source_sha,
        image_class=str(image_class),
        metrics=metric_tuple,
        max_side=int(max_side),
        transform_sha=transform_sha,
    )
    # This scalar survives the painter's final report assembly. It contains no raw
    # source or SVG data and cannot authorize another candidate because every byte
    # and source identity is independently checked again by the consumer.
    report["journal_measurement_handoff_id"] = handoff
    proof = reuse._Proof(
        parent_sha,
        candidate_sha,
        source_sha,
        str(image_class),
        metric_tuple,
        int(max_side),
        seconds,
        handoff,
        cache,
    )
    registry = dict(reuse._REGISTRY.get())
    registry[candidate_sha] = (registry.get(candidate_sha, []) + [proof])[-8:]
    reuse._REGISTRY.set(dict(list(registry.items())[-64:]))


def _take_painter_handoff(journal: Any, parent: Path, candidate: Path, report: Any):
    """Consume an exact handoff once after re-verifying the full alpha contract."""
    from app import painter_measurement_reuse as reuse

    registry = dict(reuse._REGISTRY.get())
    reuse._REGISTRY.set({})
    if not isinstance(report, dict):
        return None
    handoff = report.get("journal_measurement_handoff_id")
    if not isinstance(handoff, str) or len(handoff) != 64:
        return None
    verifier = getattr(journal, "_verified_source_alpha_contract", None)
    if not callable(verifier) or not verifier("source_alpha_vector_mask", report):
        return None
    try:
        parent_sha = reuse._sha_path(Path(parent))
        candidate_sha = reuse._sha_path(Path(candidate))
        source_sha = reuse._sha_source(journal.source_rgb)
    except Exception:
        return None
    metrics = tuple(sorted(str(value) for value in set(journal.required_metrics)))
    namespace = reuse._namespace(journal.source_rgb, journal.max_side, set(metrics))
    for proof in registry.get(candidate_sha, []):
        if not (
            proof.parent_sha == parent_sha
            and proof.candidate_sha == candidate_sha
            and proof.source_sha == source_sha
            and proof.image_class == str(journal.image_class)
            and proof.required_metrics == metrics
            and proof.max_side == int(journal.max_side)
            and proof.transform_sha == handoff
        ):
            continue
        if all(
            isinstance(proof.cache.get(reuse._metric_key(namespace, sha)), dict)
            and proof.cache[reuse._metric_key(namespace, sha)].get("sha256") == sha
            for sha in (parent_sha, candidate_sha)
        ):
            return proof
    return None


def _install_direct_probe_budget() -> None:
    from app import alpha_candidate_direct as direct

    direct._direct_renderable_indexes = _direct_renderable_indexes
    direct._weighted_source_alpha_mode = _wrap_direct_alpha_mode(
        direct._weighted_source_alpha_mode
    )
    direct._build_direct_candidate = _wrap_direct_candidate_build(
        direct._build_direct_candidate
    )


def _install_painter_handoff() -> None:
    # painter_measurement_reuse's already-tested wrappers resolve these module
    # globals at call time, so replacing only the registry producer/consumer keeps
    # all cache accounting and one-shot behavior intact.
    from app import painter_measurement_reuse as reuse

    reuse._register = _register_painter_handoff
    reuse._take = _take_painter_handoff


def install_alpha_budget_retry_scope() -> None:
    """Install all idempotent RFV-3D3 scope and evidence adapters."""
    global _INSTALLED
    if _INSTALLED:
        return
    alpha_svg_mask.apply_source_alpha_mask = _wrap_register_accepted_alpha(
        alpha_svg_mask.apply_source_alpha_mask
    )
    _install_direct_probe_budget()
    _install_painter_handoff()
    _INSTALLED = True
