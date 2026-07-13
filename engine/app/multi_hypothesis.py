"""FAZ 4 shadow multi-hypothesis selection.

This module does not change the production winner yet. It provides a deterministic,
fail-closed selector and an audit report that can run beside the existing pipeline.
Production promotion is intentionally a later, separately gated step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class HypothesisPolicy:
    fidelity_margin: float = 1.5
    edge_tolerance: float = 0.02
    max_path_ratio: float = 0.70
    min_fidelity: float = 78.0
    max_candidates: int = 12


@dataclass(frozen=True)
class HypothesisView:
    name: str
    family: str
    fidelity: float
    total_score: float
    edge_f1: float
    path_count: int
    rendered_ok: bool
    hard_fail_codes: tuple[str, ...] = ()
    unmeasured_required: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def eligible(self) -> bool:
        return (
            self.rendered_ok
            and not self.hard_fail_codes
            and not self.unmeasured_required
        )


def _family(candidate: dict[str, Any]) -> str:
    explicit = candidate.get("hypothesis_family")
    if explicit:
        return str(explicit)
    engine = str(candidate.get("engine") or "unknown")
    name = str(candidate.get("name") or "candidate")
    if engine == "gradient":
        return "gradient"
    if "center" in name or engine == "autotrace":
        return "centerline"
    if "contour" in name:
        return "contour"
    if engine == "vtracer":
        return "region"
    return engine


def _report_fields(candidate: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    report = candidate.get("final_artifact_report") or {}
    hard = tuple(sorted(str(x) for x in report.get("hard_fail_codes", []) or []))
    unmeasured = tuple(sorted(str(x) for x in report.get("unmeasured_required", []) or []))
    return hard, unmeasured


def normalize_hypothesis(candidate: dict[str, Any]) -> HypothesisView:
    details = candidate.get("score_details") or {}
    hard, unmeasured = _report_fields(candidate)
    fidelity = candidate.get("fidelity_score")
    return HypothesisView(
        name=str(candidate.get("name") or "candidate"),
        family=_family(candidate),
        fidelity=float(fidelity if fidelity is not None else 0.0),
        total_score=float(candidate.get("total_score") or 0.0),
        edge_f1=float(details.get("edge_f1") or 0.0),
        path_count=max(0, int(details.get("path_count") or 0)),
        rendered_ok=bool(candidate.get("rendered_ok")),
        hard_fail_codes=hard,
        unmeasured_required=unmeasured,
        source=candidate,
    )


def _rank_key(h: HypothesisView) -> tuple[float, float, int, float, str]:
    # Stable tie-breaking makes CI and audit reports reproducible.
    return (
        round(h.fidelity, 4),
        round(h.edge_f1, 5),
        -h.path_count,
        round(h.total_score, 4),
        h.name,
    )


def _dominates(a: HypothesisView, b: HypothesisView) -> bool:
    not_worse = (
        a.fidelity >= b.fidelity
        and a.edge_f1 >= b.edge_f1
        and a.path_count <= b.path_count
    )
    strictly_better = (
        a.fidelity > b.fidelity
        or a.edge_f1 > b.edge_f1
        or a.path_count < b.path_count
    )
    return not_worse and strictly_better


def pareto_frontier(items: Iterable[HypothesisView]) -> list[HypothesisView]:
    candidates = list(items)
    frontier = [
        item for item in candidates
        if not any(other is not item and _dominates(other, item) for other in candidates)
    ]
    return sorted(frontier, key=_rank_key, reverse=True)


def select_shadow_hypothesis(
    candidates: list[dict[str, Any]],
    policy: HypothesisPolicy | None = None,
) -> dict[str, Any]:
    """Return a fail-closed shadow winner and a fully auditable decision report."""
    p = policy or HypothesisPolicy()
    views = [normalize_hypothesis(c) for c in candidates[: p.max_candidates]]
    eligible = [h for h in views if h.eligible and h.fidelity >= p.min_fidelity]

    rejected = [
        {
            "name": h.name,
            "family": h.family,
            "reasons": (
                (["render_failed"] if not h.rendered_ok else [])
                + (["fidelity_below_floor"] if h.fidelity < p.min_fidelity else [])
                + list(h.hard_fail_codes)
                + [f"unmeasured:{x}" for x in h.unmeasured_required]
            ),
        }
        for h in views if h not in eligible
    ]

    if not eligible:
        return {
            "status": "no_eligible_hypothesis",
            "winner": None,
            "winner_source": None,
            "reason": "fail_closed",
            "families_seen": sorted({h.family for h in views}),
            "pareto_frontier": [],
            "rejected": rejected,
        }

    top = max(eligible, key=_rank_key)
    fidelity_floor = max(p.min_fidelity, top.fidelity - p.fidelity_margin)
    near = [
        h for h in eligible
        if h.fidelity >= fidelity_floor and h.edge_f1 >= top.edge_f1 - p.edge_tolerance
    ]

    # Within a safe fidelity/edge envelope, prefer a materially simpler artifact.
    simpler = [
        h for h in near
        if h.path_count <= max(1, int(top.path_count * p.max_path_ratio))
    ]
    if simpler:
        winner = max(simpler, key=lambda h: (-h.path_count, h.fidelity, h.edge_f1, h.total_score, h.name))
        reason = "safe_editability_preference" if winner is not top else "highest_quality"
    else:
        winner = top
        reason = "highest_quality"

    frontier = pareto_frontier(eligible)
    family_best: dict[str, HypothesisView] = {}
    for h in eligible:
        current = family_best.get(h.family)
        if current is None or _rank_key(h) > _rank_key(current):
            family_best[h.family] = h

    return {
        "status": "selected",
        "winner": winner.name,
        "winner_source": winner.source,
        "winner_family": winner.family,
        "reason": reason,
        "families_seen": sorted({h.family for h in views}),
        "family_best": {k: v.name for k, v in sorted(family_best.items())},
        "pareto_frontier": [h.name for h in frontier],
        "eligible": [h.name for h in sorted(eligible, key=_rank_key, reverse=True)],
        "rejected": rejected,
        "policy": {
            "fidelity_margin": p.fidelity_margin,
            "edge_tolerance": p.edge_tolerance,
            "max_path_ratio": p.max_path_ratio,
            "min_fidelity": p.min_fidelity,
            "max_candidates": p.max_candidates,
        },
    }
