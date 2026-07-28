"""Compatibility binding for exact painter TransformJournal measurement reuse.

The current painter helper does not expose its internal measurement cache. This
module replaces only that helper with the same TransformJournal contract while
capturing the request-scoped cache. A later final source-alpha journal may reuse
it only through :mod:`app.painter_measurement_reuse`, whose SHA/source/proof
identity checks remain authoritative. Any mismatch performs a fresh measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app import painter_measurement_reuse as _reuse

_INSTALLED = False


def run_painter_geometry_journal_with_capture(
    parent_path: Path,
    candidate_path: Path,
    journal_source_rgb: Any,
    image_class: str,
    transform_report: Any,
    measurement_cache: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Run the unchanged journal gates and register only an exact passing proof."""
    from app.transform_journal import TransformJournal

    cache = measurement_cache if isinstance(measurement_cache, dict) else {}
    journal = TransformJournal(
        Path(parent_path),
        journal_source_rgb,
        image_class=image_class,
        required_metrics=set(),
    )
    journal._cache = cache
    accepted_path, stage = journal.consider_candidate(
        "source_alpha_painter_candidate",
        Path(parent_path),
        Path(candidate_path),
        transform_report=transform_report,
    )
    passed = Path(accepted_path) == Path(candidate_path)
    reasons = [] if passed else [
        str(code) for code in (stage.get("reason_codes") or ["unknown"])
    ]
    if passed:
        _reuse._register(
            Path(parent_path),
            Path(candidate_path),
            journal_source_rgb,
            str(image_class),
            set(),
            int(journal.max_side),
            cache,
            transform_report,
        )
    return passed, reasons


def install_painter_measurement_capture() -> None:
    """Bind capture and exact reuse once without changing any journal threshold."""
    global _INSTALLED
    if _INSTALLED:
        return

    from app import alpha_candidate_painter
    from app.transform_journal import TransformJournal

    run_painter_geometry_journal_with_capture.__vektoryum_measurement_capture__ = True
    alpha_candidate_painter._run_painter_geometry_journal = (
        run_painter_geometry_journal_with_capture
    )
    alpha_candidate_painter.apply_candidate_painter_reconstruction = (
        _reuse._wrap_painter_apply(
            alpha_candidate_painter.apply_candidate_painter_reconstruction
        )
    )
    TransformJournal._measure = _reuse._wrap_measure(TransformJournal._measure)
    TransformJournal.consider_candidate = _reuse._wrap_consider(
        TransformJournal.consider_candidate
    )
    _INSTALLED = True
