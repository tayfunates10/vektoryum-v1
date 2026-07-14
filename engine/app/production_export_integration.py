"""Production export integration for digest-pinned canonical SVG cutover.

The core pipeline publishes a promotion-ready canonical report into a bounded,
job-scoped registry. The production ``export_all`` wrapper always generates the
legacy artifacts first. Canonical SVG replaces the legacy SVG only when the
runtime feature flag, approved SHA-256, HG-7 promotion report and HG-8 controlled
cutover all agree. Derived formats are then regenerated from the exact canonical
SVG; stale legacy derivatives are removed before each regeneration attempt.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping

from app.controlled_svg_cutover import select_controlled_svg_output
from app.exporters import export_dxf, export_eps, export_pdf, export_png
from app.pipeline_canonical_report import PipelineCanonicalSvgReport
from app.production_serializer_runtime import publish_runtime_svg
from app.production_svg_selector import ProductionSvgSelection, atomic_publish_svg

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_MAX_PENDING = 128
_PENDING: OrderedDict[str, PipelineCanonicalSvgReport] = OrderedDict()
_PENDING_LOCK = Lock()


def _job_key(job_dir: Path) -> str:
    return str(Path(job_dir).resolve())


def register_pipeline_canonical_report(
    job_dir: Path,
    pipeline_result: dict[str, Any],
) -> bool:
    """Register one promotion-ready report for the matching export call.

    Disabled, invalid or winner-less results are never retained. The registry is
    bounded as defense-in-depth for requests that terminate before export.
    """
    key = _job_key(job_dir)
    report = pipeline_result.get("canonical_svg_candidate")
    ready = (
        pipeline_result.get("best") is not None
        and isinstance(report, PipelineCanonicalSvgReport)
        and report.ready
    )
    with _PENDING_LOCK:
        _PENDING.pop(key, None)
        if not ready:
            return False
        _PENDING[key] = report
        while len(_PENDING) > _MAX_PENDING:
            _PENDING.popitem(last=False)
    return True


def consume_pipeline_canonical_report(job_dir: Path) -> PipelineCanonicalSvgReport | None:
    """Consume a report exactly once so it cannot leak across requests."""
    with _PENDING_LOCK:
        return _PENDING.pop(_job_key(job_dir), None)


def pending_report_count() -> int:
    with _PENDING_LOCK:
        return len(_PENDING)


def _runtime_enabled(env: Mapping[str, str]) -> bool:
    return str(env.get("VEKTORYUM_CANONICAL_SVG_ENABLED", "off")).strip().lower() in _TRUE_VALUES


def _remove_stale(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        # The exporter will fail explicitly if it cannot replace the file.
        pass


def _restore_legacy_svg(legacy_svg: Path, legacy_bytes: bytes) -> str | None:
    if not legacy_bytes:
        return "legacy SVG backup is empty"
    try:
        selection = ProductionSvgSelection(
            svg_bytes=legacy_bytes,
            selected_path="legacy",
            promoted=False,
            output_sha256=sha256(legacy_bytes).hexdigest(),
            errors=(),
        )
        atomic_publish_svg(selection, legacy_svg)
    except Exception as exc:  # noqa: BLE001 - report rollback failure explicitly
        return f"legacy rollback failed: {type(exc).__name__}: {exc}"
    return None


def _regenerate_derived_outputs(
    *,
    source_svg: Path,
    job_dir: Path,
    job_id: str,
    outputs: dict[str, str],
    errors: dict[str, str],
    formats: tuple[str, ...],
    png_size: tuple[int, int] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    exporters = {"pdf": export_pdf, "eps": export_eps, "dxf": export_dxf}
    for fmt in formats:
        if fmt == "svg":
            continue
        if fmt not in {"pdf", "eps", "dxf", "png"}:
            continue
        dst = Path(job_dir) / f"{job_id}.{fmt}"
        stale = Path(outputs.get(fmt, str(dst)))
        outputs.pop(fmt, None)
        errors.pop(fmt, None)
        _remove_stale(stale)
        if stale != dst:
            _remove_stale(dst)
        try:
            if fmt == "png":
                width, height = png_size if png_size else (None, None)
                export_png(source_svg, dst, width=width, height=height)
            else:
                exporters[fmt](source_svg, dst)
            outputs[fmt] = str(dst)
        except Exception as exc:  # noqa: BLE001 - one format must not poison others
            errors[fmt] = str(exc)
            _remove_stale(dst)
    return outputs, errors


def export_all_with_canonical(
    legacy_export_all: Callable[..., tuple[dict[str, str], dict[str, str]]],
    *,
    best_svg: Path,
    job_dir: Path,
    job_id: str,
    candidate_id: str | None = None,
    formats: tuple[str, ...] = ("svg", "pdf", "eps", "dxf", "png"),
    png_size: tuple[int, int] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Export legacy first, then promote an exact canonical SVG if authorized."""
    try:
        outputs, errors = legacy_export_all(
            best_svg=best_svg,
            job_dir=job_dir,
            job_id=job_id,
            candidate_id=candidate_id,
            formats=formats,
            png_size=png_size,
        )
    except Exception:
        consume_pipeline_canonical_report(job_dir)
        raise

    report = consume_pipeline_canonical_report(job_dir)
    if report is None:
        return outputs, errors

    env = os.environ if environ is None else environ
    if not _runtime_enabled(env):
        return outputs, errors

    candidate = report.candidate
    if (
        not report.ready
        or candidate is None
        or candidate.document is None
        or candidate.promotion is None
    ):
        return outputs, {**errors, "canonical_svg": "canonical pipeline report is not ready"}

    legacy_value = outputs.get("svg")
    if not legacy_value:
        return outputs, {**errors, "canonical_svg": "legacy SVG output is missing"}
    legacy_svg = Path(legacy_value)
    legacy_bytes = b""

    try:
        legacy_bytes = legacy_svg.read_bytes()
        legacy_text = legacy_bytes.decode("utf-8")
        approved = str(env.get("VEKTORYUM_CANONICAL_SVG_SHA256", "")).strip().lower()
        cutover = select_controlled_svg_output(
            legacy_svg_text=legacy_text,
            candidate=candidate.document,
            promotion=candidate.promotion,
            cutover_enabled=True,
            approved_document_sha256=approved,
        )
        if not cutover.promoted:
            reason = "; ".join(cutover.errors) or "controlled canonical cutover rejected"
            return outputs, {**errors, "canonical_svg": reason}

        runtime = publish_runtime_svg(
            legacy_svg=legacy_svg,
            destination=legacy_svg,
            cutover=cutover,
            environ=env,
        )
        if not runtime.published or not runtime.selection.promoted:
            raise ValueError(
                "; ".join(runtime.selection.errors)
                or "runtime canonical publication rejected"
            )

        published_bytes = legacy_svg.read_bytes()
        if sha256(published_bytes).hexdigest() != report.document_sha256:
            raise ValueError("published canonical SVG digest mismatch")
    except Exception as exc:  # noqa: BLE001 - restore legacy before returning
        rollback_error = None
        try:
            changed = bool(legacy_bytes) and legacy_svg.read_bytes() != legacy_bytes
        except OSError:
            changed = bool(legacy_bytes)
        if changed:
            rollback_error = _restore_legacy_svg(legacy_svg, legacy_bytes)
        detail = f"canonical runtime integration failed: {type(exc).__name__}: {exc}"
        if rollback_error:
            detail = f"{detail}; {rollback_error}"
        return outputs, {**errors, "canonical_svg": detail}

    outputs = dict(outputs)
    errors = dict(errors)
    outputs["svg"] = str(legacy_svg)
    errors.pop("svg", None)
    return _regenerate_derived_outputs(
        source_svg=legacy_svg,
        job_dir=Path(job_dir),
        job_id=job_id,
        outputs=outputs,
        errors=errors,
        formats=formats,
        png_size=png_size,
    )


__all__ = [
    "consume_pipeline_canonical_report",
    "export_all_with_canonical",
    "pending_report_count",
    "register_pipeline_canonical_report",
]
