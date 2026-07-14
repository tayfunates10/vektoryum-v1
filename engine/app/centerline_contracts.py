"""Runtime contracts for measured centerline fallback candidates.

Only SVGs carrying the graph fallback metadata are affected. AutoTrace and all
other production modes retain their existing scoring and quality behavior.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable

from app.centerline_graph import validate_centerline_report
from app.centerline_svg import read_centerline_report

_SCORE_INSTALLED = False
_QUALITY_INSTALLED = False


def _attach_score_contract(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def wrapped(
        original_path: Path,
        svg_path: Path,
        analysis_report: dict[str, Any],
        mode: str,
        geometry_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        score = original(
            original_path=original_path,
            svg_path=svg_path,
            analysis_report=analysis_report,
            mode=mode,
            geometry_report=geometry_report,
        )
        if mode != "centerline":
            return score
        report = read_centerline_report(Path(svg_path))
        if report is None:
            # AutoTrace does not carry the OpenCV fallback report.
            return score

        valid, errors = validate_centerline_report(report)
        details = dict(score.get("score_details") or {})
        details.update({
            "centerline_backend": report.get("backend"),
            "centerline_measurement_available": report.get("measurement_available"),
            "centerline_valid": valid,
            "centerline_confidence": report.get("confidence"),
            "centerline_topology": report.get("topology"),
            "centerline_errors": list(errors),
            "centerline_report": report,
        })
        result = {**score, "score_details": details}
        if not valid:
            warnings = list(details.get("warnings") or [])
            warnings.extend(errors)
            details["warnings"] = list(dict.fromkeys(warnings))
            result.update({
                "score_details": details,
                "warning_score": 0.0,
                "total_score": 0.0,
            })
        return result

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(wrapped, "_vektoryum_centerline_contract", True)
    return wrapped


def _attach_quality_contract(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def wrapped(
        score_details: dict[str, Any],
        mode: str,
        geometry_report: dict[str, float] | None = None,
        total_score: float = 0.0,
        fidelity_score: float | None = None,
        structure_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = original(
            score_details=score_details,
            mode=mode,
            geometry_report=geometry_report,
            total_score=total_score,
            fidelity_score=fidelity_score,
            structure_report=structure_report,
        )
        if mode != "centerline" or score_details.get("centerline_backend") != "opencv_skeleton_graph":
            return result

        report = score_details.get("centerline_report")
        valid, errors = validate_centerline_report(report)
        updated = dict(result)
        metrics = dict(updated.get("metrics") or {})
        metrics.update({
            "centerline_backend": "opencv_skeleton_graph",
            "centerline_confidence": score_details.get("centerline_confidence"),
            "centerline_topology": score_details.get("centerline_topology"),
            "centerline_measurement_available": score_details.get("centerline_measurement_available"),
            "centerline_valid": valid,
        })
        updated["metrics"] = metrics
        if not valid:
            updated["status"] = "needs_review"
            warnings = list(updated.get("warnings") or [])
            warnings.append(
                "Centerline fallback topology could not be verified; production-ready status is blocked."
            )
            warnings.extend(f"centerline:{code}" for code in errors)
            updated["warnings"] = list(dict.fromkeys(warnings))
        return updated

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(wrapped, "_vektoryum_centerline_contract", True)
    return wrapped


def install_centerline_quality_contract() -> None:
    """Install the lightweight legacy quality guard during package startup."""
    global _QUALITY_INSTALLED
    if _QUALITY_INSTALLED:
        return
    from app import quality  # noqa: PLC0415

    current = quality.basic_svg_quality_check
    if not getattr(current, "_vektoryum_centerline_contract", False):
        quality.basic_svg_quality_check = _attach_quality_contract(current)
    _QUALITY_INSTALLED = True


def install_centerline_score_contract() -> None:
    """Install scoring only after the fallback actually produced an SVG.

    This avoids importing the heavy scoring/fidelity stack for unrelated modes
    during package startup.  The already-imported pipeline alias is updated too,
    because worker code imports ``score_vector_candidate`` by value.
    """
    global _SCORE_INSTALLED
    if _SCORE_INSTALLED:
        return
    from app import scoring  # noqa: PLC0415

    current = scoring.score_vector_candidate
    if not getattr(current, "_vektoryum_centerline_contract", False):
        current = _attach_score_contract(current)
        scoring.score_vector_candidate = current
    pipeline = sys.modules.get("app.pipeline")
    if pipeline is not None:
        setattr(pipeline, "score_vector_candidate", current)
    _SCORE_INSTALLED = True


__all__ = [
    "install_centerline_quality_contract",
    "install_centerline_score_contract",
]
