"""Verified analyzer decision gate for ``trace_mode=auto``.

AA-2 publishes versioned analyzer metadata. AA-3 verifies that metadata against the
current decoded pixels and feature snapshot before allowing it to control production
preprocessing. Invalid or stale reports use a conservative color-preserving fallback.
A valid but low-confidence report keeps its existing recommendation as best-effort
execution while the final artifact is explicitly marked ``needs_review``. Manual
explicit modes bypass this gate unchanged.
"""
from __future__ import annotations

import hmac
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

from app.analyzer_contracts import (
    AUTO_RECOMMENDATION_MODES,
    CALIBRATION_VERSION,
    CONTRACT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    SUPPORT_MODEL_VERSION,
    build_analyzer_contract,
)

AUTO_DECISION_SCHEMA_VERSION = "analyzer-auto-decision-v1"
MIN_AUTO_CONFIDENCE = 0.50
MIN_AUTO_MARGIN = 0.05
REVIEW_FALLBACK_MODE = "logo_color"
_NEUTRAL_GUARDED_MODES = frozenset(
    {"single_color", "lineart", "minimal_ai", "geometric_logo"}
)
_NEUTRAL_MAX_CHANNEL_SPREAD = 18
_NEUTRAL_LUMA_BUCKET_WIDTH = 4
_NEUTRAL_MIN_AREA_RATIO = 0.004
_NEUTRAL_MIN_CORE_RETENTION = 0.12
_NEUTRAL_MIN_SEPARATION = 12.0
_NEUTRAL_ANALYSIS_MAX_SIDE = 384
_PRECOMPUTED_ANALYSIS: ContextVar[dict[str, Any] | None] = ContextVar(
    "vektoryum_precomputed_analysis", default=None
)


def bind_precomputed_analysis(analysis: dict[str, Any]) -> Any:
    return _PRECOMPUTED_ANALYSIS.set(deepcopy(analysis))


def consume_precomputed_analysis() -> dict[str, Any] | None:
    analysis = _PRECOMPUTED_ANALYSIS.get()
    if analysis is None:
        return None
    _PRECOMPUTED_ANALYSIS.set(None)
    return deepcopy(analysis)


def reset_precomputed_analysis(token: Any) -> None:
    _PRECOMPUTED_ANALYSIS.reset(token)


def _same_digest(left: Any, right: Any) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and len(left) == len(right) == 64
        and hmac.compare_digest(left, right)
    )


def _neutral_luminance_band_count(image: Image.Image) -> int:
    """Count spatially persistent low-chroma luminance bands deterministically.

    Binary/line-art routes intentionally collapse tone. A source with three or
    more real neutral luminance layers therefore needs a color-preserving route.
    Nearest-neighbour downsampling avoids inventing intermediate anti-alias tones;
    a 3x3 interior-support test rejects thin edge films that remain in the source.
    """
    rgba_image = image.convert("RGBA")
    width, height = rgba_image.size
    longest = max(width, height)
    if longest > _NEUTRAL_ANALYSIS_MAX_SIDE:
        scale = _NEUTRAL_ANALYSIS_MAX_SIDE / float(longest)
        rgba_image = rgba_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.NEAREST,
        )

    rgba = np.asarray(rgba_image, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.size == 0:
        return 0
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    composited = np.rint(
        rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
    ).astype(np.uint8)
    channel_spread = composited.max(axis=2).astype(np.int16) - composited.min(axis=2).astype(np.int16)
    neutral = channel_spread <= _NEUTRAL_MAX_CHANNEL_SPREAD
    if not bool(neutral.any()):
        return 0

    luminance = np.rint(
        0.2126 * composited[:, :, 0]
        + 0.7152 * composited[:, :, 1]
        + 0.0722 * composited[:, :, 2]
    ).astype(np.uint8)
    buckets = luminance.astype(np.int16) // _NEUTRAL_LUMA_BUCKET_WIDTH
    total = int(neutral.size)
    min_area = max(8, int(round(total * _NEUTRAL_MIN_AREA_RATIO)))
    centers: list[tuple[float, int]] = []

    for bucket in np.unique(buckets[neutral]).tolist():
        mask = neutral & (buckets == int(bucket))
        area = int(mask.sum())
        if area < min_area:
            continue
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        core = mask[1:-1, 1:-1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                core &= mask[
                    1 + dy : mask.shape[0] - 1 + dy,
                    1 + dx : mask.shape[1] - 1 + dx,
                ]
        core_area = int(core.sum())
        if core_area < 1 or core_area / float(area) < _NEUTRAL_MIN_CORE_RETENTION:
            continue
        values = luminance[mask].astype(np.float64)
        centers.append((float(values.mean()), area))

    if not centers:
        return 0
    centers.sort(key=lambda item: item[0])
    merged: list[tuple[float, int]] = []
    for center, area in centers:
        if not merged or center - merged[-1][0] >= _NEUTRAL_MIN_SEPARATION:
            merged.append((center, area))
            continue
        previous_center, previous_area = merged[-1]
        combined_area = previous_area + area
        merged[-1] = (
            (previous_center * previous_area + center * area) / float(combined_area),
            combined_area,
        )
    return len(merged)


def _version_errors(contract: dict[str, Any]) -> list[str]:
    expected = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "support_model_version": SUPPORT_MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
    }
    return [
        f"version_mismatch:{name}"
        for name, value in expected.items()
        if contract.get(name) != value
    ]


def verify_stored_contract(
    analysis: dict[str, Any], image: Image.Image
) -> tuple[dict[str, Any] | None, list[str]]:
    stored = analysis.get("analyzer_contract")
    if not isinstance(stored, dict):
        return None, ["contract_missing"]

    errors = _version_errors(stored)
    if stored.get("status") != "valid":
        errors.append("contract_not_valid")

    rebuilt_input = {
        key: value
        for key, value in analysis.items()
        if key
        not in {
            "analyzer_contract",
            "recommendation_confidence",
            "recommendation_margin",
            "recommendation_digest",
            "auto_decision",
        }
    }
    rebuilt = build_analyzer_contract(rebuilt_input, image)
    if rebuilt.get("status") != "valid":
        errors.extend(f"rebuilt:{item}" for item in rebuilt.get("errors", []))
        errors.append("contract_rebuild_invalid")

    for name in ("source_pixel_sha256", "feature_digest", "recommendation_digest"):
        if not _same_digest(stored.get(name), rebuilt.get(name)):
            errors.append(f"digest_mismatch:{name}")

    for name in (
        "confidence",
        "runner_up_mode",
        "runner_up_margin",
        "support_contradiction",
        "optional_signals",
    ):
        if stored.get(name) != rebuilt.get(name):
            errors.append(f"metadata_mismatch:{name}")
    if stored.get("support_scores") != rebuilt.get("support_scores"):
        errors.append("metadata_mismatch:support_scores")
    if analysis.get("recommendation_confidence") != stored.get("confidence"):
        errors.append("top_level_mismatch:confidence")
    if analysis.get("recommendation_margin") != stored.get("runner_up_margin"):
        errors.append("top_level_mismatch:margin")
    if not _same_digest(
        analysis.get("recommendation_digest"), stored.get("recommendation_digest")
    ):
        errors.append("top_level_mismatch:digest")

    selected = analysis.get("recommended_mode")
    if selected not in AUTO_RECOMMENDATION_MODES:
        errors.append("recommendation_unsupported")
    if analysis.get("detected_type") != selected:
        errors.append("recommendation_type_mismatch")

    return (None, sorted(set(errors))) if errors else (deepcopy(rebuilt), [])


def decide_trace_mode(
    analysis: dict[str, Any], image: Image.Image, requested_mode: str
) -> dict[str, Any]:
    if requested_mode != "auto":
        return {
            "schema_version": AUTO_DECISION_SCHEMA_VERSION,
            "status": "manual",
            "requested_mode": requested_mode,
            "recommended_mode": analysis.get("recommended_mode"),
            "execution_mode": requested_mode,
            "abstained": False,
            "reason_codes": ["manual_mode_bypass"],
            "confidence": None,
            "runner_up_mode": None,
            "runner_up_margin": None,
            "verified_recommendation_digest": None,
        }

    neutral_luminance_band_count = _neutral_luminance_band_count(image)
    contract, errors = verify_stored_contract(analysis, image)
    if contract is None:
        return {
            "schema_version": AUTO_DECISION_SCHEMA_VERSION,
            "status": "needs_review",
            "requested_mode": "auto",
            "recommended_mode": analysis.get("recommended_mode"),
            "execution_mode": REVIEW_FALLBACK_MODE,
            "abstained": True,
            "fallback_applied": True,
            "reason_codes": errors or ["contract_verification_failed"],
            "confidence": None,
            "runner_up_mode": None,
            "runner_up_margin": None,
            "verified_recommendation_digest": None,
            "neutral_luminance_band_count": neutral_luminance_band_count,
        }

    confidence = contract.get("confidence")
    margin = contract.get("runner_up_margin")
    selected = str(analysis.get("recommended_mode"))
    if (
        neutral_luminance_band_count >= 3
        and selected in _NEUTRAL_GUARDED_MODES
    ):
        return {
            "schema_version": AUTO_DECISION_SCHEMA_VERSION,
            "status": "needs_review",
            "requested_mode": "auto",
            "recommended_mode": selected,
            "execution_mode": REVIEW_FALLBACK_MODE,
            "abstained": True,
            "fallback_applied": True,
            "reason_codes": ["neutral_luminance_band_guard"],
            "confidence": confidence,
            "runner_up_mode": contract.get("runner_up_mode"),
            "runner_up_margin": margin,
            "verified_recommendation_digest": contract.get("recommendation_digest"),
            "neutral_luminance_band_count": neutral_luminance_band_count,
        }

    reason_codes: list[str] = []
    if not isinstance(confidence, (int, float)):
        reason_codes.append("confidence_missing")
    elif float(confidence) < MIN_AUTO_CONFIDENCE:
        reason_codes.append("confidence_below_minimum")
    if not isinstance(margin, (int, float)):
        reason_codes.append("margin_missing")
    elif float(margin) < MIN_AUTO_MARGIN:
        reason_codes.append("margin_below_minimum")
    if contract.get("support_contradiction") is True:
        reason_codes.append("support_contradiction")

    needs_review = bool(reason_codes)
    return {
        "schema_version": AUTO_DECISION_SCHEMA_VERSION,
        "status": "needs_review" if needs_review else "accepted",
        "requested_mode": "auto",
        "recommended_mode": analysis.get("recommended_mode"),
        # The report is verified, so preserve the mature analyzer's best-effort
        # execution mode. Fail-closed behavior is the review verdict, not an
        # unrelated mode rewrite that can introduce new artifact failures.
        "execution_mode": analysis["recommended_mode"],
        "abstained": needs_review,
        "fallback_applied": False,
        "reason_codes": reason_codes or ["verified_recommendation"],
        "confidence": confidence,
        "runner_up_mode": contract.get("runner_up_mode"),
        "runner_up_margin": margin,
        "verified_recommendation_digest": contract.get("recommendation_digest"),
        "neutral_luminance_band_count": neutral_luminance_band_count,
    }


def apply_auto_decision_to_final_artifact(
    final_artifact: dict[str, Any], analysis: dict[str, Any], requested_mode: str
) -> dict[str, Any]:
    if requested_mode != "auto":
        return final_artifact
    decision = analysis.get("auto_decision")
    if not isinstance(decision, dict) or decision.get("status") != "needs_review":
        return final_artifact

    if final_artifact.get("verdict") == "production_ready":
        final_artifact["verdict"] = "needs_review"
    final_artifact["quality_verdict"] = final_artifact.get("verdict", "needs_review")
    warnings = list(final_artifact.get("soft_warnings") or [])
    codes = list(final_artifact.get("soft_warning_codes") or [])
    if "analyzer_auto_review" not in codes:
        codes.append("analyzer_auto_review")
        warnings.append("Automatic mode confidence requires review.")
    final_artifact["soft_warnings"] = warnings
    final_artifact["soft_warning_codes"] = codes
    final_artifact["auto_decision"] = deepcopy(decision)
    return final_artifact


__all__ = [
    "AUTO_DECISION_SCHEMA_VERSION",
    "MIN_AUTO_CONFIDENCE",
    "MIN_AUTO_MARGIN",
    "REVIEW_FALLBACK_MODE",
    "apply_auto_decision_to_final_artifact",
    "bind_precomputed_analysis",
    "consume_precomputed_analysis",
    "decide_trace_mode",
    "reset_precomputed_analysis",
    "verify_stored_contract",
]
