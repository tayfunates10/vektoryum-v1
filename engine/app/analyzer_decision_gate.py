"""Verified analyzer decision gate for ``trace_mode=auto``.

AA-2 publishes versioned analyzer metadata. AA-3 verifies that metadata against the
current decoded pixels and feature snapshot before allowing the recommendation to
control production preprocessing. Invalid, stale or ambiguous recommendations are
not consumed; the pipeline uses the color-preserving review mode and the final
artifact is explicitly marked ``needs_review``. Manual explicit modes bypass this
gate unchanged.
"""
from __future__ import annotations

import hmac
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

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
_PRECOMPUTED_ANALYSIS: ContextVar[dict[str, Any] | None] = ContextVar(
    "vektoryum_precomputed_analysis",
    default=None,
)


def bind_precomputed_analysis(analysis: dict[str, Any]) -> Any:
    """Bind one analyzer report to the current request context."""
    return _PRECOMPUTED_ANALYSIS.set(deepcopy(analysis))


def consume_precomputed_analysis() -> dict[str, Any] | None:
    """Consume the request-scoped report once; concurrent contexts stay isolated."""
    analysis = _PRECOMPUTED_ANALYSIS.get()
    if analysis is None:
        return None
    _PRECOMPUTED_ANALYSIS.set(None)
    return deepcopy(analysis)


def reset_precomputed_analysis(token: Any) -> None:
    _PRECOMPUTED_ANALYSIS.reset(token)


def _same_digest(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != 64 or len(right) != 64:
        return False
    return hmac.compare_digest(left, right)


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
    analysis: dict[str, Any],
    image: Image.Image,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Rebuild the contract from current truth and compare all authority fields."""
    stored = analysis.get("analyzer_contract")
    if not isinstance(stored, dict):
        return None, ["contract_missing"]

    errors = _version_errors(stored)
    if stored.get("status") != "valid":
        errors.append("contract_not_valid")

    rebuilt_input = {
        key: value
        for key, value in analysis.items()
        if key not in {
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

    scalar_fields = (
        "confidence",
        "runner_up_mode",
        "runner_up_margin",
        "support_contradiction",
        "optional_signals",
    )
    for name in scalar_fields:
        if stored.get(name) != rebuilt.get(name):
            errors.append(f"metadata_mismatch:{name}")

    if stored.get("support_scores") != rebuilt.get("support_scores"):
        errors.append("metadata_mismatch:support_scores")
    if analysis.get("recommendation_confidence") != stored.get("confidence"):
        errors.append("top_level_mismatch:confidence")
    if analysis.get("recommendation_margin") != stored.get("runner_up_margin"):
        errors.append("top_level_mismatch:margin")
    if not _same_digest(analysis.get("recommendation_digest"), stored.get("recommendation_digest")):
        errors.append("top_level_mismatch:digest")

    selected = analysis.get("recommended_mode")
    if selected not in AUTO_RECOMMENDATION_MODES:
        errors.append("recommendation_unsupported")
    if analysis.get("detected_type") != selected:
        errors.append("recommendation_type_mismatch")

    if errors:
        return None, sorted(set(errors))
    return deepcopy(rebuilt), []


def decide_trace_mode(
    analysis: dict[str, Any],
    image: Image.Image,
    requested_mode: str,
) -> dict[str, Any]:
    """Return the production execution mode and an auditable decision report."""
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
            "runner_up_margin": None,
            "verified_recommendation_digest": None,
        }

    contract, errors = verify_stored_contract(analysis, image)
    if contract is None:
        return {
            "schema_version": AUTO_DECISION_SCHEMA_VERSION,
            "status": "needs_review",
            "requested_mode": "auto",
            "recommended_mode": analysis.get("recommended_mode"),
            "execution_mode": REVIEW_FALLBACK_MODE,
            "abstained": True,
            "reason_codes": errors or ["contract_verification_failed"],
            "confidence": None,
            "runner_up_margin": None,
            "verified_recommendation_digest": None,
        }

    confidence = contract.get("confidence")
    margin = contract.get("runner_up_margin")
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

    abstained = bool(reason_codes)
    return {
        "schema_version": AUTO_DECISION_SCHEMA_VERSION,
        "status": "needs_review" if abstained else "accepted",
        "requested_mode": "auto",
        "recommended_mode": analysis.get("recommended_mode"),
        "execution_mode": REVIEW_FALLBACK_MODE if abstained else analysis["recommended_mode"],
        "abstained": abstained,
        "reason_codes": reason_codes or ["verified_recommendation"],
        "confidence": confidence,
        "runner_up_mode": contract.get("runner_up_mode"),
        "runner_up_margin": margin,
        "verified_recommendation_digest": contract.get("recommendation_digest"),
    }


def apply_auto_decision_to_final_artifact(
    final_artifact: dict[str, Any],
    analysis: dict[str, Any],
    requested_mode: str,
) -> dict[str, Any]:
    """Prevent an abstained auto decision from being reported production-ready."""
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
        warnings.append(
            "Automatic mode confidence was insufficient; a color-preserving review mode was used."
        )
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
