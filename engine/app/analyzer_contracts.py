"""Versioned, deterministic and fail-closed analyzer decision metadata.

The existing analyzer remains the authority for ``recommended_mode`` in AA-2. This
module validates its public features, computes canonical digests, derives bounded
mode-support scores and calibrates confidence from committed labeled synthetic
feature evidence. AA-3 will decide when ``auto`` must abstain; AA-2 only publishes
truthful metadata and never rewrites a manual or automatic mode decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FEATURE_SCHEMA_VERSION = "analyzer-features-v1"
SUPPORT_MODEL_VERSION = "analyzer-mode-support-v1"
CALIBRATION_VERSION = "analyzer-confidence-calibration-v1"
CONTRACT_SCHEMA_VERSION = "analyzer-decision-contract-v1"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "analyzer_calibration_v1.json"

AUTO_RECOMMENDATION_MODES = (
    "geometric_logo",
    "minimal_ai",
    "logo_color",
    "single_color",
    "lineart",
    "photo_poster",
)

FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    "estimated_color_count": {"kind": "integer", "unit": "count", "minimum": 0, "maximum": 48, "required": True},
    "flat_color_count": {"kind": "integer", "unit": "count", "minimum": 0, "maximum": 48, "required": True},
    "blur_score": {"kind": "number", "unit": "laplacian_variance", "minimum": 0.0, "maximum": 1.0e12, "required": True},
    "edge_density": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": True},
    "quality_score": {"kind": "number", "unit": "score_0_100", "minimum": 0.0, "maximum": 100.0, "required": True},
    "thin_ink_ratio": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": True},
    "straight_edge_likelihood": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": True},
    "corner_likelihood": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": True},
    "has_gradient": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_geometric_logo": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_text_logo": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_color_logo": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_photo_or_complex": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_single_color": {"kind": "boolean", "unit": "flag", "required": True},
    "likely_line_art": {"kind": "boolean", "unit": "flag", "required": True},
    "semantic_photo_like": {"kind": "boolean", "unit": "flag", "required": True},
    "semantic_edge_density": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": False, "optional_signal": "hed"},
    "edge_coherence": {"kind": "number", "unit": "ratio", "minimum": 0.0, "maximum": 1.0, "required": False, "optional_signal": "hed"},
}

_SUPPORT_EVIDENCE_DEFAULTS: dict[str, Any] = {
    "estimated_color_count": None,
    "blur_score": 100.0,
    "semantic_edge_density": None,
    "edge_coherence": None,
}

_CALIBRATION_BINS = (
    (-1.0, 0.05),
    (0.05, 0.20),
    (0.20, 0.40),
    (0.40, 1.0),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgba = np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(f"{rgba.shape[1]}x{rgba.shape[0]}:RGBA8\n".encode("ascii"))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_feature_snapshot(analysis: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], dict[str, str]]:
    errors: list[str] = []
    snapshot: dict[str, Any] = {}

    for name, spec in FEATURE_SCHEMA.items():
        value = analysis.get(name)
        if value is None and not spec.get("required", False):
            snapshot[name] = None
            continue
        if value is None:
            errors.append(f"missing_feature:{name}")
            continue

        kind = spec["kind"]
        if kind == "boolean":
            if not isinstance(value, (bool, np.bool_)):
                errors.append(f"invalid_boolean:{name}")
                continue
            snapshot[name] = bool(value)
            continue

        number = _finite_number(value)
        if number is None:
            errors.append(f"nonfinite_or_invalid:{name}")
            continue
        if number < float(spec["minimum"]) or number > float(spec["maximum"]):
            errors.append(f"out_of_range:{name}")
            continue
        if kind == "integer":
            if not number.is_integer():
                errors.append(f"non_integer:{name}")
                continue
            snapshot[name] = int(number)
        else:
            snapshot[name] = round(number, 8)

    semantic = snapshot.get("semantic_edge_density")
    coherence = snapshot.get("edge_coherence")
    if semantic is None and coherence is None:
        optional_signals = {"hed": "unavailable"}
    elif semantic is not None and coherence is not None:
        optional_signals = {"hed": "measured"}
    else:
        optional_signals = {"hed": "invalid"}
        errors.append("partial_optional_signal:hed")

    if errors:
        return None, sorted(set(errors)), optional_signals
    return snapshot, [], optional_signals


def mode_support_scores(features: dict[str, Any]) -> dict[str, float]:
    flat = float(features["flat_color_count"])
    edge = float(features["edge_density"])
    quality = float(features["quality_score"])
    thin = float(features["thin_ink_ratio"])
    straight = float(features["straight_edge_likelihood"])
    corner = float(features["corner_likelihood"])
    gradient = bool(features["has_gradient"])

    scores = {
        "geometric_logo": (
            0.10
            + 0.40 * bool(features["likely_geometric_logo"])
            + 0.15 * (straight >= 0.34)
            + 0.15 * (corner >= 0.28)
            + 0.10 * (flat <= 8)
            + 0.10 * (not gradient)
        ),
        "minimal_ai": (
            0.10
            + 0.35 * bool(features["likely_text_logo"])
            + 0.15 * (flat <= 12)
            + 0.10 * (not gradient)
            + 0.10 * (quality >= 55)
            + 0.10 * (not bool(features["likely_color_logo"]))
        ),
        "logo_color": (
            0.10
            + 0.35 * bool(features["likely_color_logo"])
            + 0.15 * gradient
            + 0.15 * (flat > 8)
            + 0.10 * bool(features["likely_photo_or_complex"])
            + 0.05 * (quality < 55)
        ),
        "single_color": (
            0.10
            + 0.45 * bool(features["likely_single_color"])
            + 0.15 * (flat <= 3)
            + 0.15 * (thin <= 0.55)
            + 0.10 * (edge < 0.08)
        ),
        "lineart": (
            0.10
            + 0.45 * bool(features["likely_line_art"])
            + 0.15 * (flat <= 4)
            + 0.15 * (thin > 0.55)
            + 0.10 * (edge >= 0.018)
        ),
        "photo_poster": (
            0.10
            + 0.45 * bool(features["semantic_photo_like"])
            + 0.20 * bool(features["likely_photo_or_complex"])
            + 0.10 * (flat > 28)
            + 0.10 * (edge > 0.11)
            + 0.05 * (quality < 45)
        ),
    }
    return {name: round(min(1.0, max(0.0, float(value))), 6) for name, value in scores.items()}


def _ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    order = {mode: index for index, mode in enumerate(AUTO_RECOMMENDATION_MODES)}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


def _margin_for_selected(scores: dict[str, float], selected: str) -> tuple[str, float, float]:
    alternatives = [(name, score) for name, score in scores.items() if name != selected]
    runner_name, runner_score = sorted(
        alternatives,
        key=lambda item: (-item[1], AUTO_RECOMMENDATION_MODES.index(item[0])),
    )[0]
    margin = float(scores[selected]) - float(runner_score)
    return runner_name, round(runner_score, 6), round(margin, 6)


def _bin_index(margin: float) -> int:
    for index, (lower, upper) in enumerate(_CALIBRATION_BINS):
        inside = lower <= margin <= upper if index == 0 else lower < margin <= upper
        if inside:
            return index
    return len(_CALIBRATION_BINS) - 1 if margin > 1.0 else 0


def _normalized_evidence_features(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    normalized.setdefault("estimated_color_count", normalized.get("flat_color_count"))
    for name, value in _SUPPORT_EVIDENCE_DEFAULTS.items():
        if name == "estimated_color_count":
            continue
        normalized.setdefault(name, value)
    return normalized


@lru_cache(maxsize=1)
def load_calibration_evidence() -> dict[str, Any]:
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "analyzer-calibration-evidence-v1":
        raise ValueError("unsupported analyzer calibration evidence schema")
    if payload.get("support_model_version") != SUPPORT_MODEL_VERSION:
        raise ValueError("calibration evidence support-model mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 12:
        raise ValueError("insufficient analyzer calibration evidence")
    ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        label = case.get("label")
        features = case.get("features")
        if not case_id or case_id in ids:
            raise ValueError("duplicate or empty analyzer calibration case id")
        ids.add(case_id)
        if label not in AUTO_RECOMMENDATION_MODES:
            raise ValueError(f"unsupported analyzer calibration label: {label}")
        if not isinstance(features, dict):
            raise ValueError(f"missing analyzer calibration features: {case_id}")
        snapshot, errors, _signals = validate_feature_snapshot(
            _normalized_evidence_features(features)
        )
        if snapshot is None or errors:
            raise ValueError(f"invalid analyzer calibration features: {case_id}: {errors}")
    return payload


@lru_cache(maxsize=1)
def calibration_summary() -> dict[str, Any]:
    payload = load_calibration_evidence()
    bins = [
        {"lower": lower, "upper": upper, "correct": 0, "total": 0}
        for lower, upper in _CALIBRATION_BINS
    ]
    for case in payload["cases"]:
        features = _normalized_evidence_features(case["features"])
        scores = mode_support_scores(features)
        ranked = _ranked(scores)
        predicted = ranked[0][0]
        margin = float(ranked[0][1]) - float(ranked[1][1])
        entry = bins[_bin_index(margin)]
        entry["total"] += 1
        entry["correct"] += int(predicted == case["label"])

    for entry in bins:
        entry["confidence"] = round(
            (entry["correct"] + 1.0) / (entry["total"] + 2.0),
            6,
        )
    return {
        "schema_version": CALIBRATION_VERSION,
        "method": "laplace-smoothed-margin-bin-v1",
        "evidence_schema_version": payload["schema_version"],
        "evidence_case_count": len(payload["cases"]),
        "evidence_sha256": _sha256(payload),
        "bins": bins,
    }


def _calibrated_confidence(
    selected_score: float,
    margin: float,
    *,
    hed_status: str,
) -> tuple[float, dict[str, Any]]:
    summary = calibration_summary()
    selected_bin = dict(summary["bins"][_bin_index(margin)])
    confidence = min(float(selected_score), float(selected_bin["confidence"]))
    if margin < 0:
        confidence = min(confidence, 0.25)
    if hed_status == "unavailable":
        confidence = min(confidence, 0.85)
    return round(max(0.0, min(1.0, confidence)), 6), selected_bin


def build_analyzer_contract(analysis: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    selected = analysis.get("recommended_mode")
    snapshot, errors, optional_signals = validate_feature_snapshot(analysis)
    if selected not in AUTO_RECOMMENDATION_MODES:
        errors.append("unsupported_recommendation")
    if analysis.get("detected_type") != selected:
        errors.append("detected_type_recommendation_mismatch")

    source_digest = decoded_pixel_sha256(image)
    summary = calibration_summary()
    base = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "support_model_version": SUPPORT_MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "status": "invalid" if errors else "valid",
        "errors": sorted(set(errors)),
        "source_pixel_sha256": source_digest,
        "feature_digest": None,
        "recommendation_digest": None,
        "confidence": None,
        "runner_up_mode": None,
        "runner_up_margin": None,
        "support_scores": None,
        "support_contradiction": None,
        "optional_signals": optional_signals,
        "calibration": {
            "evidence_case_count": summary["evidence_case_count"],
            "evidence_sha256": summary["evidence_sha256"],
            "selected_bin": None,
        },
    }
    if errors or snapshot is None:
        return base

    feature_payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": snapshot,
    }
    feature_digest = _sha256(feature_payload)
    scores = mode_support_scores(snapshot)
    runner_mode, _runner_score, margin = _margin_for_selected(scores, str(selected))
    confidence, selected_bin = _calibrated_confidence(
        scores[str(selected)],
        margin,
        hed_status=optional_signals["hed"],
    )
    recommendation_payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "source_pixel_sha256": source_digest,
        "feature_digest": feature_digest,
        "recommended_mode": selected,
        "confidence": confidence,
        "runner_up_mode": runner_mode,
        "runner_up_margin": margin,
        "support_scores": scores,
        "optional_signals": optional_signals,
    }
    base.update(
        {
            "feature_digest": feature_digest,
            "recommendation_digest": _sha256(recommendation_payload),
            "confidence": confidence,
            "runner_up_mode": runner_mode,
            "runner_up_margin": margin,
            "support_scores": scores,
            "support_contradiction": margin < 0,
            "calibration": {
                "evidence_case_count": summary["evidence_case_count"],
                "evidence_sha256": summary["evidence_sha256"],
                "selected_bin": selected_bin,
            },
        }
    )
    return base


def attach_analyzer_contract(analysis: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Mutate and return the existing analyzer report without changing its decision."""
    try:
        contract = build_analyzer_contract(analysis, image)
    except BaseException as exc:  # fail closed without breaking legacy analysis availability
        contract = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "support_model_version": SUPPORT_MODEL_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "status": "invalid",
            "errors": [f"contract_exception:{type(exc).__name__}"],
            "source_pixel_sha256": None,
            "feature_digest": None,
            "recommendation_digest": None,
            "confidence": None,
            "runner_up_mode": None,
            "runner_up_margin": None,
            "support_scores": None,
            "support_contradiction": None,
            "optional_signals": {"hed": "invalid"},
            "calibration": None,
        }
    analysis["analyzer_contract"] = contract
    analysis["recommendation_confidence"] = contract.get("confidence")
    analysis["recommendation_margin"] = contract.get("runner_up_margin")
    analysis["recommendation_digest"] = contract.get("recommendation_digest")
    return analysis


__all__ = [
    "AUTO_RECOMMENDATION_MODES",
    "CALIBRATION_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_VERSION",
    "SUPPORT_MODEL_VERSION",
    "attach_analyzer_contract",
    "build_analyzer_contract",
    "calibration_summary",
    "decoded_pixel_sha256",
    "load_calibration_evidence",
    "mode_support_scores",
    "validate_feature_snapshot",
]
