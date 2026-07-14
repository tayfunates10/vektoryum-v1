from __future__ import annotations

from copy import deepcopy

from core_release_contract import (
    PRODUCTION_MODES,
    REPEAT_COUNT,
    REQUIRED_WORKFLOWS,
    SCHEMA_VERSION,
    validate_release_report,
)


_DIGEST = "a" * 64


def _sample(index: int, *, verdict: str = "production_ready") -> dict:
    return {
        "repeat_index": index,
        "status": "completed",
        "verdict": verdict,
        "reason_codes": [],
        "artifact_sha256": _DIGEST,
        "evaluator_sha256": _DIGEST,
        "output_digest_match": True,
        "score_snapshot_match": True,
        "structure": {
            "structural_safe": True,
            "has_bitmap": False,
            "nonfinite": False,
            "open_required_cycle": False,
            "path_count": 2,
        },
        "metrics": {
            "ink_recall": 0.999,
            "ink_precision": 0.999,
            "component_delta": 0,
            "seam_ratio": 0.0,
            "halo_ratio": 0.0,
        },
    }


def _payload() -> dict:
    modes = []
    for mode in PRODUCTION_MODES:
        photo = mode == "photo_poster"
        modes.append({
            "mode": mode,
            "status": "needs_review" if photo else "production_ready",
            "reason_codes": ["accepted_photo_product_limit"] if photo else [],
            "samples": [
                _sample(index, verdict="needs_review" if photo else "production_ready")
                for index in range(1, REPEAT_COUNT + 1)
            ],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": "test",
        "repeat_count": REPEAT_COUNT,
        "required_workflows": list(REQUIRED_WORKFLOWS),
        "modes": modes,
    }


def _mode(payload: dict, name: str) -> dict:
    return next(item for item in payload["modes"] if item["mode"] == name)


def test_complete_three_run_release_report_passes() -> None:
    report = validate_release_report(_payload())
    assert report["status"] == "pass", report
    assert report["reason_codes"] == []


def test_missing_mode_or_repeat_fails_closed() -> None:
    payload = _payload()
    payload["modes"].pop()
    assert "production_mode_coverage_mismatch" in validate_release_report(payload)["reason_codes"]

    payload = _payload()
    _mode(payload, "flat_logo")["samples"].pop()
    report = validate_release_report(payload)
    assert report["status"] == "fail"
    assert "flat_logo:repeat_sample_count_mismatch" in report["reason_codes"]


def test_digest_drift_and_output_digest_mismatch_fail() -> None:
    payload = _payload()
    samples = _mode(payload, "minimal_ai")["samples"]
    samples[2]["artifact_sha256"] = "b" * 64
    samples[1]["output_digest_match"] = False
    report = validate_release_report(payload)
    assert "minimal_ai:non_deterministic_artifact_digest" in report["reason_codes"]
    assert "minimal_ai:output_digest_mismatch" in report["reason_codes"]


def test_stale_score_bitmap_nonfinite_and_open_cycle_fail() -> None:
    payload = _payload()
    sample = _mode(payload, "geometric_logo")["samples"][0]
    sample["score_snapshot_match"] = False
    sample["structure"]["has_bitmap"] = True
    sample["structure"]["nonfinite"] = True
    sample["structure"]["open_required_cycle"] = True
    report = validate_release_report(payload)
    assert "geometric_logo:stale_score_snapshot" in report["reason_codes"]
    assert "geometric_logo:embedded_bitmap" in report["reason_codes"]
    assert "geometric_logo:nonfinite_geometry" in report["reason_codes"]
    assert "geometric_logo:open_required_cycle" in report["reason_codes"]


def test_in_domain_artifact_thresholds_are_fail_closed() -> None:
    payload = _payload()
    sample = _mode(payload, "lineart")["samples"][0]
    sample["metrics"].update({
        "ink_recall": 0.994,
        "ink_precision": 0.974,
        "component_delta": 1,
        "seam_ratio": 0.003,
        "halo_ratio": 0.021,
    })
    reasons = validate_release_report(payload)["reason_codes"]
    assert "lineart:ink_recall_below_min" in reasons
    assert "lineart:ink_precision_below_min" in reasons
    assert "lineart:component_delta_nonzero" in reasons
    assert "lineart:seam_ratio_above_max" in reasons
    assert "lineart:halo_ratio_above_max" in reasons


def test_photo_mode_can_never_claim_production_ready() -> None:
    payload = _payload()
    photo = _mode(payload, "photo_poster")
    photo["samples"][0]["verdict"] = "production_ready"
    report = validate_release_report(payload)
    assert report["status"] == "fail"
    assert "photo_poster:false_production_ready" in report["reason_codes"]


def test_deterministic_explicit_unavailable_verdict_is_allowed() -> None:
    payload = _payload()
    mode = _mode(payload, "centerline")
    mode["status"] = "unavailable"
    mode["reason_codes"] = ["optional_backend_unavailable"]
    mode["samples"] = [
        {
            "repeat_index": index,
            "status": "unavailable",
            "verdict": "unavailable",
            "reason_codes": ["optional_backend_unavailable"],
            "artifact_sha256": None,
        }
        for index in range(1, REPEAT_COUNT + 1)
    ]
    assert validate_release_report(payload)["status"] == "pass"


def test_mixed_available_and_unavailable_repeats_fail() -> None:
    payload = _payload()
    sample = _mode(payload, "single_color")["samples"][2]
    sample.clear()
    sample.update({
        "repeat_index": 3,
        "status": "unavailable",
        "verdict": "unavailable",
        "reason_codes": ["optional_backend_unavailable"],
        "artifact_sha256": None,
    })
    report = validate_release_report(payload)
    assert "single_color:mixed_repeat_status" in report["reason_codes"]


def test_required_workflow_list_is_pinned() -> None:
    payload = deepcopy(_payload())
    payload["required_workflows"] = ["Exact final SVG contract"]
    assert "required_workflow_contract_mismatch" in validate_release_report(payload)["reason_codes"]
