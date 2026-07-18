"""Immutable historical verifier for the RFV-3E viewBox rollback evidence.

The committed JSON records the pre-fix behavior proved by PR #104. It is not
recomputed against current production code after PR #105; current behavior is
covered by the dedicated viewBox/alpha production contract.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "vektoryum-rfv3e-viewbox-journal-diagnostics-v1"
SOURCE_MAIN_SHA = "19e91d10926f8709112b0afd6c576b886a5dfeb5"
SOURCE_PR = 103
SOURCE_HEAD_SHA = "92fa263a938a39f44c288109c8f05a8a38c98f7e"
SOURCE_RUN_ID = 29623130466
SOURCE_ARTIFACT_ID = 8424383328
SOURCE_ARTIFACT_DIGEST = "sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0"
CASES = ["qualification-public-10", "qualification-public-14", "qualification-public-18"]
_PATH_OR_SECRET = re.compile(
    r"(?:/home/|/tmp/|[A-Za-z]:\\|authorization\s*:|bearer\s+|token=|traceback)",
    re.IGNORECASE,
)


def validate_evidence(payload: dict[str, Any]) -> None:
    expected_source = {
        "repository": "tayfunates10/vektoryum-v1",
        "main_sha": SOURCE_MAIN_SHA,
        "pull_request": SOURCE_PR,
        "measurement_head_sha": SOURCE_HEAD_SHA,
        "workflow_run_id": SOURCE_RUN_ID,
        "aggregate_artifact_id": SOURCE_ARTIFACT_ID,
        "aggregate_artifact_digest": SOURCE_ARTIFACT_DIGEST,
    }
    expected_scope = {
        "case_ids": CASES,
        "observed_hard_fail_code": "viewbox_missing",
        "observed_reason_code": "exact_component_metrics_missing",
    }
    if payload.get("schema") != SCHEMA:
        raise ValueError("historical schema drift")
    if payload.get("source") != expected_source:
        raise ValueError("historical source binding drift")
    if payload.get("scope") != expected_scope:
        raise ValueError("historical scope binding drift")
    diagnosis = payload.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("historical diagnosis missing")
    if diagnosis.get("direct_restore_added_viewbox") is not True:
        raise ValueError("historical restore proof drift")
    if diagnosis.get("repaired_viewbox") != "0 0 48 32":
        raise ValueError("historical repaired viewBox drift")
    if diagnosis.get("stage_measurement_required_unmeasured") != ["alpha_fidelity"]:
        raise ValueError("historical required-unmeasured signature drift")
    if diagnosis.get("journal_with_alpha_requirement") != {
        "accepted": False,
        "status": "rolled_back",
        "reason_codes": ["required_metric_unmeasured"],
        "output_viewbox_present": False,
    }:
        raise ValueError("historical rollback signature drift")
    if diagnosis.get("journal_without_alpha_requirement") != {
        "accepted": True,
        "status": "accepted",
        "reason_codes": ["metrics_non_regressing"],
        "output_viewbox_present": True,
    }:
        raise ValueError("historical control signature drift")
    if diagnosis.get("root_cause_status") != "proven":
        raise ValueError("historical root-cause status drift")
    if diagnosis.get("root_cause_class") != "transform_journal_required_alpha_metric_deadlock":
        raise ValueError("historical root-cause class drift")
    if diagnosis.get("production_fix_authorized") is not False:
        raise ValueError("historical diagnostics cannot authorize production")
    if payload.get("release_decision") != "no_go" or payload.get("rfv4_allowed") is not False:
        raise ValueError("release/RFV-4 decision drift")
    if _PATH_OR_SECRET.search(json.dumps(payload, sort_keys=True, ensure_ascii=True)):
        raise ValueError("path, secret or traceback leaked into historical evidence")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify immutable RFV-3E viewBox evidence")
    parser.add_argument("verify", nargs="?")
    parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate_evidence(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "historical_verified", "root_cause": payload["diagnosis"]["root_cause_class"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
