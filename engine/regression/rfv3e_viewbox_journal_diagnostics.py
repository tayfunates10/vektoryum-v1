"""Fail-closed RFV-3E diagnosis for the selected-SVG viewBox rollback.

This module proves a narrow production-control defect without changing production
behaviour.  ``pipeline._restore_source_dimensions`` can infer a valid viewBox from
finite width/height, but ``TransformJournal`` rolls that candidate back for RGBA
sources because its bounded stage measurement leaves ``alpha_fidelity`` in
``required_unmeasured``.  The control run with no unmeasured required metric accepts
exactly the same byte transformation.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from app.pipeline import _restore_source_dimensions
from app.transform_journal import TransformJournal, _measure_svg_bytes

SCHEMA = "vektoryum-rfv3e-viewbox-journal-diagnostics-v1"
SOURCE_MAIN_SHA = "19e91d10926f8709112b0afd6c576b886a5dfeb5"
SOURCE_PR = 103
SOURCE_HEAD_SHA = "92fa263a938a39f44c288109c8f05a8a38c98f7e"
SOURCE_RUN_ID = 29623130466
SOURCE_ARTIFACT_ID = 8424383328
SOURCE_ARTIFACT_DIGEST = "sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0"
CASES = [
    "qualification-public-10",
    "qualification-public-14",
    "qualification-public-18",
]
_REQUIRED_VISUAL_KEYS = {
    "ssim",
    "edge_f1_1px",
    "seam_ratio",
    "component_delta",
    "hole_delta",
}
_PATH_OR_SECRET = re.compile(
    r"(?:/home/|/tmp/|[A-Za-z]:\\|authorization\s*:|bearer\s+|token=|traceback)",
    re.IGNORECASE,
)


def _svg_bytes(*, viewbox: bool) -> bytes:
    vb = ' viewBox="0 0 48 32"' if viewbox else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="32"{vb}>'
        '<path fill="#ffffff" d="M0 0H48V32H0Z"/>'
        "</svg>"
    ).encode("utf-8")


def _viewbox(data: bytes) -> str | None:
    root = ET.fromstring(data)
    return root.attrib.get("viewBox")


def _journal_metric(data: bytes, *, required_unmeasured: list[str]) -> dict[str, Any]:
    has_viewbox = _viewbox(data) is not None
    return {
        "sha256": "0" * 64,
        "byte_size": len(data),
        "structural_safe": has_viewbox,
        "structural_failure_codes": [] if has_viewbox else ["viewbox_missing"],
        "structural_failures": [] if has_viewbox else ["viewBox is required"],
        "path_count": 1,
        "node_count": 5,
        "gradient_definition_count": 0,
        "required_unmeasured": list(required_unmeasured),
        "ssim": 1.0,
        "edge_f1_1px": 1.0,
        "seam_ratio": 0.0,
        "component_delta": 0,
        "hole_delta": 0,
    }


def _run_journal_case(*, required_metrics: set[str]) -> dict[str, Any]:
    source_rgb = np.full((32, 48, 3), 255, dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="rfv3e-viewbox-") as directory:
        path = Path(directory) / "winner.svg"
        path.write_bytes(_svg_bytes(viewbox=False))
        journal = TransformJournal(
            path,
            source_rgb,
            image_class="clean_logo",
            required_metrics=required_metrics,
            budget_seconds=30.0,
            stage_timeout_seconds=30.0,
        )
        required = sorted(required_metrics)
        journal._measure = lambda data: _journal_metric(  # type: ignore[method-assign]
            data, required_unmeasured=required
        )
        accepted, report, stage = journal.run_in_place(
            "restore_source_dimensions",
            path,
            lambda candidate: (
                _restore_source_dimensions(candidate, {"width": 48, "height": 32})
                or {"attempted": True}
            ),
        )
        return {
            "accepted": accepted,
            "status": stage["status"],
            "reason_codes": stage["reason_codes"],
            "output_viewbox_present": _viewbox(path.read_bytes()) is not None,
            "transform_report": report,
        }


def run_diagnostic() -> dict[str, Any]:
    source_rgb = np.full((32, 48, 3), 255, dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="rfv3e-viewbox-direct-") as directory:
        candidate = Path(directory) / "candidate.svg"
        candidate.write_bytes(_svg_bytes(viewbox=False))
        _restore_source_dimensions(candidate, {"width": 48, "height": 32})
        direct_viewbox = _viewbox(candidate.read_bytes())
        candidate_bytes = candidate.read_bytes()

    # Exercise the real bounded stage-measurement contract while replacing only
    # the renderer with an exact deterministic source-sized image.  The visual
    # metrics become available, yet alpha_fidelity remains unmeasured.
    with patch("app.fidelity.render_svg_to_rgb", return_value=source_rgb.copy()):
        measured = _measure_svg_bytes(
            candidate_bytes,
            source_rgb,
            required_metrics={"alpha_fidelity"},
            _allow_topology_refinement=False,
        )

    blocked = _run_journal_case(required_metrics={"alpha_fidelity"})
    control = _run_journal_case(required_metrics=set())

    payload = {
        "schema": SCHEMA,
        "source": {
            "repository": "tayfunates10/vektoryum-v1",
            "main_sha": SOURCE_MAIN_SHA,
            "pull_request": SOURCE_PR,
            "measurement_head_sha": SOURCE_HEAD_SHA,
            "workflow_run_id": SOURCE_RUN_ID,
            "aggregate_artifact_id": SOURCE_ARTIFACT_ID,
            "aggregate_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        },
        "scope": {
            "case_ids": CASES,
            "observed_hard_fail_code": "viewbox_missing",
            "observed_reason_code": "exact_component_metrics_missing",
        },
        "diagnosis": {
            "direct_restore_added_viewbox": direct_viewbox == "0 0 48 32",
            "repaired_viewbox": direct_viewbox,
            "stage_measurement_structural_safe": measured.get("structural_safe") is True,
            "stage_measurement_visual_metrics_complete": _REQUIRED_VISUAL_KEYS.issubset(measured),
            "stage_measurement_required_unmeasured": measured.get("required_unmeasured"),
            "journal_with_alpha_requirement": {
                key: blocked[key]
                for key in ("accepted", "status", "reason_codes", "output_viewbox_present")
            },
            "journal_without_alpha_requirement": {
                key: control[key]
                for key in ("accepted", "status", "reason_codes", "output_viewbox_present")
            },
            "root_cause_status": "proven",
            "root_cause_class": "transform_journal_required_alpha_metric_deadlock",
            "root_cause_summary": (
                "restore_source_dimensions creates a valid viewBox, but the RGBA transform "
                "journal rolls it back solely because alpha_fidelity remains unmeasured"
            ),
            "production_fix_authorized": False,
            "allowed_fix_scope": [
                "measure alpha fidelity in the transform journal before enforcing it",
                "or gate the mandatory coordinate-contract repair with a dedicated fail-closed structural policy",
            ],
            "forbidden_scope": [
                "changing evaluator thresholds",
                "fabricating alpha or visual metrics",
                "changing winner selection",
                "changing corpus, repeat count, timeout or retry policy",
                "enabling RFV-4",
            ],
        },
        "release_decision": "no_go",
        "rfv4_allowed": False,
        "next_branch": "agent/rfv-3e-exact-metric-path-viewbox-fix",
    }
    validate_evidence(payload)
    return payload


def validate_evidence(payload: dict[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "source": {
            "repository": "tayfunates10/vektoryum-v1",
            "main_sha": SOURCE_MAIN_SHA,
            "pull_request": SOURCE_PR,
            "measurement_head_sha": SOURCE_HEAD_SHA,
            "workflow_run_id": SOURCE_RUN_ID,
            "aggregate_artifact_id": SOURCE_ARTIFACT_ID,
            "aggregate_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        },
        "scope": {
            "case_ids": CASES,
            "observed_hard_fail_code": "viewbox_missing",
            "observed_reason_code": "exact_component_metrics_missing",
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"RFV-3E viewBox diagnostic binding mismatch: {key}")

    diagnosis = payload.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("RFV-3E viewBox diagnosis is missing")
    if diagnosis.get("direct_restore_added_viewbox") is not True:
        raise ValueError("source-dimension repair did not add the expected viewBox")
    if diagnosis.get("repaired_viewbox") != "0 0 48 32":
        raise ValueError("source-dimension repair viewBox drift")
    if diagnosis.get("stage_measurement_structural_safe") is not True:
        raise ValueError("repaired candidate is not structurally safe")
    if diagnosis.get("stage_measurement_visual_metrics_complete") is not True:
        raise ValueError("repaired candidate visual stage metrics are incomplete")
    if diagnosis.get("stage_measurement_required_unmeasured") != ["alpha_fidelity"]:
        raise ValueError("alpha required-unmeasured signature drift")

    blocked = diagnosis.get("journal_with_alpha_requirement")
    if blocked != {
        "accepted": False,
        "status": "rolled_back",
        "reason_codes": ["required_metric_unmeasured"],
        "output_viewbox_present": False,
    }:
        raise ValueError("RGBA journal rollback signature drift")
    control = diagnosis.get("journal_without_alpha_requirement")
    if control != {
        "accepted": True,
        "status": "accepted",
        "reason_codes": ["metrics_non_regressing"],
        "output_viewbox_present": True,
    }:
        raise ValueError("viewBox control acceptance signature drift")
    if diagnosis.get("root_cause_status") != "proven":
        raise ValueError("root cause must remain proven")
    if diagnosis.get("root_cause_class") != "transform_journal_required_alpha_metric_deadlock":
        raise ValueError("root cause class drift")
    if diagnosis.get("production_fix_authorized") is not False:
        raise ValueError("diagnostics phase cannot authorize a production fix")
    if payload.get("release_decision") != "no_go" or payload.get("rfv4_allowed") is not False:
        raise ValueError("release/RFV-4 decision drift")
    if payload.get("next_branch") != "agent/rfv-3e-exact-metric-path-viewbox-fix":
        raise ValueError("next branch drift")

    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    if _PATH_OR_SECRET.search(serialized):
        raise ValueError("path, secret or traceback leaked into diagnostics evidence")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFV-3E viewBox journal diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "build":
        payload = run_diagnostic()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "built", "root_cause": payload["diagnosis"]["root_cause_class"]}, sort_keys=True))
        return 0
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate_evidence(payload)
        live = run_diagnostic()
        if payload != live:
            raise ValueError("committed evidence does not match the live deterministic diagnostic")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "verified", "root_cause": payload["diagnosis"]["root_cause_class"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
