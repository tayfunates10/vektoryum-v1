from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "docs" / "real_world_fidelity" / "evidence" / "rfv3_pipeline_results.json"
REQUIRED_COMPONENT_METRICS = ("ssim", "edge_f1", "alpha_iou", "delta_e00")


class MetricCoverageError(RuntimeError):
    pass


def load_results(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricCoverageError(f"invalid RFV-3 results file: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise MetricCoverageError("RFV-3 results schema mismatch")
    return payload


def classify_missing_metrics(metrics: dict[str, Any]) -> tuple[list[str], str | None]:
    missing = sorted(name for name in REQUIRED_COMPONENT_METRICS if metrics.get(name) is None)
    if not missing:
        return [], None

    # FinalArtifactEvaluator creates B_visual, C_color and D_edge_geometry after a
    # successful exact render. The committed affected cases retain alpha_iou but
    # lose SSIM and edge F1, with delta_e00 optionally absent as part of the same
    # partial quality/legacy-report fallback. Do not fabricate any missing value.
    partial_fallback_signatures = {
        ("edge_f1", "ssim"),
        ("delta_e00", "edge_f1", "ssim"),
    }
    if tuple(missing) in partial_fallback_signatures and metrics.get("alpha_iou") is not None:
        return missing, "partial_quality_report_fallback"
    return missing, "unclassified_required_metric_gap"


def diagnose(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise MetricCoverageError("RFV-3 results must contain a results list")

    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            raise MetricCoverageError("invalid RFV-3 result row")
        missing, reason = classify_missing_metrics(row["metrics"])
        if missing:
            cases.append({
                "case_id": row.get("case_id"),
                "missing_metrics": missing,
                "diagnosis": reason,
            })

    return {
        "schema": "vektoryum-rfv3-metric-coverage-diagnostics-v1",
        "case_count": len(rows),
        "missing_metric_case_count": len(cases),
        "cases": sorted(cases, key=lambda item: str(item["case_id"])),
        "fail_closed": bool(cases),
        "release_decision": "no_go" if cases else "unchanged",
        "rfv4_allowed": False,
    }


def verify_expected_diagnosis(report: dict[str, Any]) -> None:
    expected = ["qualification-public-10", "qualification-public-14", "qualification-public-18"]
    actual = [item["case_id"] for item in report["cases"]]
    if actual != expected:
        raise MetricCoverageError(f"unexpected missing-metric cases: {actual}")
    if any(item["diagnosis"] != "partial_quality_report_fallback" for item in report["cases"]):
        raise MetricCoverageError("missing metrics were not classified as partial report fallback")
    if report["fail_closed"] is not True or report["release_decision"] != "no_go" or report["rfv4_allowed"] is not False:
        raise MetricCoverageError("RFV-3D1 fail-closed decision drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--verify-committed", action="store_true")
    args = parser.parse_args()

    report = diagnose(load_results(args.results))
    if args.verify_committed:
        verify_expected_diagnosis(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
