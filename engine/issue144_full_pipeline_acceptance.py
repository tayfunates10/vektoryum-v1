"""Fail-closed full-pipeline acceptance gate for Issue #144 (AI-2 Round 2).

This gate consumes evidence emitted by the unchanged 12-case production
execution. It deliberately does *not* treat the diagnostic runner's
``FAIL_ON=none`` exit code as quality acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

AI2_NATIVE_CASES = (
    "qa-gray-border-counter",
    "qa-lowres-badge",
    "qa-micro-component-ladder",
    "qa-neutral-tone-steps",
    "qa-ring-holes",
    "qa-thin-negative-space",
    "qa-small-details",
)
REGRESSION_CASES = ("qa-monoline", "qa-shared-boundary")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("case_id")): item
        for item in (report.get("cases") or [])
        if isinstance(item, dict) and item.get("case_id")
    }


def _candidate_dominates(candidate: dict[str, Any], best: dict[str, Any]) -> bool:
    left = candidate.get("score_details") or {}
    right = best.get("score_details") or {}
    values = (
        _finite(candidate.get("fidelity_score")),
        _finite(candidate.get("total_score")),
        _finite(left.get("path_count")),
        _finite(left.get("node_count")),
        _finite(best.get("fidelity_score")),
        _finite(best.get("total_score")),
        _finite(right.get("path_count")),
        _finite(right.get("node_count")),
    )
    if any(value is None for value in values):
        return False
    cf, ct, cp, cn, bf, bt, bp, bn = values
    return bool(cf > bf and ct > bt and cp <= bp and cn <= bn)


def evaluate(output_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    native_path = root / "native-residual-evidence.json"
    root_cause_path = root / "output-quality-root-cause.json"
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    if not native_path.is_file():
        failures.append("native_residual_evidence_missing")
        native: dict[str, Any] = {}
    else:
        native = json.loads(native_path.read_text(encoding="utf-8"))
    if not root_cause_path.is_file():
        failures.append("root_cause_evidence_missing")
        root_cause: dict[str, Any] = {}
    else:
        root_cause = json.loads(root_cause_path.read_text(encoding="utf-8"))

    native_cases = _case_map(native)
    root_cases = _case_map(root_cause)

    def require(case_id: str, code: str, passed: bool, value: Any, target: Any) -> None:
        checks.append(
            {
                "case_id": case_id,
                "code": code,
                "passed": bool(passed),
                "value": value,
                "target": target,
            }
        )
        if not passed:
            failures.append(f"{case_id}:{code}")

    for case_id in AI2_NATIVE_CASES:
        case = native_cases.get(case_id)
        if not case:
            failures.append(f"{case_id}:native_case_missing")
            continue
        residual = case.get("residual") or {}
        values = {
            "visible_error_ratio": _finite(residual.get("visible_error_ratio")),
            "de00_p95": _finite(residual.get("de00_p95")),
            "source_component_recall": _finite(residual.get("source_component_recall")),
            "render_component_precision": _finite(residual.get("render_component_precision")),
            "small_component_recall": _finite(residual.get("small_component_recall")),
            "min_component_iou": _finite(residual.get("min_component_iou")),
            "boundary_p95_px": _finite(residual.get("boundary_p95_px")),
        }
        require(case_id, "deterministic_2of2", case.get("deterministic") is True, case.get("deterministic"), True)
        require(case_id, "visible_residual", values["visible_error_ratio"] is not None and values["visible_error_ratio"] <= 0.01, values["visible_error_ratio"], "<=0.01")
        require(case_id, "de00_p95", values["de00_p95"] is not None and values["de00_p95"] <= 2.3, values["de00_p95"], "<=2.3")
        require(case_id, "source_cc_recall", values["source_component_recall"] == 1.0, values["source_component_recall"], 1.0)
        require(case_id, "render_cc_precision", values["render_component_precision"] == 1.0, values["render_component_precision"], 1.0)
        require(case_id, "small_cc_recall", values["small_component_recall"] == 1.0, values["small_component_recall"], 1.0)
        require(case_id, "min_true_cc_iou", values["min_component_iou"] is not None and values["min_component_iou"] >= 0.95, values["min_component_iou"], ">=0.95")
        require(case_id, "boundary_p95", values["boundary_p95_px"] is not None and values["boundary_p95_px"] <= 0.75, values["boundary_p95_px"], "<=0.75")

    for case_id in REGRESSION_CASES:
        case = native_cases.get(case_id)
        ready = bool((case or {}).get("near_zero", {}).get("ready") is True)
        require(case_id, "regression_near_zero_pass", ready, ready, True)

    # Production winner evidence is checked for every captured case. A transform
    # candidate explicitly rolled back/failed by TransformJournal is not a final
    # selection alternative and therefore cannot make the accepted winner
    # Pareto-dominated. Missing journal eligibility is treated as eligible.
    for case_id, case in root_cases.items():
        for repeat_index, diagnostic in enumerate(case.get("pipeline_diagnostics") or [], start=1):
            best = diagnostic.get("best") or {}
            if not best:
                failures.append(f"{case_id}:repeat{repeat_index}:best_missing")
                continue
            details = best.get("score_details") or {}
            require(
                case_id,
                f"repeat{repeat_index}_raster_in_svg",
                details.get("has_bitmap") is False,
                details.get("has_bitmap"),
                False,
            )
            dominated_by = None
            for candidate in diagnostic.get("candidates") or []:
                if candidate.get("name") == best.get("name"):
                    continue
                if candidate.get("final_eligible") is False:
                    continue
                if _candidate_dominates(candidate, best):
                    dominated_by = candidate.get("name")
                    break
            require(
                case_id,
                f"repeat{repeat_index}_pareto_winner",
                dominated_by is None,
                dominated_by,
                None,
            )

    # Byte-level raster embedding check on the actual downloadable final SVGs.
    for case_id in root_cases:
        svg_path = root / "cases" / case_id / "selected.svg"
        if not svg_path.is_file():
            failures.append(f"{case_id}:selected_svg_missing")
            continue
        text = svg_path.read_text(encoding="utf-8", errors="replace").lower()
        raster_found = "<image" in text or "base64" in text or "data:image" in text
        require(case_id, "selected_svg_raster_embed", not raster_found, raster_found, False)

    return {
        "schema": "vektoryum-issue144-ai2-full-pipeline-acceptance-v1",
        "status": "pass" if not failures else "fail",
        "target_case_count": len(AI2_NATIVE_CASES),
        "regression_case_count": len(REGRESSION_CASES),
        "failure_count": len(failures),
        "failures": sorted(set(failures)),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.output)
    destination = args.report or (args.output / "issue144-ai2-acceptance.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("schema", "status", "failure_count", "failures")}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
