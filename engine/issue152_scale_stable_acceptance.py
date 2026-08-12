"""Fail-closed five-scale acceptance for Issue #152 (AI-2 Round 3).

This gate consumes the real 12-case production artifacts.  Diagnostic success
is not acceptance: every target must satisfy every declared fidelity/component/
topology axis at 64/128/256/512/1024, with 2/2 deterministic final SVG bytes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TARGET_CASES = (
    "qa-micro-component-ladder",
    "qa-small-details",
    "qa-thin-negative-space",
)
REGRESSION_CASES = (
    "qa-gray-border-counter",
    "qa-lowres-badge",
    "qa-neutral-tone-steps",
    "qa-ring-holes",
    "qa-monoline",
    "qa-shared-boundary",
    "qa-soft-alpha-shadow",
    "qa-transparent-overlap",
)
EXPECTED_LEVELS = {
    "0.25x": 64,
    "0.5x": 128,
    "1x": 256,
    "2x": 512,
    "4x": 1024,
}


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


def _topology_loss(topology: dict[str, Any] | None) -> int | None:
    if not isinstance(topology, dict) or topology.get("complete") is not True:
        return None
    values = (
        topology.get("hole_loss_count"),
        topology.get("ring_break_count"),
        topology.get("adjacency_loss_count"),
    )
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    return int(sum(int(value) for value in values))


def evaluate(output_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    paths = {
        "native": root / "native-residual-evidence.json",
        "scale": root / "multiscale-topology-evidence.json",
        "root": root / "output-quality-root-cause.json",
    }
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"{name}_evidence_missing")
            payloads[name] = {}
        else:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))

    native_cases = _case_map(payloads["native"])
    scale_cases = _case_map(payloads["scale"])
    root_cases = _case_map(payloads["root"])

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

    target_summary: dict[str, Any] = {}
    for case_id in TARGET_CASES:
        native = native_cases.get(case_id) or {}
        scale_case = scale_cases.get(case_id) or {}
        root_case = root_cases.get(case_id) or {}
        multi = scale_case.get("multi_scale") or {}
        levels = {
            str(level.get("scale")): level
            for level in (multi.get("levels") or [])
            if isinstance(level, dict) and level.get("scale")
        }

        require(case_id, "native_deterministic_2of2", native.get("deterministic") is True, native.get("deterministic"), True)
        require(case_id, "all_five_scales_measured", multi.get("all_scales_measured") is True and set(levels) == set(EXPECTED_LEVELS), sorted(levels), sorted(EXPECTED_LEVELS))
        require(case_id, "source_contract_verified", multi.get("source_contract_verified") is True, multi.get("source_contract_verified"), True)
        require(case_id, "source_scale_policy", multi.get("source_scale_policy") == "normalized_analytic_geometry_v1", multi.get("source_scale_policy"), "normalized_analytic_geometry_v1")

        repeat_shas: list[str] = []
        winners: list[dict[str, Any]] = []
        diagnostics = root_case.get("pipeline_diagnostics") or []
        for repeat_index, diagnostic in enumerate(diagnostics, start=1):
            best = (diagnostic or {}).get("best") or {}
            sha = best.get("svg_sha256")
            if isinstance(sha, str) and sha:
                repeat_shas.append(sha)
            winners.append(
                {
                    "repeat": repeat_index,
                    "name": best.get("name"),
                    "engine": best.get("engine"),
                    "svg_sha256": sha,
                    "fidelity_score": best.get("fidelity_score"),
                    "total_score": best.get("total_score"),
                    "path_count": (best.get("score_details") or {}).get("path_count"),
                    "node_count": (best.get("score_details") or {}).get("node_count"),
                    "has_bitmap": (best.get("score_details") or {}).get("has_bitmap"),
                }
            )
        deterministic_sha = len(repeat_shas) >= 2 and len(set(repeat_shas)) == 1
        require(case_id, "same_final_svg_sha_2of2", deterministic_sha, repeat_shas, "same non-empty SHA twice")

        level_summary: dict[str, Any] = {}
        for label, expected_size in EXPECTED_LEVELS.items():
            level = levels.get(label)
            if level is None:
                failures.append(f"{case_id}:{label}:level_missing")
                continue
            size = int(level.get("size") or 0)
            require(case_id, f"{label}_size", size == expected_size, size, expected_size)

            source_recall = _finite(level.get("source_component_recall"))
            render_precision = _finite(level.get("render_component_precision"))
            small_recall = _finite(level.get("small_component_recall"))
            min_iou = _finite(level.get("min_component_iou"))
            boundary_excess = _finite(level.get("normalized_boundary_excess_px"))
            visible = _finite(level.get("visible_error_ratio"))
            de00 = _finite(level.get("de00_p95"))
            topology_loss = _topology_loss(level.get("topology"))

            require(case_id, f"{label}_source_cc_recall", source_recall == 1.0, source_recall, 1.0)
            require(case_id, f"{label}_render_cc_precision", render_precision == 1.0, render_precision, 1.0)
            require(case_id, f"{label}_small_component_recall", small_recall == 1.0, small_recall, 1.0)
            require(case_id, f"{label}_min_true_component_iou", min_iou is not None and min_iou >= 0.95, min_iou, ">=0.95")
            require(case_id, f"{label}_normalized_boundary_excess", boundary_excess is not None and boundary_excess <= 0.25, boundary_excess, "<=0.25px")
            require(case_id, f"{label}_visible_residual", visible is not None and visible <= 0.01, visible, "<=0.01")
            require(case_id, f"{label}_de00_p95", de00 is not None and de00 <= 2.3, de00, "<=2.3")
            require(case_id, f"{label}_topology_loss", topology_loss == 0, topology_loss, 0)

            level_summary[label] = {
                "size": size,
                "source_cc_recall": source_recall,
                "render_cc_precision": render_precision,
                "small_component_recall": small_recall,
                "min_true_component_iou": min_iou,
                "normalized_boundary_excess_px": boundary_excess,
                "visible_residual": visible,
                "de00_p95": de00,
                "topology_loss": topology_loss,
            }

        selected = root / "cases" / case_id / "selected.svg"
        raster_found = None
        if selected.is_file():
            text = selected.read_text(encoding="utf-8", errors="replace").lower()
            raster_found = "<image" in text or "base64" in text or "data:image" in text
        require(case_id, "selected_svg_exists", selected.is_file(), selected.is_file(), True)
        require(case_id, "selected_svg_raster_embed", raster_found is False, raster_found, False)

        target_summary[case_id] = {
            "levels": level_summary,
            "repeat_final_svg_sha256": repeat_shas,
            "winners": winners,
            "raster_embedded": raster_found,
        }

    for case_id in REGRESSION_CASES:
        native = native_cases.get(case_id) or {}
        ready = bool((native.get("near_zero") or {}).get("ready") is True)
        require(case_id, "round2_regression_native_pass", ready, ready, True)

    return {
        "schema": "vektoryum-issue152-ai2-five-scale-acceptance-v1",
        "status": "pass" if not failures else "fail",
        "target_cases": list(TARGET_CASES),
        "required_sizes": list(EXPECTED_LEVELS.values()),
        "regression_cases": list(REGRESSION_CASES),
        "failure_count": len(sorted(set(failures))),
        "failures": sorted(set(failures)),
        "checks": checks,
        "targets": target_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.output)
    destination = args.report or (args.output / "issue152-ai2-five-scale-acceptance.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("schema", "status", "failure_count", "failures")}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
