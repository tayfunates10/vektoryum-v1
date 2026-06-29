from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.analyzer import analyze_image
from app.main import (
    basic_svg_quality_check,
    convert_svg_to_dxf,
    multi_candidate_vectorize,
    render_svg_to_png_for_compare,
)


ENGINE_DIR = Path(__file__).resolve().parent
REGRESSION_DIR = ENGINE_DIR / "regression"
DEFAULT_MANIFEST = REGRESSION_DIR / "manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_regression_path(value: str | None) -> Path | None:
    if not value:
        return None

    path = Path(value)

    if path.is_absolute():
        return path

    return REGRESSION_DIR / path


def _select_mode_and_quality(trace_mode: str, quality: str, analysis: dict[str, Any]) -> tuple[str, str, str | None]:
    selected_mode = trace_mode
    selected_quality = quality
    mode_warning = None

    if trace_mode == "auto":
        selected_mode = analysis["recommended_mode"]

        if selected_mode in {"geometric_logo", "minimal_ai", "logo_color", "photo_poster"}:
            selected_quality = "detailed"

    if (
        trace_mode == "auto"
        and selected_mode == "minimal_ai"
        and int(analysis.get("estimated_color_count", 0)) > 12
    ):
        selected_mode = "logo_color"
        selected_quality = "detailed"
        mode_warning = "auto minimal_ai overridden to logo_color because color count is high"

    return selected_mode, selected_quality, mode_warning


def _render_size(width: int, height: int, max_side: int = 1100) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return max(1, width), max(1, height)

    scale = max_side / max(width, height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _image_diff(actual_path: Path, baseline_path: Path) -> dict[str, float]:
    actual = Image.open(actual_path).convert("RGBA")
    baseline = Image.open(baseline_path).convert("RGBA")

    if actual.size != baseline.size:
        baseline = baseline.resize(actual.size, Image.Resampling.LANCZOS)

    actual_arr = np.array(actual).astype(np.float32)
    baseline_arr = np.array(baseline).astype(np.float32)
    diff = actual_arr - baseline_arr
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    changed_pixel_ratio = float(np.mean(np.any(abs_diff[:, :, :3] > 18.0, axis=2)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "changed_pixel_ratio": round(changed_pixel_ratio, 6),
    }


def _record_check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _validate_expected(
    case: dict[str, Any],
    analysis: dict[str, Any],
    mode_used: str,
    candidate_report: dict[str, Any],
    quality_report: dict[str, Any],
    output_errors: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    expected = case.get("expected", {})

    if "mode_used" in expected:
        _record_check(
            errors,
            mode_used == expected["mode_used"],
            f"mode_used expected {expected['mode_used']}, got {mode_used}",
        )

    if "detected_type" in expected:
        _record_check(
            errors,
            analysis.get("detected_type") == expected["detected_type"],
            f"detected_type expected {expected['detected_type']}, got {analysis.get('detected_type')}",
        )

    if "recommended_mode" in expected:
        _record_check(
            errors,
            analysis.get("recommended_mode") == expected["recommended_mode"],
            f"recommended_mode expected {expected['recommended_mode']}, got {analysis.get('recommended_mode')}",
        )

    if "likely_geometric_logo" in expected:
        _record_check(
            errors,
            bool(analysis.get("likely_geometric_logo")) is bool(expected["likely_geometric_logo"]),
            f"likely_geometric_logo expected {expected['likely_geometric_logo']}, got {analysis.get('likely_geometric_logo')}",
        )

    candidates = candidate_report.get("candidates", [])
    successful_candidates = [item for item in candidates if item.get("success", True)]

    if "candidate_min" in expected:
        _record_check(
            errors,
            len(candidates) >= int(expected["candidate_min"]),
            f"candidate count expected >= {expected['candidate_min']}, got {len(candidates)}",
        )

    if "successful_candidate_min" in expected:
        _record_check(
            errors,
            len(successful_candidates) >= int(expected["successful_candidate_min"]),
            f"successful candidate count expected >= {expected['successful_candidate_min']}, got {len(successful_candidates)}",
        )

    if "best_candidate_in" in expected:
        allowed_best = set(expected["best_candidate_in"])
        best_candidate = candidate_report.get("best_candidate")
        _record_check(
            errors,
            best_candidate in allowed_best,
            f"best_candidate expected one of {sorted(allowed_best)}, got {best_candidate}",
        )

    if "status_in" in expected:
        allowed_status = set(expected["status_in"])
        status = quality_report.get("status")
        _record_check(
            errors,
            status in allowed_status,
            f"quality status expected one of {sorted(allowed_status)}, got {status}",
        )

    if "max_warning_count" in expected:
        warnings = quality_report.get("warnings", [])
        _record_check(
            errors,
            len(warnings) <= int(expected["max_warning_count"]),
            f"warning count expected <= {expected['max_warning_count']}, got {len(warnings)}",
        )

    if not bool(expected.get("dxf_error_allowed", True)):
        _record_check(
            errors,
            "dxf" not in output_errors,
            f"dxf export error was not allowed: {output_errors.get('dxf')}",
        )

    quality_expected = expected.get("quality", {})
    path_count = int(quality_report.get("path_count", 0))
    unique_color_count = int(quality_report.get("unique_color_count", 0))
    geometry_report = quality_report.get("geometry_report") or {}
    geometry_score = float(geometry_report.get("geometry_score", 0.0))

    if "min_path_count" in quality_expected:
        _record_check(errors, path_count >= int(quality_expected["min_path_count"]), f"path_count expected >= {quality_expected['min_path_count']}, got {path_count}")

    if "max_path_count" in quality_expected:
        _record_check(errors, path_count <= int(quality_expected["max_path_count"]), f"path_count expected <= {quality_expected['max_path_count']}, got {path_count}")

    if "min_unique_color_count" in quality_expected:
        _record_check(errors, unique_color_count >= int(quality_expected["min_unique_color_count"]), f"unique_color_count expected >= {quality_expected['min_unique_color_count']}, got {unique_color_count}")

    if "max_unique_color_count" in quality_expected:
        _record_check(errors, unique_color_count <= int(quality_expected["max_unique_color_count"]), f"unique_color_count expected <= {quality_expected['max_unique_color_count']}, got {unique_color_count}")

    if "min_geometry_score" in quality_expected:
        _record_check(errors, geometry_score >= float(quality_expected["min_geometry_score"]), f"geometry_score expected >= {quality_expected['min_geometry_score']}, got {geometry_score}")

    return errors


def _validate_visual(
    case: dict[str, Any],
    render_path: Path | None,
    update_baseline: bool,
    require_baseline: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    visual = case.get("visual") or {}
    baseline_path = _resolve_regression_path(visual.get("baseline"))
    errors: list[str] = []
    warnings: list[str] = []

    if not baseline_path:
        return None, errors, warnings

    if render_path is None or not render_path.exists():
        warnings.append("render unavailable; baseline image comparison skipped")
        return None, errors, warnings

    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    if update_baseline:
        shutil.copyfile(render_path, baseline_path)
        return {"updated": True, "baseline": str(baseline_path)}, errors, warnings

    if not baseline_path.exists():
        message = f"baseline missing: {baseline_path}"
        if require_baseline:
            errors.append(message)
        else:
            warnings.append(message)
        return None, errors, warnings

    diff = _image_diff(render_path, baseline_path)

    for key in ("max_mae", "max_rmse", "max_changed_pixel_ratio"):
        if key not in visual:
            continue

        metric_name = key.replace("max_", "")
        actual = float(diff[metric_name])
        limit = float(visual[key])

        if actual > limit:
            errors.append(f"{metric_name} expected <= {limit}, got {actual}")

    diff["baseline"] = str(baseline_path)
    return diff, errors, warnings


def run_case(
    case: dict[str, Any],
    run_dir: Path,
    update_baseline: bool,
    require_baseline: bool,
) -> dict[str, Any]:
    case_id = str(case["id"])
    input_path = _resolve_regression_path(case.get("input"))

    if input_path is None:
        raise ValueError(f"{case_id}: input is not configured")

    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    final_svg_path = case_dir / f"{case_id}.svg"
    dxf_path = case_dir / f"{case_id}.dxf"
    render_path = case_dir / f"{case_id}.png"
    candidate_dir = case_dir / "candidates"

    analysis = analyze_image(input_path)
    mode_used, quality_used, mode_warning = _select_mode_and_quality(
        trace_mode=str(case.get("trace_mode", "auto")),
        quality=str(case.get("quality", "detailed")),
        analysis=analysis,
    )

    candidate_report = multi_candidate_vectorize(
        input_path=input_path,
        final_svg_path=final_svg_path,
        temp_dir=candidate_dir,
        selected_trace_mode=mode_used,
        selected_quality=quality_used,
        analysis_report=analysis,
    )

    quality_report = basic_svg_quality_check(final_svg_path, mode_used)
    output_errors: dict[str, str] = {}

    try:
        convert_svg_to_dxf(final_svg_path, dxf_path, mode_used)
    except Exception as exc:
        output_errors["dxf"] = str(exc) or exc.__class__.__name__

    render_width, render_height = _render_size(
        int(analysis.get("width", 1024)),
        int(analysis.get("height", 1024)),
    )
    rendered_ok = render_svg_to_png_for_compare(
        svg_path=final_svg_path,
        png_output_path=render_path,
        width=render_width,
        height=render_height,
    )

    if not rendered_ok and render_path.exists():
        render_path.unlink()

    errors = _validate_expected(
        case=case,
        analysis=analysis,
        mode_used=mode_used,
        candidate_report=candidate_report,
        quality_report=quality_report,
        output_errors=output_errors,
    )

    visual_report, visual_errors, visual_warnings = _validate_visual(
        case=case,
        render_path=render_path if rendered_ok else None,
        update_baseline=update_baseline,
        require_baseline=require_baseline,
    )
    errors.extend(visual_errors)

    result = {
        "id": case_id,
        "success": not errors,
        "errors": errors,
        "warnings": visual_warnings,
        "input": str(input_path),
        "mode_used": mode_used,
        "quality_used": quality_used,
        "mode_warning": mode_warning,
        "analysis": analysis,
        "quality_report": quality_report,
        "candidate_report": candidate_report,
        "output_errors": output_errors,
        "rendered_ok": rendered_ok,
        "visual_diff": visual_report,
        "artifacts": {
            "svg": str(final_svg_path),
            "dxf": str(dxf_path) if dxf_path.exists() else None,
            "render": str(render_path) if render_path.exists() else None,
            "case_dir": str(case_dir),
        },
    }

    _write_json(case_dir / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-image visual regression checks.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to regression manifest JSON.")
    parser.add_argument("--case", dest="case_ids", action="append", help="Run only the selected case id. Can be passed multiple times.")
    parser.add_argument("--update-baseline", action="store_true", help="Write rendered PNG outputs as baselines.")
    parser.add_argument("--require-baseline", action="store_true", help="Fail when a configured baseline image is missing.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip cases whose fixture image is missing.")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports and artifacts.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    if not manifest_path.is_absolute():
        manifest_path = ENGINE_DIR / manifest_path

    manifest = _read_json(manifest_path)
    cases = manifest.get("cases", [])

    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in selected]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) if args.out_dir else REGRESSION_DIR / "results" / timestamp

    if not run_dir.is_absolute():
        run_dir = ENGINE_DIR / run_dir

    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    missing = []

    for case in cases:
        input_path = _resolve_regression_path(case.get("input"))

        if input_path is None or not input_path.exists():
            message = f"{case.get('id')}: missing fixture {input_path}"
            missing.append(message)

            if args.allow_missing:
                print(f"SKIP {message}")
                continue

            print(f"FAIL {message}")
            results.append({"id": case.get("id"), "success": False, "errors": [message]})
            continue

        try:
            result = run_case(
                case=case,
                run_dir=run_dir,
                update_baseline=bool(args.update_baseline),
                require_baseline=bool(args.require_baseline),
            )
            results.append(result)
            status = "PASS" if result["success"] else "FAIL"
            print(f"{status} {result['id']} mode={result['mode_used']} best={result['candidate_report'].get('best_candidate')}")

            for warning in result.get("warnings", []):
                print(f"  WARN {warning}")

            for error in result.get("errors", []):
                print(f"  ERROR {error}")

        except Exception as exc:
            result = {
                "id": case.get("id"),
                "success": False,
                "errors": [str(exc) or exc.__class__.__name__],
            }
            results.append(result)
            print(f"FAIL {case.get('id')} {result['errors'][0]}")

    summary = {
        "success": all(item.get("success") for item in results) and (bool(results) or bool(args.allow_missing)),
        "run_dir": str(run_dir),
        "missing": missing,
        "results": [
            {
                "id": item.get("id"),
                "success": item.get("success"),
                "errors": item.get("errors", []),
                "mode_used": item.get("mode_used"),
                "best_candidate": (item.get("candidate_report") or {}).get("best_candidate"),
            }
            for item in results
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    print(f"Report: {run_dir / 'summary.json'}")

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
