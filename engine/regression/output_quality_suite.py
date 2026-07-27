"""Production-output diagnostic suite for Vektoryum.

This module exercises the unchanged production pipeline with deterministic,
synthetic, binary-free fixtures that target previously observed failure modes:
gray-border residue, shared-boundary seams, counter/hole loss, monoline breaks,
small-detail loss, low-resolution badges and alpha overlap.

It never changes winner selection or production thresholds. It emits immutable
JSON/Markdown evidence plus source/render/difference PNGs and can fail on:
- structural: missing/unsafe/unrenderable/nondeterministic output
- hard: structural failures plus FinalArtifactEvaluator hard quality failures
- none: diagnostic-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.final_artifact_evaluator import FinalArtifactReport, evaluate_final_svg
from app.pipeline_entry import run_pipeline
from app.source_truth import composite_rgba, render_svg_to_rgba

SCHEMA = "vektoryum-output-quality-diagnostic-v1"
REPORT_FILENAME = "output-quality-report.json"
SUMMARY_FILENAME = "output-quality-summary.md"

STRUCTURAL_CODES = frozenset(
    {
        "pipeline_exception",
        "selected_svg_path_missing",
        "selected_svg_file_missing",
        "artifact_sha_mismatch",
        "artifact_nondeterministic",
        "render_failed",
        "xml_parse_failed",
        "root_not_svg",
        "viewbox_missing",
        "viewbox_invalid",
        "embedded_raster",
        "external_reference",
        "unsafe_uri",
        "unsafe_css",
        "event_handler",
        "nonfinite_geometry",
        "byte_changed_during_read",
    }
)


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    category: str
    image_class: str
    source_path: Path
    source_sha256: str
    required_metrics: frozenset[str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _white_composite(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return np.asarray(background.convert("RGB"), dtype=np.uint8).copy()


def _draw_gray_border_counter(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 36, 232, 220), radius=38, fill=(150, 150, 150, 255))
    draw.rounded_rectangle((38, 50, 218, 206), radius=30, fill=(15, 15, 15, 255))
    draw.ellipse((86, 78, 170, 174), fill=(255, 255, 255, 255))
    draw.rectangle((122, 168, 144, 220), fill=(255, 255, 255, 255))
    return image


def _draw_shared_boundary(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 32, 128, 224), fill=(220, 35, 45, 255))
    draw.rectangle((128, 32, 232, 224), fill=(30, 90, 220, 255))
    draw.ellipse((92, 92, 164, 164), fill=(245, 195, 25, 255))
    return image


def _draw_ring_holes(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 116, 116), fill=(18, 18, 18, 255))
    draw.ellipse((46, 46, 90, 90), fill=(255, 255, 255, 255))
    draw.rounded_rectangle((134, 22, 232, 116), radius=20, fill=(18, 18, 18, 255))
    draw.ellipse((158, 44, 208, 94), fill=(255, 255, 255, 255))
    draw.ellipse((44, 142, 212, 230), fill=(18, 18, 18, 255))
    draw.ellipse((78, 162, 178, 212), fill=(255, 255, 255, 255))
    return image


def _draw_monoline(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line((20, 128, 236, 128), fill=(0, 0, 0, 255), width=3)
    draw.line((128, 20, 128, 236), fill=(0, 0, 0, 255), width=3)
    draw.rectangle((48, 48, 208, 208), outline=(0, 0, 0, 255), width=3)
    draw.ellipse((76, 76, 180, 180), outline=(0, 0, 0, 255), width=3)
    draw.arc((92, 92, 164, 164), 20, 310, fill=(0, 0, 0, 255), width=2)
    return image


def _draw_small_details(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 36, 236, 220), radius=24, outline=(20, 20, 20, 255), width=5)
    for index, radius in enumerate((2, 3, 4, 5, 6, 7)):
        x = 48 + index * 30
        y = 88 if index % 2 == 0 else 156
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(20, 20, 20, 255))
    for index in range(6):
        x = 44 + index * 32
        draw.line((x, 112, x + 14, 112), fill=(215, 35, 45, 255), width=2)
    return image


def _draw_transparent_overlap(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 40, 168, 208), fill=(30, 150, 245, 150))
    draw.rounded_rectangle((92, 28, 228, 220), radius=28, fill=(250, 45, 85, 190))
    draw.ellipse((100, 88, 156, 160), fill=(255, 255, 255, 0))
    return image


def _draw_lowres_badge(size: int = 256) -> Image.Image:
    tiny = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
    draw = ImageDraw.Draw(tiny)
    draw.ellipse((2, 2, 29, 29), fill=(25, 25, 25, 255))
    draw.ellipse((7, 7, 24, 24), fill=(235, 235, 235, 255))
    draw.rectangle((14, 5, 17, 27), fill=(190, 25, 35, 255))
    draw.rectangle((5, 14, 27, 17), fill=(190, 25, 35, 255))
    return tiny.resize((size, size), Image.Resampling.NEAREST)


_CASE_BUILDERS = (
    ("qa-gray-border-counter", "gray_border", "clean_logo", _draw_gray_border_counter, frozenset()),
    ("qa-shared-boundary", "shared_boundary", "geometric", _draw_shared_boundary, frozenset()),
    ("qa-ring-holes", "counter_holes", "clean_logo", _draw_ring_holes, frozenset()),
    ("qa-monoline", "monoline", "lineart", _draw_monoline, frozenset()),
    ("qa-small-details", "small_detail", "geometric", _draw_small_details, frozenset()),
    (
        "qa-transparent-overlap",
        "transparent",
        "illustration",
        _draw_transparent_overlap,
        frozenset({"alpha_fidelity"}),
    ),
    ("qa-lowres-badge", "low_resolution", "illustration", _draw_lowres_badge, frozenset()),
)


def generate_cases(root: Path) -> list[DiagnosticCase]:
    root = Path(root)
    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    cases: list[DiagnosticCase] = []
    for case_id, category, image_class, builder, required in _CASE_BUILDERS:
        path = fixtures / f"{case_id}.png"
        builder().save(path, "PNG", optimize=False)
        cases.append(
            DiagnosticCase(
                case_id=case_id,
                category=category,
                image_class=image_class,
                source_path=path,
                source_sha256=sha256_file(path),
                required_metrics=required,
            )
        )
    return cases


def difference_map(source_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.uint8)
    render = np.asarray(render_rgb, dtype=np.uint8)
    if source.shape != render.shape:
        render = cv2.resize(render, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    delta = np.max(np.abs(source.astype(np.int16) - render.astype(np.int16)), axis=2).astype(np.uint8)
    heat = cv2.applyColorMap(delta, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric(report: FinalArtifactReport, group: str, key: str) -> float | int | None:
    value = (report.metrics.get(group) or {}).get(key)
    return value if _finite(value) else None


def _case_severity(hard_codes: Iterable[str], soft_codes: Iterable[str], unmeasured: Iterable[str]) -> str:
    hard = set(hard_codes)
    if hard & STRUCTURAL_CODES:
        return "critical"
    if hard:
        return "high"
    if list(soft_codes):
        return "medium"
    if list(unmeasured):
        return "low"
    return "pass"


def _structural_failures(case_result: dict[str, Any]) -> list[str]:
    failures = set(case_result.get("hard_fail_codes") or []) & STRUCTURAL_CODES
    if case_result.get("pipeline_status") != "ok":
        failures.add("pipeline_exception")
    if not case_result.get("selected_svg_present"):
        failures.add("selected_svg_file_missing")
    if case_result.get("render_status") != "ok":
        failures.add("render_failed")
    if case_result.get("deterministic") is False:
        failures.add("artifact_nondeterministic")
    return sorted(failures)


def _copy_svg(svg_path: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(svg_path, destination)
    return sha256_file(destination)


def run_diagnostic_case(
    case: DiagnosticCase,
    *,
    output_root: Path,
    engine_version: str,
    repeat_count: int,
    trace_mode: str = "auto",
) -> dict[str, Any]:
    case_root = Path(output_root) / "cases" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    source_copy = case_root / "source.png"
    shutil.copyfile(case.source_path, source_copy)

    repeat_shas: list[str] = []
    repeat_durations: list[float] = []
    reports: list[FinalArtifactReport] = []
    render_status = "not_run"
    selected_svg_present = False
    pipeline_status = "ok"
    pipeline_error: str | None = None
    canonical_svg = case_root / "selected.svg"

    with Image.open(case.source_path) as opened:
        source_rgba_image = opened.convert("RGBA")
        source_rgba = np.asarray(source_rgba_image, dtype=np.uint8).copy()
        source_rgb = _white_composite(source_rgba_image)
        source_alpha = source_rgba[:, :, 3]

        for repeat_index in range(1, repeat_count + 1):
            job_dir = case_root / "jobs" / f"repeat-{repeat_index}"
            started = time.perf_counter()
            try:
                output = run_pipeline(
                    source_rgba_image.copy(),
                    case.source_path,
                    trace_mode,
                    job_dir,
                )
            except BaseException as exc:  # fail-closed evidence
                pipeline_status = "failed"
                pipeline_error = f"{type(exc).__name__}: {exc}"
                break
            repeat_durations.append(round((time.perf_counter() - started) * 1000.0, 6))
            best = output.get("best") or {}
            raw_path = best.get("svg_path")
            if not raw_path:
                pipeline_status = "failed"
                pipeline_error = "selected_svg_path_missing"
                break
            svg_path = Path(raw_path)
            if not svg_path.is_file():
                pipeline_status = "failed"
                pipeline_error = "selected_svg_file_missing"
                break
            selected_svg_present = True
            artifact_sha = sha256_file(svg_path)
            repeat_shas.append(artifact_sha)
            if repeat_index == 1:
                copied_sha = _copy_svg(svg_path, canonical_svg)
                if copied_sha != artifact_sha:
                    pipeline_status = "failed"
                    pipeline_error = "artifact_sha_mismatch"
                    break
            report = evaluate_final_svg(
                svg_path,
                source_rgb,
                source_alpha=source_alpha,
                image_class=case.image_class,
                required_metrics=set(case.required_metrics),
            )
            reports.append(report)

        deterministic = len(set(repeat_shas)) <= 1 if repeat_shas else None
        if selected_svg_present and canonical_svg.is_file():
            rendered = render_svg_to_rgba(canonical_svg, source_rgba_image.width, source_rgba_image.height)
            if rendered is None:
                render_status = "failed"
            else:
                render_status = "ok"
                rendered_rgba = np.asarray(rendered, dtype=np.uint8)
                Image.fromarray(rendered_rgba, mode="RGBA").save(case_root / "render.png")
                rendered_rgb = composite_rgba(rendered_rgba, 255)
                Image.fromarray(difference_map(source_rgb, rendered_rgb), mode="RGB").save(
                    case_root / "difference.png"
                )

    primary = reports[0] if reports else None
    hard_codes = list(primary.hard_fail_codes) if primary else []
    soft_codes = list(primary.soft_warning_codes) if primary else []
    unmeasured = list(primary.unmeasured_required) if primary else []
    if pipeline_error and pipeline_error in STRUCTURAL_CODES and pipeline_error not in hard_codes:
        hard_codes.append(pipeline_error)
    if deterministic is False and "artifact_nondeterministic" not in hard_codes:
        hard_codes.append("artifact_nondeterministic")
    if render_status == "failed" and "render_failed" not in hard_codes:
        hard_codes.append("render_failed")

    result: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "image_class": case.image_class,
        "source_sha256": case.source_sha256,
        "engine_version": engine_version,
        "pipeline_status": pipeline_status,
        "pipeline_error": pipeline_error,
        "selected_svg_present": selected_svg_present,
        "selected_svg_sha256": repeat_shas[0] if repeat_shas else None,
        "repeat_count_requested": repeat_count,
        "repeat_count_completed": len(repeat_shas),
        "repeat_artifact_sha256": repeat_shas,
        "repeat_render_ms": repeat_durations,
        "deterministic": deterministic,
        "render_status": render_status,
        "evaluator_verdict": primary.verdict if primary else "not_run",
        "hard_fail_codes": sorted(set(hard_codes)),
        "soft_warning_codes": sorted(set(soft_codes)),
        "unmeasured_required": sorted(set(unmeasured)),
        "metrics": {
            "ssim": _metric(primary, "B_visual", "ssim") if primary else None,
            "ms_ssim": _metric(primary, "B_visual", "ms_ssim") if primary else None,
            "edge_f1_1px": _metric(primary, "D_edge_geometry", "edge_f1_1px") if primary else None,
            "edge_f1_2px": _metric(primary, "D_edge_geometry", "edge_f1_2px") if primary else None,
            "chamfer_p95": _metric(primary, "D_edge_geometry", "chamfer_p95") if primary else None,
            "hausdorff_max": _metric(primary, "D_edge_geometry", "hausdorff_max") if primary else None,
            "de00_mean": _metric(primary, "C_color", "de00_mean") if primary else None,
            "de00_p95": _metric(primary, "C_color", "de00_p95") if primary else None,
            "palette_agree": _metric(primary, "C_color", "palette_agree") if primary else None,
            "component_delta": _metric(primary, "E_topology", "component_delta") if primary else None,
            "hole_delta": _metric(primary, "E_topology", "hole_delta") if primary else None,
            "min_component_iou": _metric(primary, "F_small_detail", "min_component_iou") if primary else None,
            "mean_component_iou": _metric(primary, "F_small_detail", "mean_component_iou") if primary else None,
            "seam_ratio": _metric(primary, "G_gradient_alpha", "seam_ratio") if primary else None,
            "alpha_iou": _metric(primary, "G_gradient_alpha", "alpha_iou") if primary else None,
            "alpha_mae": _metric(primary, "G_gradient_alpha", "alpha_mae") if primary else None,
            "path_count": _metric(primary, "H_editability", "path_count") if primary else None,
            "node_count": _metric(primary, "H_editability", "node_count") if primary else None,
            "nodes_per_path": _metric(primary, "H_editability", "nodes_per_path") if primary else None,
            "svg_bytes": canonical_svg.stat().st_size if canonical_svg.is_file() else None,
            "render_ms_median": (
                float(np.median(np.asarray(repeat_durations, dtype=np.float64)))
                if repeat_durations
                else None
            ),
        },
        "artifacts": {
            "source": f"cases/{case.case_id}/source.png",
            "selected_svg": f"cases/{case.case_id}/selected.svg" if canonical_svg.is_file() else None,
            "render": f"cases/{case.case_id}/render.png" if render_status == "ok" else None,
            "difference": f"cases/{case.case_id}/difference.png" if render_status == "ok" else None,
        },
    }
    result["structural_failures"] = _structural_failures(result)
    result["severity"] = _case_severity(
        result["hard_fail_codes"],
        result["soft_warning_codes"],
        result["unmeasured_required"],
    )
    return result


def build_report(
    case_results: list[dict[str, Any]],
    *,
    engine_version: str,
    repeat_count: int,
    fail_on: str,
) -> dict[str, Any]:
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "pass": 4}
    ordered = sorted(
        case_results,
        key=lambda item: (severity_order.get(str(item.get("severity")), 99), str(item.get("case_id"))),
    )
    counts = {name: 0 for name in severity_order}
    for item in ordered:
        counts[str(item["severity"])] += 1
    structural = sorted(
        {
            code
            for item in ordered
            for code in (item.get("structural_failures") or [])
        }
    )
    hard = sorted(
        {
            code
            for item in ordered
            for code in (item.get("hard_fail_codes") or [])
        }
    )
    return {
        "schema": SCHEMA,
        "engine_version": engine_version,
        "case_count": len(ordered),
        "repeat_count": repeat_count,
        "fail_on": fail_on,
        "severity_counts": counts,
        "structural_failure_codes": structural,
        "hard_failure_codes": hard,
        "all_outputs_present": all(item.get("selected_svg_present") for item in ordered),
        "all_deterministic": all(item.get("deterministic") is True for item in ordered),
        "cases": ordered,
    }


def should_fail(report: dict[str, Any], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    if fail_on == "structural":
        return bool(report.get("structural_failure_codes"))
    if fail_on == "hard":
        return bool(report.get("hard_failure_codes"))
    raise ValueError(f"unsupported fail_on mode: {fail_on}")


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = report["severity_counts"]
    rows = []
    for item in report["cases"]:
        metrics = item["metrics"]
        rows.append(
            "| {case} | {severity} | {verdict} | {ssim} | {edge} | {de} | {seam} | {holes} | {hard} |".format(
                case=item["case_id"],
                severity=item["severity"],
                verdict=item["evaluator_verdict"],
                ssim="—" if metrics["ssim"] is None else f"{metrics['ssim']:.4f}",
                edge="—" if metrics["edge_f1_1px"] is None else f"{metrics['edge_f1_1px']:.4f}",
                de="—" if metrics["de00_p95"] is None else f"{metrics['de00_p95']:.2f}",
                seam="—" if metrics["seam_ratio"] is None else f"{metrics['seam_ratio']:.5f}",
                holes="—" if metrics["hole_delta"] is None else str(metrics["hole_delta"]),
                hard=", ".join(item["hard_fail_codes"]) or "—",
            )
        )
    summary = "\n".join(
        [
            "# Vektoryum output-quality diagnostic",
            "",
            f"- Engine: `{report['engine_version']}`",
            f"- Cases: **{report['case_count']}**",
            f"- Repeats per case: **{report['repeat_count']}**",
            f"- Gate mode: **{report['fail_on']}**",
            f"- Critical / High / Medium / Low / Pass: **{counts['critical']} / {counts['high']} / {counts['medium']} / {counts['low']} / {counts['pass']}**",
            f"- Structural codes: `{', '.join(report['structural_failure_codes']) or 'none'}`",
            f"- Hard quality codes: `{', '.join(report['hard_failure_codes']) or 'none'}`",
            "",
            "| Case | Severity | Evaluator | SSIM | Edge F1 | ΔE00 p95 | Seam | Hole Δ | Hard codes |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "Each case directory contains `source.png`, `selected.svg`, `render.png` and `difference.png` when rendering succeeds.",
            "",
        ]
    )
    (output_root / SUMMARY_FILENAME).write_text(summary, encoding="utf-8")


def run_suite(
    output_root: Path,
    *,
    engine_version: str,
    repeat_count: int,
    fail_on: str,
    case_limit: int | None = None,
) -> dict[str, Any]:
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    if case_limit is not None and case_limit < 1:
        raise ValueError("case_limit must be positive when provided")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(output_root / "generated")
    if case_limit is not None:
        cases = cases[:case_limit]
    results = [
        run_diagnostic_case(
            case,
            output_root=output_root,
            engine_version=engine_version,
            repeat_count=repeat_count,
        )
        for case in cases
    ]
    report = build_report(
        results,
        engine_version=engine_version,
        repeat_count=repeat_count,
        fail_on=fail_on,
    )
    write_report(output_root, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--fail-on", choices=("none", "structural", "hard"), default="structural")
    args = parser.parse_args()
    report = run_suite(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        fail_on=args.fail_on,
        case_limit=args.case_limit,
    )
    print(
        json.dumps(
            {
                "status": "failed" if should_fail(report, args.fail_on) else "ok",
                "schema": report["schema"],
                "case_count": report["case_count"],
                "severity_counts": report["severity_counts"],
                "structural_failure_codes": report["structural_failure_codes"],
                "hard_failure_codes": report["hard_failure_codes"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1 if should_fail(report, args.fail_on) else 0)


if __name__ == "__main__":
    main()
