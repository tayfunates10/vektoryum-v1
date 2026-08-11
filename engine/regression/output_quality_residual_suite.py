"""Near-zero residual extension for Vektoryum's production output diagnostic.

The existing output-quality suite remains the owner of production execution,
FinalArtifactEvaluator evidence and its release thresholds. This module adds five
targeted deterministic fixtures and a diagnostic-only, fail-closed residual
contract without changing winner selection or production policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from engine.regression import output_quality_suite as base_suite
from engine.regression.residual_error_metrics import (
    SCALES,
    blocker_histogram,
    build_near_zero_contract,
    measure_multiscale_svg,
    measure_residual_error,
)

RESIDUAL_SCHEMA = "vektoryum-output-quality-residual-v1"
REPORT_FILENAME = base_suite.REPORT_FILENAME
SUMMARY_FILENAME = base_suite.SUMMARY_FILENAME
DiagnosticCase = base_suite.DiagnosticCase

# Root-cause wrapping temporarily replaces this symbol. run_diagnostic_case then
# forwards it into the unchanged base suite so pipeline decision capture still
# works for all twelve cases.
run_pipeline = base_suite.run_pipeline


def _draw_micro_component_ladder(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    scale = size / 256.0
    draw.rounded_rectangle(
        (
            round(24 * scale),
            round(34 * scale),
            round(232 * scale),
            round(150 * scale),
        ),
        radius=max(1, round(18 * scale)),
        fill=(24, 24, 24, 255),
    )
    y = round(202 * scale)
    for index, nominal_side in enumerate(range(1, 9)):
        side = max(1, round(nominal_side * scale))
        x = round((34 + index * 26) * scale)
        draw.rectangle((x, y, x + side - 1, y + side - 1), fill=(24, 24, 24, 255))
    return image


def _draw_thin_negative_space(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    scale = size / 256.0
    draw.rounded_rectangle(
        (
            round(20 * scale),
            round(28 * scale),
            round(236 * scale),
            round(228 * scale),
        ),
        radius=max(1, round(22 * scale)),
        fill=(18, 18, 18, 255),
    )
    centers = (62, 106, 150, 194)
    for nominal_width, center in enumerate(centers, start=1):
        width = max(1, round(nominal_width * scale))
        cx = round(center * scale)
        x0 = cx - width // 2
        x1 = x0 + width - 1
        draw.rectangle(
            (x0, round(58 * scale), x1, round(198 * scale)),
            fill=(255, 255, 255, 255),
        )
    # Closed counters ensure the white slits are true negative-space details,
    # not merely background-connected cuts.
    draw.ellipse(
        (
            round(86 * scale),
            round(90 * scale),
            round(170 * scale),
            round(174 * scale),
        ),
        fill=(255, 255, 255, 255),
    )
    draw.ellipse(
        (
            round(91 * scale),
            round(95 * scale),
            round(165 * scale),
            round(169 * scale),
        ),
        fill=(18, 18, 18, 255),
    )
    return image


def _draw_hard_stop_gradient(size: int = 256) -> Image.Image:
    rgba = np.full((size, size, 4), 255, dtype=np.uint8)
    y0 = max(1, round(size * 48 / 256))
    y1 = min(size, round(size * 208 / 256))
    x0 = max(1, round(size * 24 / 256))
    stop = min(size - 1, round(size * 128 / 256))
    x1 = min(size, round(size * 232 / 256))
    span = max(1, stop - x0)
    start = np.asarray((225, 40, 35), dtype=np.float64)
    end = np.asarray((245, 188, 35), dtype=np.float64)
    for x in range(x0, stop):
        t = 0.0 if span <= 1 else (x - x0) / float(span - 1)
        rgba[y0:y1, x, :3] = np.rint(start * (1.0 - t) + end * t).astype(np.uint8)
    rgba[y0:y1, stop:x1, :3] = np.asarray((35, 92, 220), dtype=np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _draw_soft_alpha_shadow(size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = size * 0.50
    cy = size * 0.66
    rx = max(1.0, size * 0.36)
    ry = max(1.0, size * 0.18)
    distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    shadow_alpha = np.rint(np.clip(1.0 - distance, 0.0, 1.0) * 105.0).astype(np.uint8)
    shadow = np.zeros((size, size, 4), dtype=np.uint8)
    shadow[:, :, :3] = 28
    shadow[:, :, 3] = shadow_alpha
    image = Image.fromarray(shadow, mode="RGBA")

    blue = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(blue)
    draw.ellipse(
        (
            round(size * 24 / 256),
            round(size * 48 / 256),
            round(size * 168 / 256),
            round(size * 208 / 256),
        ),
        fill=(28, 148, 242, 172),
    )
    image = Image.alpha_composite(image, blue)

    red = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(red)
    draw.rounded_rectangle(
        (
            round(size * 92 / 256),
            round(size * 34 / 256),
            round(size * 228 / 256),
            round(size * 214 / 256),
        ),
        radius=max(1, round(size * 28 / 256)),
        fill=(248, 48, 82, 198),
    )
    image = Image.alpha_composite(image, red)
    return image


def _draw_neutral_tone_steps(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    scale = size / 256.0
    layers = (
        ((18, 24, 238, 232), 244),
        ((34, 40, 222, 216), 230),
        ((52, 58, 204, 198), 208),
        ((76, 82, 180, 174), 142),
        ((104, 108, 152, 148), 24),
    )
    for box, tone in layers:
        draw.rounded_rectangle(
            tuple(round(value * scale) for value in box),
            radius=max(1, round(12 * scale)),
            fill=(tone, tone, tone, 255),
        )
    return image


_EXTRA_CASE_BUILDERS = (
    (
        "qa-micro-component-ladder",
        "micro_component",
        "clean_logo",
        _draw_micro_component_ladder,
        frozenset(),
    ),
    (
        "qa-thin-negative-space",
        "thin_negative_space",
        "lineart",
        _draw_thin_negative_space,
        frozenset(),
    ),
    (
        "qa-hard-stop-gradient",
        "hard_stop_gradient",
        "geometric",
        _draw_hard_stop_gradient,
        frozenset(),
    ),
    (
        "qa-soft-alpha-shadow",
        "soft_alpha_shadow",
        "illustration",
        _draw_soft_alpha_shadow,
        frozenset({"alpha_fidelity"}),
    ),
    (
        "qa-neutral-tone-steps",
        "neutral_tone_steps",
        "clean_logo",
        _draw_neutral_tone_steps,
        frozenset(),
    ),
)

_CASE_BUILDERS = tuple(base_suite._CASE_BUILDERS) + _EXTRA_CASE_BUILDERS
_BUILDER_BY_CASE = {case_id: builder for case_id, _category, _image_class, builder, _required in _CASE_BUILDERS}


def raw_rgba_sha256(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGBA"), dtype=np.uint8).tobytes()).hexdigest()


def generate_cases(root: Path) -> list[DiagnosticCase]:
    """Generate the original seven fixtures plus five residual-focused cases."""
    root = Path(root)
    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    cases: list[DiagnosticCase] = []
    for case_id, category, image_class, builder, required in _CASE_BUILDERS:
        path = fixtures / f"{case_id}.png"
        builder(256).convert("RGBA").save(path, "PNG", optimize=False)
        cases.append(
            DiagnosticCase(
                case_id=case_id,
                category=category,
                image_class=image_class,
                source_path=path,
                source_sha256=base_suite.sha256_file(path),
                required_metrics=required,
            )
        )
    return cases


def _unmeasured_multiscale() -> dict[str, Any]:
    requested = [f"{float(scale):g}x" for scale in SCALES]
    return {
        "scales_requested": requested,
        "missing_scales": requested.copy(),
        "levels": [],
        "min_source_component_recall": None,
        "min_small_component_recall": None,
        "max_normalized_boundary_excess_px": None,
        "max_visible_error_ratio": None,
        "all_scales_measured": False,
    }


def run_diagnostic_case(
    case: DiagnosticCase,
    *,
    output_root: Path,
    engine_version: str,
    repeat_count: int,
    trace_mode: str = "auto",
) -> dict[str, Any]:
    original_pipeline = base_suite.run_pipeline
    base_suite.run_pipeline = run_pipeline
    try:
        result = base_suite.run_diagnostic_case(
            case,
            output_root=output_root,
            engine_version=engine_version,
            repeat_count=repeat_count,
            trace_mode=trace_mode,
        )
    finally:
        base_suite.run_pipeline = original_pipeline

    case_root = Path(output_root) / "cases" / case.case_id
    residual: dict[str, Any] | None = None
    multi_scale = _unmeasured_multiscale()
    residual_error: str | None = None
    multiscale_error: str | None = None

    source_path = case_root / "source.png"
    render_path = case_root / "render.png"
    selected_svg = case_root / "selected.svg"
    if source_path.is_file() and render_path.is_file():
        try:
            with Image.open(source_path) as source_image, Image.open(render_path) as render_image:
                source_rgba = np.asarray(source_image.convert("RGBA"), dtype=np.uint8).copy()
                render_rgba = np.asarray(render_image.convert("RGBA"), dtype=np.uint8).copy()
            residual = measure_residual_error(source_rgba, render_rgba)
        except BaseException as exc:  # diagnostic evidence must fail closed
            residual_error = f"{type(exc).__name__}: {exc}"

    builder = _BUILDER_BY_CASE.get(case.case_id)
    if selected_svg.is_file() and builder is not None:
        try:
            multi_scale = measure_multiscale_svg(selected_svg, builder, base_size=256)
        except BaseException as exc:  # diagnostic evidence must fail closed
            multiscale_error = f"{type(exc).__name__}: {exc}"
            multi_scale = _unmeasured_multiscale()

    near_zero = build_near_zero_contract(
        residual,
        deterministic=result.get("deterministic"),
        structural_failures=result.get("structural_failures") or [],
        multi_scale=multi_scale,
    )
    result["source_rgba_sha256"] = (
        raw_rgba_sha256(_BUILDER_BY_CASE[case.case_id](256))
        if case.case_id in _BUILDER_BY_CASE
        else None
    )
    result["residual"] = residual
    result["residual_measurement_error"] = residual_error
    result["multi_scale"] = multi_scale
    result["multi_scale_measurement_error"] = multiscale_error
    result["near_zero"] = near_zero
    result["near_zero_measured"] = residual is not None
    return result


def build_report(
    case_results: list[dict[str, Any]],
    *,
    engine_version: str,
    repeat_count: int,
    fail_on: str,
) -> dict[str, Any]:
    report = base_suite.build_report(
        case_results,
        engine_version=engine_version,
        repeat_count=repeat_count,
        fail_on=fail_on,
    )
    contracts = [item.get("near_zero") for item in report["cases"]]
    pass_count = sum(1 for contract in contracts if (contract or {}).get("ready") is True)
    measured_count = sum(1 for item in report["cases"] if item.get("near_zero_measured") is True)
    report.update(
        {
            "residual_schema": RESIDUAL_SCHEMA,
            "near_zero_policy_version": next(
                (
                    contract.get("policy_version")
                    for contract in contracts
                    if isinstance(contract, dict) and contract.get("policy_version")
                ),
                None,
            ),
            "near_zero_pass_count": pass_count,
            "near_zero_measured_count": measured_count,
            "all_near_zero": bool(report["cases"]) and pass_count == len(report["cases"]),
            "near_zero_blocker_histogram": blocker_histogram(contracts),
        }
    )
    return report


def should_fail(report: dict[str, Any], fail_on: str) -> bool:
    if fail_on == "near-zero":
        cases = report.get("cases") or []
        return not cases or any((item.get("near_zero") or {}).get("ready") is not True for item in cases)
    return base_suite.should_fail(report, fail_on)


def _fmt(value: object, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value):.{digits}f}"


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    base_suite.write_report(output_root, report)
    summary_path = Path(output_root) / SUMMARY_FILENAME
    lines = [
        "",
        "## Near-zero residual contract",
        "",
        f"- Policy: `{report.get('near_zero_policy_version') or 'unmeasured'}`",
        f"- Near-zero: **{report.get('near_zero_pass_count', 0)}/{report.get('case_count', 0)}**",
        f"- Residual measured: **{report.get('near_zero_measured_count', 0)}/{report.get('case_count', 0)}**",
        f"- Production gate: **false** (diagnostic-only)",
        f"- Blockers: `{json.dumps(report.get('near_zero_blocker_histogram') or {}, sort_keys=True)}`",
        "",
        "| Case | Near-zero | Residual | ΔE00 p95 | Source CC | Small CC | Min CC IoU | Boundary p95 | Blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("cases") or []:
        residual = item.get("residual") or {}
        contract = item.get("near_zero") or {}
        lines.append(
            "| {case} | {ready} | {residual} | {de} | {source_cc} | {small_cc} | {min_iou} | {boundary} | {blockers} |".format(
                case=item.get("case_id"),
                ready="PASS" if contract.get("ready") is True else "FAIL",
                residual=(
                    "—"
                    if residual.get("visible_error_ratio") is None
                    else f"{float(residual['visible_error_ratio']) * 100.0:.3f}%"
                ),
                de=_fmt(residual.get("de00_p95"), 3),
                source_cc=_fmt(residual.get("source_component_recall"), 3),
                small_cc=_fmt(residual.get("small_component_recall"), 3),
                min_iou=_fmt(residual.get("min_component_iou"), 3),
                boundary=_fmt(residual.get("boundary_p95_px"), 3),
                blockers=", ".join(contract.get("blocker_codes") or []) or "—",
            )
        )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


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
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument(
        "--fail-on",
        choices=("none", "structural", "hard", "near-zero"),
        default="structural",
    )
    args = parser.parse_args()
    report = run_suite(
        args.output,
        engine_version=args.engine_version,
        repeat_count=args.repeat_count,
        fail_on=args.fail_on,
        case_limit=args.case_limit,
    )
    failed = should_fail(report, args.fail_on)
    print(
        json.dumps(
            {
                "status": "failed" if failed else "ok",
                "schema": report["schema"],
                "residual_schema": report["residual_schema"],
                "case_count": report["case_count"],
                "severity_counts": report["severity_counts"],
                "near_zero_pass_count": report["near_zero_pass_count"],
                "near_zero_measured_count": report["near_zero_measured_count"],
                "near_zero_blocker_histogram": report["near_zero_blocker_histogram"],
                "structural_failure_codes": report["structural_failure_codes"],
                "hard_failure_codes": report["hard_failure_codes"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
