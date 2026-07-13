"""FAZ 3 — exact-final alpha fidelity on white, black and checker backgrounds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def _svg(body: str, n: int = 96) -> bytes:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{n}" height="{n}" '
        f'viewBox="0 0 {n} {n}">{body}</svg>'
    ).encode()


def _source(n: int = 96, alpha: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from app.source_truth import composite_rgba

    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    rgba[:, :, :3] = (227, 0, 11)
    rgba[:, :, 3] = alpha
    return rgba, composite_rgba(rgba, 255), rgba[:, :, 3].copy()


def test_exact_semtransparent_paint_is_measured_on_all_backgrounds() -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    rgba, white_rgb, alpha = _source()
    report = evaluate_final_svg_bytes(
        _svg('<rect width="96" height="96" fill="#e3000b" fill-opacity="0.5019608"/>'),
        white_rgb,
        source_alpha=alpha,
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    )
    group = report.metrics["G_gradient_alpha"]
    assert group["source_has_alpha"] is True
    assert group["alpha_fidelity_status"] in {"passed", "measured"}
    assert group["alpha_iou"] >= 0.995
    assert group["alpha_mae"] <= 0.005
    assert set(group["backgrounds"]) == {"white", "black", "checker"}
    for metric in group["backgrounds"].values():
        assert metric["ssim"] >= 0.995
        assert metric["rgb_mae"] <= 0.008
    assert "alpha_fidelity" not in report.unmeasured_required
    assert not any(code.startswith("alpha_") for code in report.hard_fail_codes)


def test_white_matching_opaque_flattening_fails_black_and_checker_alpha_gates() -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    _rgba, white_rgb, alpha = _source()
    # This opaque color is approximately the source's white-composited appearance.
    flattened = tuple(int(value) for value in white_rgb[0, 0])
    fill = "#%02x%02x%02x" % flattened
    report = evaluate_final_svg_bytes(
        _svg(f'<rect width="96" height="96" fill="{fill}"/>'),
        white_rgb,
        source_alpha=alpha,
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    )
    group = report.metrics["G_gradient_alpha"]
    assert report.verdict == "failed"
    assert group["alpha_iou"] < 0.995 or group["alpha_mae"] > 0.005
    assert group["backgrounds"]["white"]["rgb_mae"] < 0.01
    assert group["backgrounds"]["black"]["rgb_mae"] > group["backgrounds"]["white"]["rgb_mae"]
    assert group["backgrounds"]["checker"]["rgb_mae"] > group["backgrounds"]["white"]["rgb_mae"]
    assert set(report.hard_fail_codes) & {
        "alpha_iou_below_min", "alpha_mae_above_max",
        "alpha_black_ssim_below_min", "alpha_checker_ssim_below_min",
        "alpha_black_mae_above_max", "alpha_checker_mae_above_max",
    }


def test_alpha_loss_cannot_be_hidden_by_high_white_background_ssim() -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    _rgba, white_rgb, alpha = _source(alpha=64)
    flattened = tuple(int(value) for value in white_rgb[0, 0])
    fill = "#%02x%02x%02x" % flattened
    report = evaluate_final_svg_bytes(
        _svg(f'<rect width="96" height="96" fill="{fill}"/>'),
        white_rgb,
        source_alpha=alpha,
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    )
    group = report.metrics["G_gradient_alpha"]
    assert group["backgrounds"]["white"]["ssim"] > 0.99
    assert report.verdict == "failed"
    assert group["alpha_fidelity_status"] == "failed"
