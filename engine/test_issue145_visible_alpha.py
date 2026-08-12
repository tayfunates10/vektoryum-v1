from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.final_artifact_evaluator import evaluate_final_svg
from app.pipeline_entry import run_pipeline
from app.source_truth import render_svg_to_rgba
from regression.residual_error_metrics import measure_residual_error


_SOFT_RAW_RGBA_SHA256 = "18f45af7651377c7d0f2a49b4c3098b459b9334ea369eba9dad4e27a12e73a8f"
_OVERLAP_RAW_RGBA_SHA256 = "a5600f3ce08388eb75e3a657ecba9a118524c6ab7936300a247f504a1dd0ac50"


def _draw_transparent_overlap(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 40, 168, 208), fill=(30, 150, 245, 150))
    draw.rounded_rectangle((92, 28, 228, 220), radius=28, fill=(250, 45, 85, 190))
    draw.ellipse((100, 88, 156, 160), fill=(255, 255, 255, 0))
    return image


def _draw_soft_alpha_shadow(size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = size * 0.50
    cy = size * 0.66
    rx = max(1.0, size * 0.36)
    ry = max(1.0, size * 0.18)
    distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    shadow_alpha = np.rint(np.clip(1.0 - distance, 0.0, 1.0) * 105.0).astype(
        np.uint8
    )
    shadow = np.zeros((size, size, 4), dtype=np.uint8)
    shadow[:, :, :3] = 28
    shadow[:, :, 3] = shadow_alpha
    image = Image.fromarray(shadow, mode="RGBA")

    blue = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(blue)
    draw.ellipse((24, 48, 168, 208), fill=(28, 148, 242, 172))
    image = Image.alpha_composite(image, blue)

    red = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(red)
    draw.rounded_rectangle(
        (92, 34, 228, 214),
        radius=28,
        fill=(248, 48, 82, 198),
    )
    return Image.alpha_composite(image, red)


def _raw_rgba_sha256(image: Image.Image) -> str:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(rgba).tobytes()).hexdigest()


def _white_composite(rgba: np.ndarray) -> np.ndarray:
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        array[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


@pytest.mark.parametrize(
    ("case_id", "builder", "expected_raw_sha"),
    [
        ("qa-soft-alpha-shadow", _draw_soft_alpha_shadow, _SOFT_RAW_RGBA_SHA256),
        ("qa-transparent-overlap", _draw_transparent_overlap, _OVERLAP_RAW_RGBA_SHA256),
    ],
)
def test_issue145_real_production_visible_alpha_contract(
    tmp_path: Path,
    case_id: str,
    builder,
    expected_raw_sha: str,
) -> None:
    source_image = builder(256).convert("RGBA")
    assert _raw_rgba_sha256(source_image) == expected_raw_sha

    source_path = tmp_path / f"{case_id}.png"
    source_image.save(source_path, "PNG", optimize=False)
    source_rgba = np.asarray(source_image, dtype=np.uint8).copy()
    source_rgb = _white_composite(source_rgba)
    artifact_shas: list[str] = []

    for repeat in (1, 2):
        job_dir = tmp_path / f"{case_id}-repeat-{repeat}"
        job_dir.mkdir()
        result = run_pipeline(
            source_image.copy(),
            source_path,
            "auto",
            job_dir,
        )
        best = result.get("best") or {}
        svg_path = Path(best["svg_path"])
        assert svg_path.is_file()

        data = svg_path.read_bytes()
        artifact_shas.append(hashlib.sha256(data).hexdigest())
        lowered = data.lower()
        assert b"<image" not in lowered
        assert b"data:image" not in lowered
        assert b"base64" not in lowered

        rendered = render_svg_to_rgba(svg_path, 256, 256)
        assert rendered is not None
        residual = measure_residual_error(source_rgba, rendered)
        alpha = residual["alpha"] or {}

        assert residual["visible_error_ratio"] <= 0.01
        assert residual["de00_p95"] <= 2.3
        assert residual["boundary_p95_px"] <= 0.75
        assert residual["source_component_recall"] == pytest.approx(1.0)
        assert residual["render_component_precision"] == pytest.approx(1.0)
        assert residual["min_component_iou"] >= 0.95
        assert float(alpha["alpha_iou"]) >= 0.999
        assert float(alpha["alpha_mae"]) <= 0.5 / 255.0

        evaluator = evaluate_final_svg(
            svg_path,
            source_rgb,
            source_alpha=source_rgba[:, :, 3],
            image_class="illustration",
            required_metrics={"alpha_fidelity"},
        )
        multibackground_codes = [
            code
            for code in evaluator.hard_fail_codes
            if code.startswith(("alpha_white_", "alpha_black_", "alpha_checker_"))
        ]
        assert multibackground_codes == []

        alpha_report = result.get("alpha_mask_report") or {}
        visible_after = alpha_report.get("visible_paint_after") or {}
        assert visible_after.get("passed") is True
        assert float(visible_after["visible_composite_residual"]) <= 0.01
        assert float(visible_after["visible_composite_de00_p95"]) <= 2.3
        assert int(alpha_report.get("raster_embed_count", 0)) == 0

        if alpha_report.get("visible_paint_repair_applied"):
            assert int(alpha_report["visible_paint_after_bytes"]) <= int(
                alpha_report["visible_paint_byte_limit"]
            )
            assert int(alpha_report["visible_paint_after_path_count"]) <= int(
                alpha_report["visible_paint_path_limit"]
            )
            assert int(alpha_report["visible_paint_after_node_count"]) <= int(
                alpha_report["visible_paint_node_limit"]
            )

    assert artifact_shas[0] == artifact_shas[1]
