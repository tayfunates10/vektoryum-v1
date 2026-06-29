from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from app.analyzer import analyze_image
from app.main import (
    basic_svg_quality_check,
    convert_svg_to_dxf,
    multi_candidate_vectorize,
)


def make_antialiased_geometric_logo(path: Path) -> None:
    scale = 3
    image = Image.new("RGBA", (720 * scale, 420 * scale), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    s = scale

    draw.rectangle((24 * s, 24 * s, 696 * s, 396 * s), outline=(255, 0, 0, 255), width=9 * s)
    draw.rectangle((70 * s, 95 * s, 190 * s, 315 * s), fill=(0, 0, 0, 255))
    draw.rectangle((190 * s, 95 * s, 330 * s, 132 * s), fill=(0, 0, 0, 255))
    draw.rectangle((190 * s, 278 * s, 330 * s, 315 * s), fill=(0, 0, 0, 255))
    draw.polygon(
        [
            (410 * s, 88 * s),
            (600 * s, 88 * s),
            (528 * s, 198 * s),
            (616 * s, 318 * s),
            (464 * s, 318 * s),
            (376 * s, 198 * s),
        ],
        fill=(0, 0, 0, 255),
    )
    draw.rectangle((448 * s, 140 * s, 522 * s, 262 * s), fill=(255, 255, 255, 255))

    image = image.resize((720, 420), Image.Resampling.LANCZOS)
    image.save(path)


def make_transparent_single_color_cut(path: Path) -> None:
    image = Image.new("RGBA", (520, 420), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    draw.polygon([(80, 340), (230, 80), (390, 340)], fill=(0, 0, 0, 255))
    draw.ellipse((185, 170, 285, 270), fill=(255, 255, 255, 0))

    image.save(path)


def make_sparse_lineart(path: Path) -> None:
    image = Image.new("RGBA", (640, 420), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    for index in range(6):
        draw.line((80 + index * 20, 60, 500 - index * 10, 340), fill=(0, 0, 0, 255), width=2)

    draw.rectangle((60, 50, 580, 360), outline=(0, 0, 0, 255), width=3)
    draw.ellipse((180, 120, 430, 300), outline=(0, 0, 0, 255), width=3)

    image.save(path)


def make_multicolor_logo(path: Path) -> None:
    image = Image.new("RGBA", (720, 420), (245, 235, 210, 255))
    draw = ImageDraw.Draw(image)

    for y in range(420):
        draw.line((0, y, 720, y), fill=(245 - y // 8, 235 - y // 10, 210 - y // 6, 255))

    draw.ellipse((80, 50, 215, 185), fill=(245, 160, 40, 255))
    draw.polygon([(40, 310), (180, 130), (320, 310)], fill=(70, 120, 105, 255))
    draw.polygon([(230, 315), (400, 110), (590, 315)], fill=(90, 130, 160, 255))

    for x, color in [
        (120, (120, 100, 90, 255)),
        (250, (150, 130, 110, 255)),
        (380, (95, 85, 80, 255)),
        (500, (165, 145, 115, 255)),
    ]:
        draw.rounded_rectangle((x, 280, x + 100, 350), radius=8, fill=color)

    image.save(path)


def make_gradient_color_logo(path: Path) -> None:
    image = Image.new("RGBA", (820, 520), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    for radius in range(210, 20, -4):
        color = (
            255,
            int(80 + radius * 0.55),
            int(20 + radius * 0.18),
            255,
        )
        draw.ellipse((410 - radius, 260 - radius, 410 + radius, 260 + radius), fill=color)

    draw.polygon([(130, 400), (310, 140), (470, 400)], fill=(80, 120, 155, 255))
    draw.polygon([(360, 405), (540, 120), (720, 405)], fill=(105, 80, 140, 255))
    draw.rounded_rectangle((220, 345, 610, 430), radius=22, fill=(35, 38, 42, 255))

    image.save(path)


def make_photo_like_complex(path: Path) -> None:
    height, width = 520, 820
    rng = np.random.default_rng(42)
    base = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            base[y, x] = [
                int(70 + 130 * x / width),
                int(80 + 120 * y / height),
                int(120 + 80 * np.sin((x + y) / 80)),
            ]

    noise = rng.normal(0, 28, base.shape)
    arr = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _assert_case(
    name: str,
    maker: Callable[[Path], None],
    expected_mode: str,
    expected_best: set[str],
    min_candidates: int,
    require_production_ready: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"{name}.png"
        svg_path = tmp_path / f"{name}.svg"
        dxf_path = tmp_path / f"{name}.dxf"
        candidate_dir = tmp_path / f"{name}_candidates"

        maker(input_path)
        analysis = analyze_image(input_path)

        assert analysis["recommended_mode"] == expected_mode, analysis

        report = multi_candidate_vectorize(
            input_path=input_path,
            final_svg_path=svg_path,
            temp_dir=candidate_dir,
            selected_trace_mode=expected_mode,
            selected_quality="detailed",
            analysis_report=analysis,
        )

        quality_report = basic_svg_quality_check(svg_path, expected_mode)
        svg_text = svg_path.read_text(encoding="utf-8", errors="ignore").lower()

        assert report["best_candidate"] in expected_best, report
        assert len(report["candidates"]) >= min_candidates, report
        assert "<image" not in svg_text, "SVG output must not embed bitmap image tags"
        assert quality_report["path_count"] >= 1, quality_report
        assert quality_report["unique_color_count"] >= 1, quality_report

        if require_production_ready:
            assert quality_report["status"] == "production_ready", quality_report

        convert_svg_to_dxf(svg_path, dxf_path, expected_mode)
        assert dxf_path.exists(), "DXF export was not created"


def main() -> None:
    _assert_case(
        name="antialiased_geometric_logo",
        maker=make_antialiased_geometric_logo,
        expected_mode="geometric_logo",
        expected_best={"geo_standard", "geo_clean", "geo_contour", "geo_mixed", "geo_detail"},
        min_candidates=5,
        require_production_ready=True,
    )
    _assert_case(
        name="transparent_single_color_cut",
        maker=make_transparent_single_color_cut,
        expected_mode="single_color",
        expected_best={"single_clean", "single_potrace", "single_contour"},
        min_candidates=3,
        require_production_ready=True,
    )
    _assert_case(
        name="sparse_lineart",
        maker=make_sparse_lineart,
        expected_mode="lineart",
        expected_best={"lineart_clean", "lineart_detail", "lineart_potrace", "lineart_autotrace"},
        min_candidates=4,
        require_production_ready=True,
    )
    _assert_case(
        name="multicolor_logo",
        maker=make_multicolor_logo,
        expected_mode="logo_color",
        expected_best={"logo_clean", "logo_standard", "logo_detail_rich", "logo_color_preserve", "logo_smooth"},
        min_candidates=5,
        require_production_ready=False,
    )
    _assert_case(
        name="gradient_color_logo",
        maker=make_gradient_color_logo,
        expected_mode="logo_color",
        expected_best={"logo_clean", "logo_standard", "logo_detail_rich", "logo_color_preserve", "logo_smooth"},
        min_candidates=5,
        require_production_ready=True,
    )
    _assert_case(
        name="photo_like_complex",
        maker=make_photo_like_complex,
        expected_mode="photo_poster",
        expected_best={"photo_poster_clean", "photo_poster_detail"},
        min_candidates=2,
        require_production_ready=True,
    )

    print("Synthetic vector quality regression tests passed.")


if __name__ == "__main__":
    main()
