from __future__ import annotations

from PIL import Image, ImageDraw

from app.canonical_svg_candidate import build_canonical_svg_candidate


def _fixture() -> Image.Image:
    image = Image.new("RGBA", (12, 10), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 9, 7), fill=(220, 20, 60, 255))
    draw.rectangle((4, 3, 7, 6), fill=(0, 0, 0, 255))
    return image


def test_candidate_is_valid_and_deterministic() -> None:
    first = build_canonical_svg_candidate(_fixture(), max_colors=8)
    second = build_canonical_svg_candidate(_fixture(), max_colors=8)

    assert first.valid is True
    assert first.errors == ()
    assert first.document is not None
    assert first.promotion is not None and first.promotion.ready is True
    assert first.document.svg_text.startswith("<svg")
    assert first.document.document_sha256 == first.promotion.document_sha256
    assert first.document.document_sha256 == second.document.document_sha256
    assert first.document.svg_text == second.document.svg_text
    assert first.graph_stats["valid"] is True
    assert first.palette_size >= 2


def test_transparency_is_composited_deterministically() -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((1, 1, 6, 6), fill=(0, 80, 255, 255))

    report = build_canonical_svg_candidate(image, max_colors=4)

    assert report.valid is True
    assert report.document is not None
    assert report.document.width == 8
    assert report.document.height == 8


def test_invalid_configuration_fails_closed_without_payload() -> None:
    report = build_canonical_svg_candidate(_fixture(), max_colors=1)

    assert report.valid is False
    assert report.document is None
    assert report.promotion is None
    assert report.errors == ("max_colors must be an integer between 2 and 64",)


def test_pixel_budget_fails_closed_without_work() -> None:
    report = build_canonical_svg_candidate(
        Image.new("RGB", (20, 20), "white"),
        max_pixels=100,
    )

    assert report.valid is False
    assert report.document is None
    assert report.errors == ("image exceeds canonical candidate pixel budget",)


def test_repeat_count_cannot_bypass_promotion_contract() -> None:
    report = build_canonical_svg_candidate(_fixture(), repeat_runs=2)

    assert report.valid is False
    assert report.document is None
    assert report.errors == ("repeat_runs must be an integer greater than or equal to 3",)
