"""Deterministic Benchmark v1 seed-corpus generator and report writer.

The seed corpus is synthetic and CC0, so CI can exercise the benchmark data path
without committing binary fixtures or downloading untrusted third-party assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.manifest import BenchmarkCase, validate_manifest

CATEGORIES = (
    "logos",
    "seals",
    "technical",
    "signatures",
    "gradients",
    "low_resolution",
    "transparent",
    "multilingual",
)


def _draw_case(category: str, size: int = 192) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if category == "logos":
        draw.rectangle((24, 36, 168, 156), fill=(20, 30, 45, 255))
        draw.ellipse((58, 58, 134, 134), fill=(230, 35, 45, 255))
    elif category == "seals":
        draw.ellipse((18, 18, 174, 174), outline=(15, 15, 15, 255), width=10)
        draw.ellipse((48, 48, 144, 144), outline=(15, 15, 15, 255), width=4)
    elif category == "technical":
        draw.line((20, 96, 172, 96), fill=(0, 0, 0, 255), width=3)
        draw.line((96, 20, 96, 172), fill=(0, 0, 0, 255), width=3)
        draw.rectangle((52, 52, 140, 140), outline=(0, 0, 0, 255), width=3)
    elif category == "signatures":
        points = [(18, 118), (48, 74), (72, 126), (108, 54), (132, 122), (174, 86)]
        draw.line(points, fill=(10, 10, 10, 255), width=4, joint="curve")
    elif category == "gradients":
        for x in range(size):
            t = x / (size - 1)
            draw.line((x, 28, x, 164), fill=(int(255 * (1 - t)), 60, int(255 * t), 255))
    elif category == "low_resolution":
        tiny = Image.new("RGB", (24, 24), "white")
        td = ImageDraw.Draw(tiny)
        td.rectangle((3, 3, 20, 20), fill="black")
        td.ellipse((7, 7, 16, 16), fill="white")
        image = tiny.resize((size, size), Image.Resampling.NEAREST).convert("RGBA")
    elif category == "transparent":
        draw.ellipse((28, 28, 164, 164), fill=(40, 160, 255, 128))
        draw.rectangle((70, 18, 122, 174), fill=(255, 40, 80, 180))
    else:  # multilingual
        draw.rectangle((20, 46, 172, 146), outline=(20, 20, 20, 255), width=4)
        draw.text((34, 76), "TR EN 123", fill=(20, 20, 20, 255))
    return image


def generate_seed_corpus(output_dir: Path) -> list[BenchmarkCase]:
    fixtures = output_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    cases: list[BenchmarkCase] = []
    for index, category in enumerate(CATEGORIES, start=1):
        case_id = f"seed-{index:02d}-{category}"
        path = fixtures / f"{case_id}.png"
        _draw_case(category).save(path, format="PNG", optimize=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                category=category,
                source_path=str(path.relative_to(output_dir)),
                license_id="CC0-1.0",
                source_sha256=digest,
                tags=("synthetic", "seed-v1"),
            )
        )
    validate_manifest(cases, corpus_root=output_dir)
    return cases


def write_reports(output_dir: Path, cases: list[BenchmarkCase]) -> None:
    payload = {
        "schema_version": "benchmark-seed-v1",
        "case_count": len(cases),
        "categories": sorted({case.category for case in cases}),
        "cases": [asdict(case) for case in cases],
    }
    (output_dir / "seed_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = "".join(
        f"<tr><td>{case.case_id}</td><td>{case.category}</td><td>{case.license_id}</td></tr>"
        for case in cases
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Vektoryum Benchmark Seed</title>"
        "</head><body><h1>Benchmark v1 Seed Corpus</h1>"
        f"<p>Case count: {len(cases)}</p><table><thead><tr><th>ID</th><th>Category</th>"
        f"<th>License</th></tr></thead><tbody>{rows}</tbody></table></body></html>"
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = generate_seed_corpus(args.output)
    write_reports(args.output, cases)
    print(json.dumps({"status": "ok", "case_count": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
