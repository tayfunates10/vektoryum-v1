"""Exact-final SVG için özgün, deterministik source-of-truth korpusu.

T1/T2/T3 önce bağımsız oracle SVG olarak tanımlanır, sonra kontrollü CairoSVG
render'ı ile pipeline girdisine çevrilir. Production kodu fixture adını veya
oracle metadata'sını görmez. T5'in pipeline girdisi temiz referans değil,
gerçekten Q32 olarak encode edilip tekrar decode edilmiş JPEG baytlarıdır.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import cairosvg
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Fixture:
    name: str
    rgb: np.ndarray
    alpha: np.ndarray | None
    input_bytes: bytes
    input_mime: str
    reference_rgba: np.ndarray
    oracle_svg: bytes | None
    oracle: dict[str, Any] = field(default_factory=dict)
    protected_rois: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    complexity_budget: dict[str, int] = field(default_factory=dict)


def _render_oracle(svg: bytes, width: int, height: int) -> np.ndarray:
    png = cairosvg.svg2png(
        bytestring=svg, output_width=width, output_height=height,
    )
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()


def _white_composite(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def _png_input(rgba: np.ndarray, keep_alpha: bool) -> bytes:
    buf = io.BytesIO()
    image = Image.fromarray(rgba, "RGBA") if keep_alpha else Image.fromarray(rgba[:, :, :3], "RGB")
    image.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def _from_oracle(
    name: str,
    svg_text: str,
    n: int,
    *,
    oracle: dict[str, Any],
    protected_rois: dict[str, tuple[int, int, int, int]] | None = None,
    complexity_budget: dict[str, int] | None = None,
) -> Fixture:
    svg = svg_text.encode("utf-8")
    rgba = _render_oracle(svg, n, n)
    alpha = rgba[:, :, 3].copy()
    has_alpha = bool((alpha < 255).any())
    return Fixture(
        name=name,
        rgb=_white_composite(rgba),
        alpha=alpha if has_alpha else None,
        input_bytes=_png_input(rgba, keep_alpha=has_alpha),
        input_mime="image/png",
        reference_rgba=rgba,
        oracle_svg=svg,
        oracle={**oracle, "source_has_alpha": has_alpha},
        protected_rois=dict(protected_rois or {}),
        complexity_budget=dict(complexity_budget or {}),
    )


def t1_topology(n: int = 300) -> Fixture:
    """Ortak sınır, T-junction, gerçek counter ve 2/4/6 px çaplı noktalar."""
    x1, x2, x3 = round(n * .1), round(n * .5), round(n * .9)
    y1, y2, y3 = round(n * .1), round(n * .6), round(n * .85)
    hx1, hy1, hx2, hy2 = round(n * .18), round(n * .18), round(n * .34), round(n * .42)
    ix1, iy1, ix2, iy2 = round(n * .22), round(n * .22), round(n * .30), round(n * .38)
    dots: list[str] = []
    rois: dict[str, tuple[int, int, int, int]] = {
        "nested_red_island": (ix1, iy1, ix2, iy2),
        "counter": (hx1, hy1, hx2, hy2),
        "t_junction": (x2 - 4, y2 - 4, x2 + 4, y2 + 4),
    }
    for index, diameter in enumerate((2, 4, 6)):
        cx, cy = round(n * (0.20 + index * .12)), round(n * .92)
        x, y = cx - diameter / 2, cy - diameter / 2
        dots.append(
            f'<rect x="{x:g}" y="{y:g}" width="{diameter}" height="{diameter}" fill="#143cbe"/>'
        )
        rois[f"dot_diameter_{diameter}px"] = (
            max(0, cx - diameter - 2), max(0, cy - diameter - 2),
            min(n, cx + diameter + 2), min(n, cy + diameter + 2),
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{n}" height="{n}">
      <rect width="{n}" height="{n}" fill="#fafafa"/>
      <path fill="#0a0a0a" fill-rule="evenodd" d="M{x1} {y1}H{x2}V{y2}H{x1}Z M{hx1} {hy1}H{hx2}V{hy2}H{hx1}Z"/>
      <rect x="{x2}" y="{y1}" width="{x3-x2}" height="{y2-y1}" fill="#e3000b"/>
      <rect x="{x1}" y="{y2}" width="{x3-x1}" height="{y3-y2}" fill="#ffed00"/>
      <rect x="{ix1}" y="{iy1}" width="{ix2-ix1}" height="{iy2-iy1}" fill="#e3000b"/>
      {''.join(dots)}
    </svg>'''
    return _from_oracle(
        "t1_topology", svg, n,
        oracle={
            "class": "geometric",
            "source_semantics": {
                "shared_boundary": True, "t_junctions": 1,
                "counters": 1, "small_components": 3,
            },
            "accepted_equivalents": "Path decomposition may differ; rendered component/hole deltas must remain zero.",
        },
        protected_rois=rois,
        complexity_budget={"max_paths": 64, "max_nodes": 512, "max_bytes": 40_000},
    )


def t2_gradient_alpha(n: int = 300) -> Fixture:
    """İki linear tanım, bir radial tanım, alpha maskesi ve saydam örtüşme."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{n}" height="{n}">
      <defs>
        <linearGradient id="colorLinear" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#e3000b"/><stop offset="1" stop-color="#ffed00"/>
        </linearGradient>
        <linearGradient id="alphaLinear" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="white" stop-opacity="0.08"/>
          <stop offset="1" stop-color="white" stop-opacity="1"/>
        </linearGradient>
        <radialGradient id="radialPaint" cx="50%" cy="45%" r="48%">
          <stop offset="0" stop-color="#143cbe"/><stop offset="1" stop-color="#ffed00" stop-opacity="0.25"/>
        </radialGradient>
        <mask id="alphaMask"><rect width="{n}" height="{n}" fill="url(#alphaLinear)"/></mask>
      </defs>
      <g mask="url(#alphaMask)">
        <rect width="{n}" height="{n}" fill="url(#colorLinear)"/>
        <circle cx="{n/2:g}" cy="{n*.45:g}" r="{n*.34:g}" fill="url(#radialPaint)" opacity="0.82"/>
        <rect x="{n*.18:g}" y="{n*.22:g}" width="{n*.64:g}" height="{n*.28:g}" fill="#ffffff" opacity="0.28"/>
      </g>
    </svg>'''
    return _from_oracle(
        "t2_gradient_alpha", svg, n,
        oracle={
            "class": "clean_logo",
            "source_semantics": {
                "linear_gradient_definitions": 2,
                "radial_gradient_definitions": 1,
                "uses_alpha_mask": True,
                "has_semistransparent_overlap": True,
            },
            "accepted_equivalents": "Equivalent paint graphs are allowed; definition count itself is not a quality veto.",
            "required_metrics": ["alpha_fidelity", "gradient_fidelity"],
        },
        protected_rois={
            "alpha_top": (0, 0, n, max(4, n // 12)),
            "alpha_bottom": (0, n - max(4, n // 12), n, n),
            "radial_center": (round(n*.4), round(n*.35), round(n*.6), round(n*.55)),
        },
        complexity_budget={"max_paths": 96, "max_nodes": 768, "max_bytes": 64_000},
    )


def t3_micro_detail(n: int = 320) -> Fixture:
    """1 px halka, counter, küçük ® benzeri işaret ve 3/5/7 px çaplı noktalar."""
    ring_cx, ring_cy, ring_r = round(n*.3), round(n*.3), round(n*.18)
    donut_cx, donut_cy, outer_r, inner_r = round(n*.7), round(n*.3), round(n*.16), round(n*.08)
    rcx, rcy, rr = round(n*.82), round(n*.82), max(6, round(n*.05))
    dots: list[str] = []
    rois = {
        "thin_ring_1px": (ring_cx-ring_r-3, ring_cy-ring_r-3, ring_cx+ring_r+3, ring_cy+ring_r+3),
        "counter": (donut_cx-inner_r-3, donut_cy-inner_r-3, donut_cx+inner_r+3, donut_cy+inner_r+3),
        "registered_mark": (rcx-rr-3, rcy-rr-3, rcx+rr+3, rcy+rr+3),
    }
    for index, diameter in enumerate((3, 5, 7)):
        cx, cy = round(n * (0.20 + index * .14)), round(n*.70)
        dots.append(f'<circle cx="{cx}" cy="{cy}" r="{diameter/2:g}" fill="#143cbe"/>')
        rois[f"dot_diameter_{diameter}px"] = (cx-diameter-2, cy-diameter-2, cx+diameter+2, cy+diameter+2)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{n}" height="{n}">
      <rect width="{n}" height="{n}" fill="#fafafa"/>
      <circle cx="{ring_cx}" cy="{ring_cy}" r="{ring_r}" fill="none" stroke="#0a0a0a" stroke-width="1"/>
      <path fill="#e3000b" fill-rule="evenodd" d="M{donut_cx-outer_r} {donut_cy}a{outer_r} {outer_r} 0 1 0 {2*outer_r} 0a{outer_r} {outer_r} 0 1 0 {-2*outer_r} 0 M{donut_cx-inner_r} {donut_cy}a{inner_r} {inner_r} 0 1 0 {2*inner_r} 0a{inner_r} {inner_r} 0 1 0 {-2*inner_r} 0"/>
      <g fill="none" stroke="#0a0a0a" stroke-width="1"><circle cx="{rcx}" cy="{rcy}" r="{rr}"/><path d="M{rcx-rr/2:g} {rcy+rr/2:g}V{rcy-rr/2:g}H{rcx}Q{rcx+rr/2:g} {rcy-rr/2:g} {rcx} {rcy}L{rcx+rr/2:g} {rcy+rr/2:g}"/></g>
      {''.join(dots)}
    </svg>'''
    return _from_oracle(
        "t3_micro_detail", svg, n,
        oracle={
            "class": "lineart",
            "source_semantics": {"one_pixel_ring": True, "counters": 1, "small_components": 3},
            "accepted_equivalents": "Primitive/path choice may differ; every protected semantic component must survive.",
        },
        protected_rois=rois,
        complexity_budget={"max_paths": 96, "max_nodes": 1024, "max_bytes": 64_000},
    )


def t5_lowres_jpeg(n: int = 160) -> Fixture:
    """Mikro rozetin gerçek Q32 JPEG girdisi; blokları path'e çevirme tuzağı."""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{n}" height="{n}">
      <rect width="{n}" height="{n}" fill="#fafafa"/>
      <circle cx="{n/2:g}" cy="{n/2:g}" r="{n*.4:g}" fill="#e3000b"/>
      <circle cx="{n/2:g}" cy="{n/2:g}" r="{n*.22:g}" fill="#fafafa"/>
      <rect x="{n*.42:g}" y="{n*.42:g}" width="{n*.16:g}" height="{n*.16:g}" fill="#0a0a0a"/>
    </svg>'''.encode()
    clean_rgba = _render_oracle(svg, n, n)
    clean_rgb = clean_rgba[:, :, :3]
    encoded = io.BytesIO()
    Image.fromarray(clean_rgb, "RGB").save(
        encoded, "JPEG", quality=32, subsampling=2, optimize=False, progressive=False,
    )
    jpeg_bytes = encoded.getvalue()
    with Image.open(io.BytesIO(jpeg_bytes)) as decoded:
        decoded_rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    return Fixture(
        name="t5_lowres_jpeg",
        rgb=decoded_rgb,
        alpha=None,
        input_bytes=jpeg_bytes,
        input_mime="image/jpeg",
        reference_rgba=clean_rgba,
        oracle_svg=svg,
        oracle={
            "class": "low_res_logo",
            "source_has_alpha": False,
            "jpeg_quality": 32,
            "source_semantics": {"rings": 2, "center_square": 1},
            "accepted_equivalents": "JPEG block/ringing artifacts are not semantic vector geometry.",
        },
        protected_rois={"center_mark": (round(n*.38), round(n*.38), round(n*.62), round(n*.62))},
        complexity_budget={"max_paths": 500, "max_nodes": 5_000, "max_bytes": 150_000},
    )


def all_fixtures() -> list[Fixture]:
    """Pipeline'a verilecek gerçek girdileri döndürür (T5 dahil)."""
    return [t1_topology(), t2_gradient_alpha(), t3_micro_detail(), t5_lowres_jpeg()]
