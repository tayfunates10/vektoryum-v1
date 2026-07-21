"""Alpha-safe production bindings for transparent raster inputs.

Color preprocessing keeps its established RGB for palette and geometry work.
Before tracing, partially transparent boundary pixels use straight source RGB;
fully transparent pixels keep the established comparison-background RGB because
their source RGB is undefined and the final source-alpha mask removes it.

The trace input remains deliberately opaque RGB. Source alpha is applied once,
after all SVG mutations, by :mod:`app.alpha_svg_mask`. Transparent gradient
candidates receive the same straight-boundary RGB contract instead of a
white-composited RGBA input, preventing source alpha from being composited twice.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_ALPHA_COLOR_MODES = {
    "geometric_logo",
    "minimal_ai",
    "flat_logo",
    "logo_color",
    "photo_poster",
}


def _rgba_from_source_at_size(source_path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load source RGBA at the exact trace size and mirror-transform contract."""
    with Image.open(source_path) as source:
        rgba_image = source.convert("RGBA")
        if rgba_image.size != size:
            rgba_image = rgba_image.resize(size, Image.Resampling.LANCZOS)
        rgba = np.asarray(rgba_image, dtype=np.uint8).copy()

    # preprocess_for_mode applies this before dispatch. Reapply it to the source
    # RGBA plane so alpha follows the same deterministic geometric transform as
    # the RGB trace input.
    from app.preprocess import _symmetrize_if_mirror  # noqa: PLC0415

    return np.asarray(_symmetrize_if_mirror(rgba, {"steps": []}), dtype=np.uint8)


def _snap_to_trace_palette(colors: np.ndarray, trace_rgb: np.ndarray) -> np.ndarray:
    """Snap each RGB in ``colors`` to the nearest colour already in ``trace_rgb``.

    The trace image is the quantized/cleaned RGB whose small palette drives
    VTracer colour layering. Soft-boundary source pixels are mapped onto that
    exact palette so the anti-fringe repair never introduces new colours and the
    established banding is preserved. Deterministic: ties resolve to the first
    palette entry in lexicographic ``(r, g, b)`` order.
    """
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if colors.size == 0:
        return colors.astype(np.uint8)
    palette = np.unique(
        np.asarray(trace_rgb, dtype=np.uint8).reshape(-1, 3), axis=0
    ).astype(np.int32)
    snapped = np.empty_like(colors, dtype=np.uint8)
    query = colors.astype(np.int32)
    # Bounded-memory chunks keep the pairwise distance matrix small for images
    # with many soft-boundary pixels.
    for start in range(0, len(query), 4096):
        chunk = query[start : start + 4096][:, None, :]
        distances = ((chunk - palette[None, :, :]) ** 2).sum(axis=2)
        snapped[start : start + 4096] = palette[distances.argmin(axis=1)].astype(
            np.uint8
        )
    return snapped


def _atomic_write_rgb(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".alpha-stage.png",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            temporary,
            format="PNG",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_source_alpha(
    source_path: Path,
    processed_path: Path,
    report: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Prepare trace-safe RGB and bind the transformed source alpha, fail-closed."""
    processed_path = Path(processed_path)
    with Image.open(processed_path) as processed_image:
        processed_rgb = np.asarray(
            processed_image.convert("RGB"), dtype=np.uint8
        ).copy()
        target_size = processed_image.size

    source_rgba = _rgba_from_source_at_size(Path(source_path), target_size)
    if source_rgba.ndim != 3 or source_rgba.shape[2] != 4:
        raise RuntimeError("source_alpha_contract_invalid_rgba")

    source_alpha = source_rgba[:, :, 3].copy()
    if bool(np.all(source_alpha == 255)):
        return processed_path, report

    # Opaque interiors retain the existing quantized/cleaned RGB. Soft boundary
    # pixels adopt the straight source color but SNAPPED to the nearest color
    # already present in the quantized/cleaned trace image. This keeps the
    # anti-fringe intent (a real object color, never the white composite) while
    # preserving the trace palette: un-snapped straight RGB re-injects the full
    # continuous source ramp at supersampled hard edges (a 2x LANCZOS band edge
    # becomes 0<alpha<255), de-quantizing the trace input and collapsing VTracer
    # colour banding (gradient bands merge into a single flat fill). Fully
    # transparent pixels retain the existing processed composite: their RGB is
    # invisible in the source and the final vector mask removes the traced
    # background.
    output_rgb = processed_rgb.copy()
    partial = (source_alpha > 0) & (source_alpha < 255)
    transparent = source_alpha == 0
    boundary_rgb = _snap_to_trace_palette(
        source_rgba[:, :, :3][partial], processed_rgb
    )
    output_rgb[partial] = boundary_rgb

    _atomic_write_rgb(processed_path, output_rgb)

    # Read-after-write proof: a failed codec/write must never silently alter the
    # trace-background or soft-boundary contract.
    with Image.open(processed_path) as verified_image:
        if verified_image.mode != "RGB":
            raise RuntimeError("source_alpha_trace_input_not_rgb")
        verified_rgb = np.asarray(verified_image, dtype=np.uint8).copy()
    if verified_rgb.shape != output_rgb.shape or not np.array_equal(
        verified_rgb, output_rgb
    ):
        raise RuntimeError("source_alpha_trace_input_verification_failed")
    if not np.array_equal(verified_rgb[transparent], processed_rgb[transparent]):
        raise RuntimeError("source_alpha_trace_background_changed")
    if not np.array_equal(verified_rgb[partial], boundary_rgb):
        raise RuntimeError("source_alpha_soft_boundary_rgb_changed")

    alpha_bytes = np.ascontiguousarray(source_alpha).tobytes()
    steps = report.setdefault("steps", [])
    if "source_alpha_staged" not in steps:
        steps.append("source_alpha_staged")
    report["source_alpha"] = {
        "status": "staged_for_vector_mask",
        "width": int(target_size[0]),
        "height": int(target_size[1]),
        "minimum": int(source_alpha.min(initial=255)),
        "maximum": int(source_alpha.max(initial=0)),
        "transparent_pixel_fraction": round(float(np.mean(source_alpha < 255)), 8),
        "soft_alpha_fraction": round(
            float(np.mean((source_alpha > 0) & (source_alpha < 255))), 8
        ),
        "alpha_sha256": hashlib.sha256(alpha_bytes).hexdigest(),
        "trace_input_mode": "RGB",
        "trace_background_policy": "retain_processed_composite",
        "soft_boundary_rgb_policy": "palette_snapped_source_rgb",
        "finalizer": "rfv3d2-source-alpha-vector-mask-v1",
    }
    return processed_path, report


def _gradient_trace_rgb(source_path: Path) -> tuple[np.ndarray, bool]:
    """Return the historical opaque gradient RGB with straight soft-edge RGB.

    Opaque inputs are reported as unchanged so the wrapper can preserve the exact
    old call path. For transparent inputs, alpha-zero RGB remains white (the
    gradient engine's established comparison background), while 0<alpha<255 uses
    unassociated source RGB. The resulting image is opaque RGB, so the gradient
    loader cannot composite the same source alpha a second time.
    """
    with Image.open(source_path) as source:
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    alpha = rgba[:, :, 3]
    if bool(np.all(alpha == 255)) or bool(np.all(alpha == 0)):
        # Fully opaque input keeps exact historical behavior. A fully transparent
        # image has no visible boundary color to repair and is rejected later by
        # the empty source-alpha mask contract, so its existing call path is also
        # preserved.
        return rgba[:, :, :3].copy(), False

    alpha_f = alpha.astype(np.float32)[:, :, None] / 255.0
    rgb = np.clip(
        rgba[:, :, :3].astype(np.float32) * alpha_f
        + 255.0 * (1.0 - alpha_f),
        0.0,
        255.0,
    ).astype(np.uint8)
    partial = (alpha > 0) & (alpha < 255)
    rgb[partial] = rgba[:, :, :3][partial]
    return rgb, True


def wrap_preprocess_for_mode(
    original: Callable[..., tuple[Path, dict[str, Any]]],
) -> Callable[..., tuple[Path, dict[str, Any]]]:
    """Wrap preprocess_for_mode with source-alpha staging for color modes."""
    if getattr(original, "__vektoryum_alpha_preserving__", False):
        return original

    @wraps(original)
    def alpha_preserving_preprocess(
        image_path: Path,
        mode: str,
        output_dir: Path,
        analysis: dict[str, Any] | None = None,
        color_override: int | None = None,
        output_suffix: str = "",
    ) -> tuple[Path, dict[str, Any]]:
        processed_path, report = original(
            image_path,
            mode,
            output_dir,
            analysis=analysis,
            color_override=color_override,
            output_suffix=output_suffix,
        )
        if mode not in _ALPHA_COLOR_MODES:
            return Path(processed_path), report
        return _stage_source_alpha(
            Path(image_path), Path(processed_path), dict(report)
        )

    alpha_preserving_preprocess.__vektoryum_alpha_preserving__ = True
    return alpha_preserving_preprocess


def wrap_gradient_vectorizer(
    original: Callable[..., None],
) -> Callable[..., None]:
    """Give transparent gradients an opaque straight-boundary RGB input."""
    if getattr(original, "__vektoryum_alpha_safe__", False):
        return original

    @wraps(original)
    def alpha_safe_gradient(
        input_path: Path,
        output_path: Path,
        params: dict[str, Any] | None = None,
    ) -> None:
        source_path = Path(input_path)
        staged_rgb, transparent = _gradient_trace_rgb(source_path)
        if not transparent:
            # Preserve the exact historical path and bytes for opaque sources.
            original(source_path, output_path, params)
            return

        descriptor, temporary_name = tempfile.mkstemp(
            dir=Path(output_path).parent,
            prefix=f".{Path(output_path).stem}.",
            suffix=".gradient-alpha-rgb.png",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            Image.fromarray(staged_rgb, mode="RGB").save(temporary, format="PNG")
            with Image.open(temporary) as verified:
                verified_rgb = np.asarray(verified.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(verified_rgb, staged_rgb):
                raise RuntimeError("gradient_alpha_rgb_staging_verification_failed")
            original(temporary, output_path, params)
        finally:
            temporary.unlink(missing_ok=True)

    alpha_safe_gradient.__vektoryum_alpha_safe__ = True
    return alpha_safe_gradient
