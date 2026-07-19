"""Alpha-safe production bindings for transparent raster inputs.

Color preprocessing still performs its established palette and geometry work on
white-composited RGB. Before tracing, this module replaces undefined transparent
RGB with canonical black and restores straight source RGB on partially
transparent pixels. The trace input remains deliberately opaque RGB: source
alpha is applied once, after all SVG mutations, by ``app.alpha_svg_mask``.

Keeping alpha out of the tracer prevents renderer-dependent alpha multiplication
while the staged hash and read-back proof keep the source plane bound to the job.
Transparent gradient candidates remain fail-closed until that engine has an
alpha-aware region/mask contract.
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
    # pixels use straight source RGB, and fully transparent pixels are canonical
    # black. The final vector mask supplies the only alpha plane.
    output_rgb = processed_rgb.copy()
    partial = (source_alpha > 0) & (source_alpha < 255)
    transparent = source_alpha == 0
    output_rgb[partial] = source_rgba[:, :, :3][partial]
    output_rgb[transparent] = 0

    _atomic_write_rgb(processed_path, output_rgb)

    # Read-after-write proof: a failed codec/write must never silently return to
    # the previous white-composited trace input.
    with Image.open(processed_path) as verified_image:
        if verified_image.mode != "RGB":
            raise RuntimeError("source_alpha_trace_input_not_rgb")
        verified_rgb = np.asarray(verified_image, dtype=np.uint8).copy()
    if verified_rgb.shape != output_rgb.shape or not np.array_equal(
        verified_rgb, output_rgb
    ):
        raise RuntimeError("source_alpha_trace_input_verification_failed")
    if not bool(np.all(verified_rgb[transparent] == 0)):
        raise RuntimeError("source_alpha_transparent_rgb_not_canonical")

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
        "finalizer": "rfv3d2-source-alpha-vector-mask-v1",
    }
    return processed_path, report


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
    """Reject transparent gradient inputs until native alpha masking exists."""
    if getattr(original, "__vektoryum_alpha_safe__", False):
        return original

    @wraps(original)
    def alpha_safe_gradient(
        input_path: Path,
        output_path: Path,
        params: dict[str, Any] | None = None,
    ) -> None:
        with Image.open(input_path) as source:
            alpha = np.asarray(source.convert("RGBA"), dtype=np.uint8)[:, :, 3].copy()
        if bool(np.any(alpha < 255)):
            raise RuntimeError(
                "transparent_gradient_candidate_requires_alpha_aware_mask"
            )
        original(input_path, output_path, params)

    alpha_safe_gradient.__vektoryum_alpha_safe__ = True
    return alpha_safe_gradient
