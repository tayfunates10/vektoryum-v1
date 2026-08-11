"""Deterministic neutral/luminance source classification for AI-2.

The detector intentionally answers a narrow question: does the *source* contain
at least three spatially supported, neutral, distinct luminance plateaus?  It is
not a generic colour classifier and it never routes images to ``logo_color``.

Anti-aliased black/white edges contain many intermediate grey pixels, so simply
counting RGB/luminance values would create false "third bands".  We therefore
only count neutral pixels that live inside a locally-flat 3x3 neighbourhood and
have opaque source support.  The resulting histogram is deterministic; there is
no k-means seed or order-dependent clustering in this decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# These are detector-shape constants, not release-quality thresholds.  Existing
# palette caps / fidelity gates are deliberately left untouched by AI-2 P0-B2.
_NEUTRAL_CHANNEL_SPREAD = 12
_LOCAL_LUMA_SPREAD = 4
_BAND_JOIN_GAP = 6
_MIN_BAND_SEPARATION = 12
_MIN_OPAQUE_ALPHA = 250
_MIN_FLAT_PIXELS = 6
_MIN_FLAT_FRACTION = 0.0005

# P0-B2b deliberately reuses the geometric candidate family: it preserves the
# limited neutral palette without opening the broad ``logo_color`` path.
_LAYERED_NEUTRAL_AUTO_MODE = "geometric_logo"


@dataclass(frozen=True)
class NeutralLuminanceReport:
    layered_neutral: bool
    band_count: int
    band_centers: tuple[int, ...]
    band_pixel_counts: tuple[int, ...]
    opaque_pixels: int
    neutral_pixels: int
    flat_neutral_pixels: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["band_centers"] = list(self.band_centers)
        data["band_pixel_counts"] = list(self.band_pixel_counts)
        return data


def _source_arrays(source: Image.Image | Path | str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(source, Image.Image):
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    else:
        with Image.open(Path(source)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3], rgba[:, :, 3]


def _merge_histogram_bands(hist: np.ndarray, minimum_count: int) -> list[tuple[int, int]]:
    """Collapse nearby supported luminance bins into weighted plateaus."""
    supported = [int(i) for i in np.flatnonzero(hist >= minimum_count)]
    if not supported:
        return []

    groups: list[list[int]] = [[supported[0]]]
    for value in supported[1:]:
        if value - groups[-1][-1] <= _BAND_JOIN_GAP:
            groups[-1].append(value)
        else:
            groups.append([value])

    bands: list[tuple[int, int]] = []
    for group in groups:
        counts = hist[group].astype(np.int64)
        total = int(counts.sum())
        if total < minimum_count:
            continue
        center = int(round(float(np.average(np.asarray(group), weights=counts))))
        bands.append((center, total))

    # A wide AA shoulder can leave two supported histogram islands close to one
    # another.  Merge those too unless their weighted plateaus are genuinely
    # distinct in luminance.
    merged: list[tuple[int, int]] = []
    for center, count in bands:
        if merged and center - merged[-1][0] < _MIN_BAND_SEPARATION:
            prev_center, prev_count = merged[-1]
            total = prev_count + count
            new_center = int(round((prev_center * prev_count + center * count) / total))
            merged[-1] = (new_center, total)
        else:
            merged.append((center, count))
    return merged


def _detect_arrays(rgb: np.ndarray, alpha: np.ndarray) -> NeutralLuminanceReport:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or alpha.shape != rgb.shape[:2]:
        raise ValueError("neutral detector expects RGB + alpha arrays with matching dimensions")

    opaque = alpha >= _MIN_OPAQUE_ALPHA
    opaque_count = int(opaque.sum())
    if opaque_count == 0:
        return NeutralLuminanceReport(False, 0, (), (), 0, 0, 0, "no_opaque_source_pixels")

    rgb_i = rgb.astype(np.int16)
    spread = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    neutral = opaque & (spread <= _NEUTRAL_CHANNEL_SPREAD)
    neutral_count = int(neutral.sum())
    if neutral_count == 0:
        return NeutralLuminanceReport(False, 0, (), (), opaque_count, 0, 0, "no_neutral_pixels")

    # OpenCV's RGB->GRAY transform is integer/deterministic for uint8 input.
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    local_hi = cv2.dilate(luma, kernel)
    local_lo = cv2.erode(luma, kernel)
    neutral_support = cv2.erode(neutral.astype(np.uint8), kernel) > 0
    flat = neutral & neutral_support & ((local_hi.astype(np.int16) - local_lo.astype(np.int16)) <= _LOCAL_LUMA_SPREAD)
    flat_count = int(flat.sum())

    minimum_count = max(_MIN_FLAT_PIXELS, int(round(opaque_count * _MIN_FLAT_FRACTION)))
    if flat_count < minimum_count * 3:
        return NeutralLuminanceReport(
            False, 0, (), (), opaque_count, neutral_count, flat_count,
            "insufficient_spatially_supported_neutral_pixels",
        )

    hist = np.bincount(luma[flat].reshape(-1), minlength=256)
    bands = _merge_histogram_bands(hist, minimum_count)
    centers = tuple(center for center, _ in bands)
    counts = tuple(count for _, count in bands)
    layered = len(bands) >= 3
    return NeutralLuminanceReport(
        layered, len(bands), centers, counts, opaque_count, neutral_count, flat_count,
        "layered_neutral" if layered else "fewer_than_three_neutral_luminance_bands",
    )


@lru_cache(maxsize=128)
def _detect_path_cached(path_text: str, mtime_ns: int, size: int) -> NeutralLuminanceReport:
    # mtime/size are deliberately part of the cache key so a job-local file
    # replacement cannot reuse a stale source classification.
    del mtime_ns, size
    rgb, alpha = _source_arrays(Path(path_text))
    return _detect_arrays(rgb, alpha)


def detect_neutral_luminance_bands(source: Image.Image | Path | str) -> dict[str, Any]:
    """Return a deterministic report for real, spatially-supported neutral bands."""
    if isinstance(source, Image.Image):
        rgb, alpha = _source_arrays(source)
        return _detect_arrays(rgb, alpha).to_dict()

    path = Path(source)
    stat = path.stat()
    return _detect_path_cached(str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)).to_dict()


def geometric_preserves_neutral_palette(mode: str, report: dict[str, Any]) -> bool:
    """P0-B2a guard: *only* layered-neutral ``geometric_logo`` bypasses B/W/R."""
    return mode == "geometric_logo" and bool(report.get("layered_neutral"))


def route_auto_layered_neutral_mode(recommended_mode: str, report: dict[str, Any]) -> str:
    """P0-B2b: narrowly escape binary ``single_color`` for true neutral layers.

    Only an analyzer recommendation of ``single_color`` is eligible, and only
    when the same source detector proves 3+ spatially-supported neutral luminance
    plateaus.  Explicit user-selected modes are handled by the caller and never
    reach this helper.  ``logo_color`` is intentionally not a target.
    """
    if recommended_mode == "single_color" and bool(report.get("layered_neutral")):
        return _LAYERED_NEUTRAL_AUTO_MODE
    return recommended_mode
