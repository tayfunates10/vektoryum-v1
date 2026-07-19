"""Pre-construction and rollback gates for vector source-alpha masks."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from defusedxml import ElementTree as SafeET

from app.alpha_preprocess import _rgba_from_source_at_size

_MAX_ALPHA_LEVELS = 128
_MAX_MASK_SIDE = 1600
_JOURNAL_BYTE_GROWTH_FACTOR = 3
_JOURNAL_BYTE_GROWTH_ABSOLUTE = 250_000
_FIXED_MARKUP_OVERHEAD = 4096
_MIN_RECT_BYTES = 40


def _dimension(value: str | None) -> float | None:
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:px)?\s*",
        str(value or ""),
    )
    if not match:
        return None
    parsed = float(match.group(1))
    return parsed if np.isfinite(parsed) and parsed > 0 else None


def _viewbox_size(root: Any) -> tuple[int, int]:
    raw = root.get("viewBox") or root.get("viewbox")
    if raw:
        parts = [float(value) for value in re.split(r"[\s,]+", raw.strip()) if value]
        if (
            len(parts) == 4
            and all(np.isfinite(value) for value in parts)
            and parts[2] > 0
            and parts[3] > 0
        ):
            return max(1, int(round(parts[2]))), max(1, int(round(parts[3])))
    width = _dimension(root.get("width"))
    height = _dimension(root.get("height"))
    if width is None or height is None:
        raise RuntimeError("source_alpha_mask_budget_missing_coordinate_contract")
    return max(1, int(round(width))), max(1, int(round(height)))


def _quantize_alpha(alpha: np.ndarray) -> np.ndarray:
    values = np.unique(alpha)
    if int(np.count_nonzero(values)) <= _MAX_ALPHA_LEVELS:
        return alpha.astype(np.uint8, copy=True)
    steps = _MAX_ALPHA_LEVELS - 1
    return np.rint(alpha.astype(np.float32) * steps / 255.0).astype(np.uint8)


def _byte_limit(before_size: int) -> int:
    baseline = max(1, int(before_size))
    return max(
        baseline * _JOURNAL_BYTE_GROWTH_FACTOR,
        baseline + _JOURNAL_BYTE_GROWTH_ABSOLUTE,
    )


def _count_merged_rectangles(
    quantized: np.ndarray,
    *,
    hard_limit: int,
) -> int:
    """Count merged runs without materializing rectangle tuples or XML nodes."""
    height, width = quantized.shape
    active: set[tuple[int, int, int]] = set()
    completed = 0

    for y in range(height):
        row = quantized[y]
        current: set[tuple[int, int, int]] = set()
        x = 0
        while x < width:
            level = int(row[x])
            start = x
            x += 1
            while x < width and int(row[x]) == level:
                x += 1
            if level > 0:
                current.add((level, start, x))

        completed += len(active - current)
        active = current
        represented = completed + len(active)
        if represented > hard_limit:
            raise RuntimeError(
                "source_alpha_mask_rectangle_budget_exceeded:"
                f"{represented}>{hard_limit}"
            )

    return completed + len(active)


def _preflight(svg_path: Path, source_path: Path) -> dict[str, int] | None:
    before_size = Path(svg_path).stat().st_size
    byte_limit = _byte_limit(before_size)
    available = byte_limit - before_size - _FIXED_MARKUP_OVERHEAD
    if available <= 0:
        raise RuntimeError("source_alpha_mask_byte_budget_unavailable")
    rectangle_limit = max(1, available // _MIN_RECT_BYTES)

    root = SafeET.fromstring(Path(svg_path).read_bytes())
    width, height = _viewbox_size(root)
    scale = min(1.0, _MAX_MASK_SIDE / float(max(width, height)))
    raster_width = max(1, int(round(width * scale)))
    raster_height = max(1, int(round(height * scale)))
    rgba = _rgba_from_source_at_size(
        Path(source_path), (raster_width, raster_height)
    )
    alpha = np.asarray(rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(alpha == 255)):
        return None

    quantized = _quantize_alpha(alpha)
    rectangle_count = _count_merged_rectangles(
        quantized,
        hard_limit=rectangle_limit,
    )

    digits = len(str(max(raster_width, raster_height)))
    rect_upper_bound = 38 + 4 * digits
    group_upper_bound = 96 * min(
        _MAX_ALPHA_LEVELS,
        int(np.unique(quantized).size),
    )
    projected_upper_bound = (
        before_size
        + _FIXED_MARKUP_OVERHEAD
        + rectangle_count * rect_upper_bound
        + group_upper_bound
    )
    if projected_upper_bound > byte_limit:
        raise RuntimeError(
            "source_alpha_mask_byte_budget_exceeded:"
            f"{projected_upper_bound}>{byte_limit}"
        )

    return {
        "preflight_rectangle_limit": int(rectangle_limit),
        "preflight_rectangle_count": int(rectangle_count),
        "preflight_byte_limit": int(byte_limit),
        "preflight_projected_upper_bound": int(projected_upper_bound),
    }


def _create_atomic_backup(svg_path: Path) -> Path:
    descriptor, backup_name = tempfile.mkstemp(
        dir=svg_path.parent,
        prefix=f".{svg_path.name}.",
        suffix=".alpha-rollback.svg",
    )
    os.close(descriptor)
    backup = Path(backup_name)
    try:
        shutil.copy2(svg_path, backup)
    except Exception:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _restore_atomic_backup(backup: Path, svg_path: Path) -> None:
    if not backup.exists():
        raise RuntimeError("source_alpha_mask_rollback_backup_missing")
    os.replace(backup, svg_path)


def wrap_apply_source_alpha_mask(
    original: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Budget-check before allocation and roll back every rejected write."""
    if getattr(original, "__vektoryum_budget_guarded__", False):
        return original

    @wraps(original)
    def guarded(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        preflight = _preflight(target, Path(source_path))
        backup = _create_atomic_backup(target)
        try:
            report = original(target, Path(source_path), mode)
        except BaseException:
            # The wrapped builder atomically replaces `target` before running its
            # render/alpha hard gates. Restore the exact pre-call file for direct
            # callers as well as pipeline callers whenever any later gate rejects.
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)

        if preflight is not None and report.get("applied"):
            report.update(preflight)
        report["rollback_guard"] = "armed_and_committed"
        return report

    guarded.__vektoryum_budget_guarded__ = True
    return guarded
