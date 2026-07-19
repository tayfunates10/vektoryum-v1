"""Pre-construction, adaptive-encoding and rollback gates for alpha masks."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextvars import ContextVar, Token
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from defusedxml import ElementTree as SafeET

from app.alpha_preprocess import _rgba_from_source_at_size

_MAX_ALPHA_LEVELS = 128
_MAX_MASK_SIDE = 1600
_JOURNAL_PATH_GROWTH_FACTOR = 4
_JOURNAL_PATH_GROWTH_ABSOLUTE = 500
_JOURNAL_NODE_GROWTH_FACTOR = 4
_JOURNAL_NODE_GROWTH_ABSOLUTE = 2500
_JOURNAL_BYTE_GROWTH_FACTOR = 3
_JOURNAL_BYTE_GROWTH_ABSOLUTE = 250_000
_FIXED_MARKUP_OVERHEAD = 4096
_MIN_RECT_BYTES = 40
_PATH_COMMANDS_PER_RECTANGLE = 5
_PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_ALPHA_MASK_ENCODING: ContextVar[str] = ContextVar(
    "vektoryum_alpha_mask_encoding", default="rect"
)


def current_alpha_mask_encoding() -> str:
    return _ALPHA_MASK_ENCODING.get()


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1] if "}" in name else name


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


def _journal_limits(root: Any, before_size: int) -> dict[str, int]:
    paths = [
        element
        for element in root.iter()
        if _local_name(str(element.tag)).lower() == "path"
    ]
    path_count = len(paths)
    node_count = sum(
        len(_PATH_COMMAND.findall(str(path.get("d") or ""))) for path in paths
    )
    bp = max(1, path_count)
    bn = max(1, node_count)
    bb = max(1, before_size)
    return {
        "parent_path_count": path_count,
        "parent_node_count": node_count,
        "path_limit": max(
            bp * _JOURNAL_PATH_GROWTH_FACTOR,
            bp + _JOURNAL_PATH_GROWTH_ABSOLUTE,
        ),
        "node_limit": max(
            bn * _JOURNAL_NODE_GROWTH_FACTOR,
            bn + _JOURNAL_NODE_GROWTH_ABSOLUTE,
        ),
        "byte_limit": max(
            bb * _JOURNAL_BYTE_GROWTH_FACTOR,
            bb + _JOURNAL_BYTE_GROWTH_ABSOLUTE,
        ),
    }


def _rect_markup_size(x: int, y: int, width: int, height: int) -> int:
    return len(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" />'
    )


def _path_command_size(x: int, y: int, width: int, height: int) -> int:
    return len(f"M{x} {y}h{width}v{height}h-{width}Z")


def _scan_merged_rectangles(
    quantized: np.ndarray,
    *,
    hard_limit: int,
) -> dict[str, int]:
    """Measure merged geometry without retaining rectangle tuples or XML nodes."""
    height, width = quantized.shape
    active: dict[tuple[int, int, int], int] = {}
    rectangle_count = 0
    rect_markup_bytes = 0
    path_command_bytes = 0

    def close(level: int, x0: int, x1: int, y0: int, y1: int) -> None:
        nonlocal rectangle_count, rect_markup_bytes, path_command_bytes
        del level
        rect_width = x1 - x0
        rect_height = y1 - y0
        rectangle_count += 1
        if rectangle_count > hard_limit:
            raise RuntimeError(
                "source_alpha_mask_geometry_budget_exceeded:"
                f"{rectangle_count}>{hard_limit}"
            )
        rect_markup_bytes += _rect_markup_size(
            x0, y0, rect_width, rect_height
        )
        path_command_bytes += _path_command_size(
            x0, y0, rect_width, rect_height
        )

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

        for key in list(active):
            if key in current:
                continue
            level, x0, x1 = key
            close(level, x0, x1, active.pop(key), y)
        for key in current:
            active.setdefault(key, y)

        if rectangle_count + len(active) > hard_limit:
            raise RuntimeError(
                "source_alpha_mask_geometry_budget_exceeded:"
                f"{rectangle_count + len(active)}>{hard_limit}"
            )

    for (level, x0, x1), y0 in active.items():
        close(level, x0, x1, y0, height)

    return {
        "rectangle_count": rectangle_count,
        "rect_markup_bytes": rect_markup_bytes,
        "path_command_bytes": path_command_bytes,
    }


def _preflight(svg_path: Path, source_path: Path) -> dict[str, Any] | None:
    before_size = Path(svg_path).stat().st_size
    root = SafeET.fromstring(Path(svg_path).read_bytes())
    limits = _journal_limits(root, before_size)
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
    nonzero_levels = {int(value) for value in np.unique(quantized) if int(value) > 0}
    group_count = len(nonzero_levels)
    if group_count == 0:
        raise RuntimeError("source_alpha_mask_empty_foreground")

    available = limits["byte_limit"] - before_size - _FIXED_MARKUP_OVERHEAD
    if available <= 0:
        raise RuntimeError("source_alpha_mask_byte_budget_unavailable")
    rect_count_limit = max(1, available // _MIN_RECT_BYTES)

    path_count_after = limits["parent_path_count"] + group_count
    if path_count_after <= limits["path_limit"]:
        path_node_capacity = max(
            0,
            (limits["node_limit"] - limits["parent_node_count"])
            // _PATH_COMMANDS_PER_RECTANGLE,
        )
    else:
        path_node_capacity = 0
    hard_geometry_limit = max(rect_count_limit, path_node_capacity)
    if hard_geometry_limit <= 0:
        raise RuntimeError("source_alpha_mask_no_admissible_vector_encoding")

    geometry = _scan_merged_rectangles(
        quantized,
        hard_limit=hard_geometry_limit,
    )
    rectangle_count = geometry["rectangle_count"]
    group_markup = group_count * 128
    rect_projected = (
        before_size
        + _FIXED_MARKUP_OVERHEAD
        + group_markup
        + geometry["rect_markup_bytes"]
    )
    path_spaces = max(0, rectangle_count - group_count)
    path_projected = (
        before_size
        + _FIXED_MARKUP_OVERHEAD
        + group_markup
        + group_count * 32
        + geometry["path_command_bytes"]
        + path_spaces
    )
    path_node_after = (
        limits["parent_node_count"]
        + rectangle_count * _PATH_COMMANDS_PER_RECTANGLE
    )

    rect_allowed = rect_projected <= limits["byte_limit"]
    path_allowed = bool(
        path_count_after <= limits["path_limit"]
        and path_node_after <= limits["node_limit"]
        and path_projected <= limits["byte_limit"]
    )
    if rect_allowed:
        encoding = "rect"
        projected = rect_projected
    elif path_allowed:
        encoding = "path"
        projected = path_projected
    else:
        raise RuntimeError(
            "source_alpha_mask_all_encodings_rejected:"
            f"rect_bytes={rect_projected}/{limits['byte_limit']},"
            f"path_bytes={path_projected}/{limits['byte_limit']},"
            f"path_count={path_count_after}/{limits['path_limit']},"
            f"path_nodes={path_node_after}/{limits['node_limit']}"
        )

    return {
        "mask_encoding": encoding,
        "preflight_rectangle_count": int(rectangle_count),
        "preflight_alpha_level_count": int(group_count),
        "preflight_byte_limit": int(limits["byte_limit"]),
        "preflight_projected_byte_size": int(projected),
        "preflight_rect_projected_byte_size": int(rect_projected),
        "preflight_path_projected_byte_size": int(path_projected),
        "preflight_parent_path_count": int(limits["parent_path_count"]),
        "preflight_path_count_after": int(path_count_after),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_parent_node_count": int(limits["parent_node_count"]),
        "preflight_path_node_count_after": int(path_node_after),
        "preflight_node_limit": int(limits["node_limit"]),
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
    """Choose an admissible encoding and roll back every rejected write."""
    if getattr(original, "__vektoryum_budget_guarded__", False):
        return original

    @wraps(original)
    def guarded(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        preflight = _preflight(target, Path(source_path))
        encoding = str((preflight or {}).get("mask_encoding") or "rect")
        token: Token[str] = _ALPHA_MASK_ENCODING.set(encoding)
        backup = _create_atomic_backup(target)
        try:
            report = original(target, Path(source_path), mode)
        except BaseException:
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)
        finally:
            _ALPHA_MASK_ENCODING.reset(token)

        if preflight is not None and report.get("applied"):
            report.update(preflight)
        report["rollback_guard"] = "armed_and_committed"
        return report

    guarded.__vektoryum_budget_guarded__ = True
    return guarded
