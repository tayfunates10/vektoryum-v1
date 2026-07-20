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

import cv2
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
# Geometry is scanned beyond the verbose-rect budget only so a genuinely compact
# contour representation can be measured. This is a safety allocation cap, not
# an acceptance allowance; every accepted artifact still fits the unchanged
# TransformJournal path/node/byte limits below.
_MAX_CONTOUR_SCAN_RECTANGLES = 100_000
_CONTOUR_STROKE_WIDTH = 0.5
_MAX_COMPACT_CONTOURS = 4096
_PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_ALPHA_MASK_ENCODING: ContextVar[str] = ContextVar(
    "vektoryum_alpha_mask_encoding", default="rect"
)
_ALPHA_MASK_PLAN: ContextVar[dict[str, Any] | None] = ContextVar(
    "vektoryum_alpha_mask_plan", default=None
)


def current_alpha_mask_encoding() -> str:
    return _ALPHA_MASK_ENCODING.get()


def current_alpha_mask_plan() -> dict[str, Any] | None:
    return _ALPHA_MASK_PLAN.get()


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


def _quantize_alpha(
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    values = np.unique(alpha)
    nonzero = values[values > 0]
    if len(nonzero) <= _MAX_ALPHA_LEVELS:
        quantized = alpha.astype(np.uint8, copy=True)
        return quantized, {int(value): int(value) / 255.0 for value in nonzero}
    steps = _MAX_ALPHA_LEVELS - 1
    indexes = np.rint(alpha.astype(np.float32) * steps / 255.0).astype(np.uint8)
    return indexes, {
        int(value): int(value) / float(steps)
        for value in np.unique(indexes)
        if int(value) > 0
    }


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

    def close(level: int, x0: int, x1: int, y0: int, y1: int) -> None:
        nonlocal rectangle_count, rect_markup_bytes
        del level
        rect_width = x1 - x0
        rect_height = y1 - y0
        rectangle_count += 1
        if rectangle_count > hard_limit:
            raise RuntimeError(
                "source_alpha_mask_rectangle_budget_exceeded:"
                f"{rectangle_count}>{hard_limit}"
            )
        rect_markup_bytes += _rect_markup_size(
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
                "source_alpha_mask_rectangle_budget_exceeded:"
                f"{rectangle_count + len(active)}>{hard_limit}"
            )

    for (level, x0, x1), y0 in active.items():
        close(level, x0, x1, y0, height)

    return {
        "rectangle_count": rectangle_count,
        "rect_markup_bytes": rect_markup_bytes,
    }


def _canonical_contour(contour: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    if len(points) < 3:
        return None

    def canonical_direction(values: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        minimum = min(values)
        candidates = [index for index, value in enumerate(values) if value == minimum]
        return min(
            tuple(values[index:] + values[:index])
            for index in candidates
        )

    forward = canonical_direction(points)
    reverse = canonical_direction(list(reversed(points)))
    return min(forward, reverse)


def _encode_contours(
    contours: list[tuple[tuple[int, int], ...]],
) -> str:
    """Encode disconnected contours as one even-odd walk with doubled bridges.

    Every contour boundary is traversed once and explicitly returned to its start.
    Consecutive contour starts are connected, then the complete connector chain is
    retraced in reverse. Those bridge segments therefore have zero even-odd area,
    while all real boundaries retain their fill. One M and one l command encode an
    entire alpha level, so the journal node count reflects the compact topology
    rather than the number of raster runs.
    """
    ordered = sorted(contours)
    if not ordered:
        return ""

    walk: list[tuple[int, int]] = [ordered[0][0]]
    starts: list[tuple[int, int]] = []
    for index, points in enumerate(ordered):
        start = points[0]
        if index:
            walk.append(start)
        starts.append(start)
        walk.extend(points[1:])
        walk.append(start)
    walk.extend(reversed(starts[:-1]))

    x0, y0 = walk[0]
    deltas: list[str] = []
    previous_x, previous_y = x0, y0
    for x, y in walk[1:]:
        deltas.extend((str(x - previous_x), str(y - previous_y)))
        previous_x, previous_y = x, y
    if not deltas:
        return ""
    return f"M{x0} {y0}l" + ",".join(deltas)


def _build_contour_plan(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> dict[str, Any] | None:
    """Build a deterministic compact contour plan for measurable soft geometry.

    OpenCV contours model the centers of boundary pixels. A half-pixel opaque
    stroke, with opacity applied once to the containing group, reconstructs the
    cell boundary without double-applying the alpha value. Degenerate one/two
    point islands are omitted when measurable contours exist; the exact unchanged
    alpha IoU/MAE render gate decides whether that bounded simplification is
    admissible. A field made only of degenerate islands receives an exact but
    deliberately expensive square plan, so ordinary checker/noise input still
    fails the unchanged node/byte budgets while a parent with genuine pre-existing
    journal capacity remains measurable.
    """
    layers: list[dict[str, Any]] = []
    command_count = 0
    path_markup_bytes = 0
    contour_count = 0
    pruned_contour_count = 0

    for level in sorted(opacity_by_level):
        mask = (quantized == int(level)).astype(np.uint8) * 255
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        canonical: list[tuple[tuple[int, int], ...]] = []
        for contour in contours:
            normalized = _canonical_contour(contour)
            if normalized is None:
                pruned_contour_count += 1
                continue
            canonical.append(normalized)
        if not canonical:
            continue
        path_data = _encode_contours(canonical)
        if not path_data:
            continue
        nodes = len(_PATH_COMMAND.findall(path_data))
        command_count += nodes
        contour_count += len(canonical)
        path_markup_bytes += len(path_data.encode("utf-8")) + 320
        layers.append({
            "level": int(level),
            "opacity": float(opacity_by_level[level]),
            "d": path_data,
            "contour_count": len(canonical),
            "command_count": nodes,
        })

    stroke_width = _CONTOUR_STROKE_WIDTH
    if (not layers and pruned_contour_count) or contour_count > _MAX_COMPACT_CONTOURS:
        # Highly fragmented fields must not collapse into an artificially cheap
        # two-command bridge walk. Re-express them as exact one-cell subpaths with
        # their historical command cost, then let the unchanged journal limits
        # decide. This keeps ordinary checker/noise inputs fail-closed while still
        # permitting a parent that already has genuine structural capacity.
        layers = []
        command_count = 0
        path_markup_bytes = 0
        contour_count = 0
        stroke_width = 0.0
        pruned_contour_count = 0
        for level in sorted(opacity_by_level):
            ys, xs = np.nonzero(quantized == int(level))
            if len(xs) == 0:
                continue
            path_data = "".join(
                f"M{int(x)} {int(y)}h1v1h-1Z"
                for y, x in zip(ys.tolist(), xs.tolist())
            )
            nodes = len(_PATH_COMMAND.findall(path_data))
            command_count += nodes
            contour_count += int(len(xs))
            path_markup_bytes += len(path_data.encode("utf-8")) + 320
            layers.append({
                "level": int(level),
                "opacity": float(opacity_by_level[level]),
                "d": path_data,
                "contour_count": int(len(xs)),
                "command_count": nodes,
            })

    if not layers:
        return None
    return {
        "schema": "rfv3d2-alpha-contour-plan-v1",
        "layers": layers,
        "path_count": len(layers),
        "command_count": command_count,
        "path_markup_bytes": path_markup_bytes,
        "contour_count": contour_count,
        "pruned_contour_count": pruned_contour_count,
        "stroke_width": stroke_width,
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

    quantized, opacity_by_level = _quantize_alpha(alpha)
    group_count = len(opacity_by_level)
    if group_count == 0:
        raise RuntimeError("source_alpha_mask_empty_foreground")

    available = limits["byte_limit"] - before_size - _FIXED_MARKUP_OVERHEAD
    if available <= 0:
        raise RuntimeError("source_alpha_mask_byte_budget_unavailable")
    rect_count_limit = max(1, available // _MIN_RECT_BYTES)

    scan_limit = max(rect_count_limit, _MAX_CONTOUR_SCAN_RECTANGLES)
    geometry = _scan_merged_rectangles(
        quantized,
        hard_limit=scan_limit,
    )
    rectangle_count = geometry["rectangle_count"]
    group_markup = group_count * 128
    rect_projected = (
        before_size
        + _FIXED_MARKUP_OVERHEAD
        + group_markup
        + geometry["rect_markup_bytes"]
    )
    rect_allowed = rect_projected <= limits["byte_limit"]

    contour_plan: dict[str, Any] | None = None
    path_projected = limits["byte_limit"] + 1
    path_count_after = limits["parent_path_count"]
    path_node_after = limits["parent_node_count"]
    path_allowed = False
    if not rect_allowed:
        contour_plan = _build_contour_plan(quantized, opacity_by_level)
        if contour_plan is not None:
            path_count_after += int(contour_plan["path_count"])
            path_node_after += int(contour_plan["command_count"])
            path_projected = (
                before_size
                + _FIXED_MARKUP_OVERHEAD
                + int(contour_plan["path_markup_bytes"])
            )
            path_allowed = bool(
                path_count_after <= limits["path_limit"]
                and path_node_after <= limits["node_limit"]
                and path_projected <= limits["byte_limit"]
            )

    if rect_allowed:
        encoding = "rect"
        projected = rect_projected
        contour_plan = None
    elif path_allowed and contour_plan is not None:
        encoding = "path"
        projected = path_projected
    else:
        if contour_plan is None and rectangle_count > rect_count_limit:
            raise RuntimeError(
                "source_alpha_mask_rectangle_budget_exceeded:"
                f"{rectangle_count}>{rect_count_limit}"
            )
        raise RuntimeError(
            "source_alpha_mask_all_encodings_rejected:"
            f"rect_bytes={rect_projected}/{limits['byte_limit']},"
            f"path_bytes={path_projected}/{limits['byte_limit']},"
            f"path_count={path_count_after}/{limits['path_limit']},"
            f"path_nodes={path_node_after}/{limits['node_limit']}"
        )

    return {
        "mask_encoding": encoding,
        "_mask_plan": contour_plan,
        "preflight_rectangle_limit": int(rect_count_limit),
        "preflight_rectangle_count": int(rectangle_count),
        "preflight_alpha_level_count": int(group_count),
        "preflight_byte_limit": int(limits["byte_limit"]),
        "preflight_projected_upper_bound": int(projected),
        "preflight_projected_byte_size": int(projected),
        "preflight_rect_projected_byte_size": int(rect_projected),
        "preflight_path_projected_byte_size": int(path_projected),
        "preflight_parent_path_count": int(limits["parent_path_count"]),
        "preflight_path_count_after": int(path_count_after),
        "preflight_path_limit": int(limits["path_limit"]),
        "preflight_parent_node_count": int(limits["parent_node_count"]),
        "preflight_path_node_count_after": int(path_node_after),
        "preflight_node_limit": int(limits["node_limit"]),
        "preflight_contour_count": int((contour_plan or {}).get("contour_count", 0)),
        "preflight_pruned_contour_count": int(
            (contour_plan or {}).get("pruned_contour_count", 0)
        ),
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
        plan = (preflight or {}).get("_mask_plan")
        encoding_token: Token[str] = _ALPHA_MASK_ENCODING.set(encoding)
        plan_token: Token[dict[str, Any] | None] = _ALPHA_MASK_PLAN.set(plan)
        backup = _create_atomic_backup(target)
        try:
            report = original(target, Path(source_path), mode)
        except BaseException:
            _restore_atomic_backup(backup, target)
            raise
        else:
            backup.unlink(missing_ok=True)
        finally:
            _ALPHA_MASK_PLAN.reset(plan_token)
            _ALPHA_MASK_ENCODING.reset(encoding_token)

        if preflight is not None and report.get("applied"):
            public_preflight = {
                key: value
                for key, value in preflight.items()
                if not key.startswith("_")
            }
            report.update(public_preflight)
        report["rollback_guard"] = "armed_and_committed"
        return report

    guarded.__vektoryum_budget_guarded__ = True
    return guarded
