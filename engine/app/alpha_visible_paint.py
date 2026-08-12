"""Fail-closed visible-RGBA repair for source-alpha final artifacts.

Issue #145 exposed a production gap: the normal source-alpha mask can reproduce the
alpha plane exactly while preserving an incorrect opaque trace/gradient paint. This
module does not touch the source-alpha mask or any global threshold. When an accepted
alpha artifact still fails visible white/black/checker composite fidelity, it replaces
only the paint inside the existing source-alpha mask with deterministic vector regions
derived from the source's visible-color segmentation.

The repair is strictly bounded by the existing parent-relative byte/path/node budgets,
contains no raster embedding, and is published only after stricter alpha/composite
measurement succeeds. Otherwise the pre-alpha parent bytes are restored and the
pipeline fails closed.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.alpha_candidate_knockout import _path_node_counts, _viewbox
from app.alpha_candidate_validation import visible_composite_residual_metrics
from app.alpha_mask_contour import trace_cell_contours
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import (
    alpha_plane_metrics,
    multibackground_pairs,
    render_svg_to_rgba,
    resize_rgba,
)

_SVG_NS = "http://www.w3.org/2000/svg"
_VISIBLE_RESIDUAL_MAX = 0.01
_DE00_P95_MAX = 2.3
_ALPHA_IOU_MIN = 0.999
_ALPHA_MAE_MAX = 0.5 / 255.0
_REPAIR_GRID_MAX_SIDE = 512
_SEGMENTATION_COLORS = 8
_RASTER_TOKENS = ("<image", "data:image", "base64")


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1].split(":")[-1].lower()


def _source_white(rgba: np.ndarray) -> np.ndarray:
    array = np.asarray(rgba, dtype=np.uint8)
    alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(
        array[:, :, :3].astype(np.float32) * alpha
        + 255.0 * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def _de00_p95(source_rgba: np.ndarray, rendered_rgba: np.ndarray) -> float:
    from app.final_artifact_evaluator import _color_metrics  # noqa: PLC0415

    worst = 0.0
    for source_rgb, rendered_rgb in multibackground_pairs(
        source_rgba, rendered_rgba
    ).values():
        worst = max(
            worst,
            float(_color_metrics(source_rgb, rendered_rgb)["de00_p95"]),
        )
    return float(worst)


def _quality_metrics(
    source_rgba: np.ndarray,
    rendered_rgba: np.ndarray,
    *,
    mode: str,
) -> dict[str, Any]:
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.final_artifact_evaluator import _thresholds  # noqa: PLC0415

    source = np.asarray(source_rgba, dtype=np.uint8)
    rendered = np.asarray(rendered_rgba, dtype=np.uint8)
    alpha = alpha_plane_metrics(source[:, :, 3], rendered[:, :, 3])
    composite = visible_composite_residual_metrics(source, rendered)
    thresholds = _thresholds(_MODE_IMAGE_CLASS.get(mode, "clean_logo"), None)
    iou_floor = max(float(thresholds["alpha_iou_min"]), _ALPHA_IOU_MIN)
    mae_ceiling = min(float(thresholds["alpha_mae_max"]), _ALPHA_MAE_MAX)
    de00 = _de00_p95(source, rendered)
    backgrounds = dict(composite.get("visible_composite_backgrounds") or {})
    passed = bool(
        float(alpha["alpha_iou"]) >= iou_floor
        and float(alpha["alpha_mae"]) <= mae_ceiling
        and float(composite["visible_composite_residual"]) <= _VISIBLE_RESIDUAL_MAX
        and de00 <= _DE00_P95_MAX
    )
    return {
        "alpha_iou": float(alpha["alpha_iou"]),
        "alpha_mae": float(alpha["alpha_mae"]),
        "alpha_p95": float(alpha["alpha_p95"]),
        "alpha_max": float(alpha["alpha_max"]),
        "visible_composite_residual": float(
            composite["visible_composite_residual"]
        ),
        "visible_composite_de00_p95": float(de00),
        "visible_composite_backgrounds": backgrounds,
        "alpha_iou_min": float(iou_floor),
        "alpha_mae_max": float(mae_ceiling),
        "visible_composite_residual_max": _VISIBLE_RESIDUAL_MAX,
        "visible_composite_de00_p95_max": _DE00_P95_MAX,
        "passed": passed,
    }


def _render_at_grid(
    svg_path: Path,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    root = ET.parse(svg_path).getroot()
    _vx, _vy, view_width, view_height = _viewbox(root)
    scale = min(
        1.0, _REPAIR_GRID_MAX_SIDE / float(max(view_width, view_height))
    )
    width = max(1, int(round(view_width * scale)))
    height = max(1, int(round(view_height * scale)))
    source = _rgba_from_source_at_size(Path(source_path), (width, height))
    rendered = render_svg_to_rgba(Path(svg_path), width, height)
    if rendered is None:
        raise RuntimeError("source_alpha_visible_paint_render_unmeasured")
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)
    return source, np.asarray(rendered, dtype=np.uint8), (width, height)


def _visible_segmentation(
    source_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical visible labels, visible palette and straight RGB paint."""
    from app.graph_source import canonical_segmentation  # noqa: PLC0415

    source = np.asarray(source_rgba, dtype=np.uint8)
    visible = _source_white(source)
    unique_visible = int(np.unique(visible.reshape(-1, 3), axis=0).shape[0])
    k = max(1, min(_SEGMENTATION_COLORS, unique_visible))
    labels, visible_palette = canonical_segmentation(visible, k=k)

    straight_palette = np.zeros((k, 3), dtype=np.uint8)
    alpha = source[:, :, 3]
    rgb = source[:, :, :3]
    for label in range(k):
        selected = (labels == label) & (alpha > 0)
        if not bool(np.any(selected)):
            continue
        weights = alpha[selected].astype(np.float64)
        values = rgb[selected].astype(np.float64)
        representative = np.average(values, axis=0, weights=weights)
        straight_palette[label] = np.rint(representative).clip(0, 255).astype(
            np.uint8
        )
    return labels.astype(np.uint8), visible_palette, straight_palette


def _loop_path_data(loops: list[list[tuple[int, int]]]) -> str:
    parts: list[str] = []
    for corners in loops:
        if len(corners) < 3:
            continue
        x0, y0 = corners[0]
        previous_x, previous_y = x0, y0
        commands = [f"M{x0} {y0}"]
        for x, y in corners[1:]:
            if y == previous_y:
                commands.append(f"H{x}")
            elif x == previous_x:
                commands.append(f"V{y}")
            else:
                commands.append(f"L{x} {y}")
            previous_x, previous_y = x, y
        commands.append("Z")
        parts.append("".join(commands))
    return "".join(parts)


def _source_paint_group(
    root: ET.Element,
    source_rgba: np.ndarray,
    *,
    grid_width: int,
    grid_height: int,
) -> tuple[ET.Element, dict[str, Any]]:
    labels, visible_palette, straight_palette = _visible_segmentation(source_rgba)
    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    view_x, view_y, view_width, view_height = _viewbox(root)
    group = ET.Element(
        qname("g"),
        {
            "data-vektoryum-visible-alpha-paint": "source-segmentation-v1",
            "transform": (
                f"translate({view_x:.12g} {view_y:.12g}) "
                f"scale({view_width / float(grid_width):.12g} "
                f"{view_height / float(grid_height):.12g})"
            ),
        },
    )
    alpha_positive = np.asarray(source_rgba[:, :, 3], dtype=np.uint8) > 0
    path_count = 0
    loop_count = 0
    used_labels: list[int] = []
    for label in range(len(straight_palette)):
        selected = (labels == label) & alpha_positive
        if not bool(np.any(selected)):
            continue
        loops = trace_cell_contours(selected)
        path_data = _loop_path_data(loops)
        if not path_data:
            continue
        color = straight_palette[label]
        ET.SubElement(
            group,
            qname("path"),
            {
                "d": path_data,
                "fill": "#{:02x}{:02x}{:02x}".format(
                    int(color[0]), int(color[1]), int(color[2])
                ),
                "fill-rule": "evenodd",
            },
        )
        path_count += 1
        loop_count += len(loops)
        used_labels.append(label)
    if path_count <= 0:
        raise RuntimeError("source_alpha_visible_paint_empty")
    return group, {
        "visible_segmentation_palette_size": int(len(visible_palette)),
        "visible_paint_path_count": int(path_count),
        "visible_paint_loop_count": int(loop_count),
        "visible_paint_used_label_count": int(len(used_labels)),
    }


def _mask_application_group(root: ET.Element) -> ET.Element:
    matches = [
        element
        for element in root.iter()
        if _local_name(element.tag) == "g"
        and element.get("data-vektoryum-source-alpha") == "vector-mask-v1"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "source_alpha_visible_paint_mask_application_unavailable:"
            f"{len(matches)}"
        )
    return matches[0]


def _serialized(root: ET.Element) -> bytes:
    ET.register_namespace("", _SVG_NS)
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )


def _contains_raster(data: bytes) -> bool:
    text = data.decode("utf-8", errors="ignore").lower()
    return any(token in text for token in _RASTER_TOKENS)


def _write_temp(target: Path, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=Path(target).parent,
        prefix=f".{Path(target).name}.",
        suffix=".visible-alpha.svg",
    )
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_bytes(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def repair_visible_alpha_paint(
    svg_path: Path,
    source_path: Path,
    mode: str,
    *,
    parent_bytes: bytes,
) -> dict[str, Any]:
    """Repair visible paint under an already accepted source-alpha mask."""
    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415

    target = Path(svg_path)
    source = Path(source_path)
    accepted_bytes = target.read_bytes()
    source_eval, rendered_before, (grid_width, grid_height) = _render_at_grid(
        target, source
    )
    before_quality = _quality_metrics(source_eval, rendered_before, mode=mode)
    if before_quality["passed"]:
        return {
            "visible_paint_repair_status": "not_required",
            "visible_paint_repair_applied": False,
            "visible_paint_before": before_quality,
            "visible_paint_after": before_quality,
            "raster_embed_count": 0,
        }

    try:
        parent_root = ET.fromstring(parent_bytes)
        limits = _journal_limits(parent_root, len(parent_bytes))
        candidate_root = ET.fromstring(accepted_bytes)
        wrapper = _mask_application_group(candidate_root)
        for child in list(wrapper):
            wrapper.remove(child)
        paint, paint_report = _source_paint_group(
            candidate_root,
            source_eval,
            grid_width=grid_width,
            grid_height=grid_height,
        )
        wrapper.append(paint)
        candidate_bytes = _serialized(candidate_root)

        if _contains_raster(candidate_bytes):
            raise RuntimeError("source_alpha_visible_paint_raster_embed_rejected")
        candidate_counts = _path_node_counts(candidate_root)
        if len(candidate_bytes) > int(limits["byte_limit"]):
            raise RuntimeError(
                "source_alpha_visible_paint_byte_budget_rejected:"
                f"{len(candidate_bytes)}>{limits['byte_limit']}"
            )
        if int(candidate_counts[0]) > int(limits["path_limit"]):
            raise RuntimeError(
                "source_alpha_visible_paint_path_budget_rejected:"
                f"{candidate_counts[0]}>{limits['path_limit']}"
            )
        if int(candidate_counts[1]) > int(limits["node_limit"]):
            raise RuntimeError(
                "source_alpha_visible_paint_node_budget_rejected:"
                f"{candidate_counts[1]}>{limits['node_limit']}"
            )

        temporary = _write_temp(target, candidate_bytes)
        try:
            source_check, rendered_after, _grid = _render_at_grid(temporary, source)
            after_quality = _quality_metrics(source_check, rendered_after, mode=mode)
            if not after_quality["passed"]:
                raise RuntimeError(
                    "source_alpha_visible_paint_quality_rejected:"
                    f"residual={after_quality['visible_composite_residual']:.6f};"
                    f"de00_p95={after_quality['visible_composite_de00_p95']:.6f};"
                    f"alpha_iou={after_quality['alpha_iou']:.6f};"
                    f"alpha_mae={after_quality['alpha_mae']:.6f}"
                )

            from app.alpha_candidate_knockout import _source_rgb_on_white  # noqa: PLC0415
            from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
            from app.final_artifact_evaluator import evaluate_final_svg  # noqa: PLC0415

            evaluation = evaluate_final_svg(
                temporary,
                _source_rgb_on_white(source_check),
                source_alpha=source_check[:, :, 3],
                image_class=_MODE_IMAGE_CLASS.get(mode, "clean_logo"),
                required_metrics={"alpha_fidelity"},
            )
            appearance_codes = [
                code
                for code in evaluation.hard_fail_codes
                if code.startswith(
                    ("alpha_white_", "alpha_black_", "alpha_checker_")
                )
            ]
            if appearance_codes:
                raise RuntimeError(
                    "source_alpha_visible_paint_multibackground_rejected:"
                    + ",".join(appearance_codes)
                )

            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except Exception:
        target.write_bytes(parent_bytes)
        raise

    after_bytes = target.read_bytes()
    return {
        "visible_paint_repair_status": "accepted",
        "visible_paint_repair_applied": True,
        "visible_paint_schema": "issue145-source-segmentation-paint-v1",
        "visible_paint_before": before_quality,
        "visible_paint_after": after_quality,
        "visible_paint_before_alpha_artifact_bytes": int(len(accepted_bytes)),
        "visible_paint_after_bytes": int(len(after_bytes)),
        "visible_paint_parent_bytes": int(len(parent_bytes)),
        "visible_paint_parent_path_count": int(limits["parent_path_count"]),
        "visible_paint_parent_node_count": int(limits["parent_node_count"]),
        "visible_paint_path_limit": int(limits["path_limit"]),
        "visible_paint_node_limit": int(limits["node_limit"]),
        "visible_paint_byte_limit": int(limits["byte_limit"]),
        "visible_paint_after_path_count": int(candidate_counts[0]),
        "visible_paint_after_node_count": int(candidate_counts[1]),
        "visible_paint_after_sha256": hashlib.sha256(after_bytes).hexdigest(),
        "raster_embed_count": 0,
        **paint_report,
    }


def make_visible_alpha_paint_repair(
    original_apply: Callable[[Path, Path, str], dict[str, Any]],
) -> Callable[[Path, Path, str], dict[str, Any]]:
    """Wrap the assembled alpha finalizer with strict visible-RGBA publication."""
    if getattr(original_apply, "__vektoryum_visible_alpha_paint_repair__", False):
        return original_apply

    @wraps(original_apply)
    def visible_repair(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        target = Path(svg_path)
        parent_bytes = target.read_bytes()
        report = original_apply(target, Path(source_path), mode)
        if not isinstance(report, dict) or report.get("status") == "not_applicable":
            return report
        try:
            visible = repair_visible_alpha_paint(
                target,
                Path(source_path),
                mode,
                parent_bytes=parent_bytes,
            )
        except Exception:
            if target.read_bytes() != parent_bytes:
                target.write_bytes(parent_bytes)
            raise
        result = dict(report)
        result.update(visible)
        result["after_byte_size"] = int(target.stat().st_size)
        result["after_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        return result

    visible_repair.__vektoryum_visible_alpha_paint_repair__ = True
    return visible_repair
