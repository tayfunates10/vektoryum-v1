"""Scale-stable painter-mask alpha reconstruction (final RFV-3D2 fallback).

The renderer-native 512 cell grid reproduces source alpha exactly on the 512
measurement grid, but its constant 2x2-blocky fade ribbon fragments the real
TransformJournal's native-resolution topology refinement (live class_reklam:
components 91->379, holes 15->217). A source-native grid fixes topology but
per-cell tilings then lose alpha under the renderer's 512 coverage compositing.

This module resolves both scales with one deterministic, vector-only geometry:

- alpha is sampled on the source-native raster (bounded), quantized through the
  unchanged ``_quantize_alpha`` contract;
- every alpha level INCLUDING zero becomes exact union contours (shared cell
  edges cancel), so enclosed fully-transparent lakes are explicitly repainted;
- all loops are painted once, largest-area first (containers before contained,
  equal areas resolved darker-first), as opaque grayscale ``<polygon>`` fills
  over an opaque black base inside a single luminance ``<mask>``;
- opaque-over-opaque compositing makes boundary pixels a linear coverage mix of
  neighbouring level grays at every render scale, so junction error scales with
  the local alpha gradient exactly like real image resampling;
- the candidate's paths, path data, colors and identity stay untouched: paint
  moves into a reusable group, receives the smallest admissible same-color
  support stroke behind its fills, and is drawn once through the mask.

Validation is fail-closed with unchanged thresholds: the native-grid render
must reproduce the staged alpha plane, the INTER_AREA-downscaled native render
must match the identically downscaled source on the bounded evaluation grid,
FinalArtifactEvaluator's alpha-plane codes are enforced against the full-
resolution source, and candidate path/node counts must be preserved. RGB
appearance and every parent-relative regression stay owned by the real
TransformJournal stage that follows.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.alpha_artwork_identity import (
    ROLE_ARTWORK_CONTAINER,
    ROLE_CANVAS_UNDERPAINT,
    ROLE_MASK_APPLICATION,
    ROLE_MASK_DEFINITION,
    alpha_transaction_id,
    artwork_fingerprint,
    tag_transform_node,
)
from app.alpha_mask_contour import loop_signed_area, trace_cell_contours
from app.alpha_preprocess import _rgba_from_source_at_size
from app.source_truth import alpha_plane_metrics, render_svg_to_rgba, resize_rgba

_SVG_NS = "http://www.w3.org/2000/svg"
_PAINTER_GRID_MAX_SIDE = 1600
_PAINTER_STROKE_PIXELS = (1.0, 1.5, 2.0, 3.0)
# A proven full-canvas underpaint supplies alpha support but can expose its own
# comparison colour through an uncovered one-pixel AA fringe. The measured
# existing 1.5px candidate is the smallest support width that preserves the
# unchanged source-topology sentinel; this is an encoding preflight, not a
# quality-threshold change.
_PAINTER_UNDERPAINT_MIN_STROKE_PIXELS = 1.5
_PAINTER_EVAL_SIDE = 512.0
_ALPHA_PLANE_FAILURE_CODES = {"alpha_iou_below_min", "alpha_mae_above_max"}


def _painter_loops(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
) -> list[tuple[float, list[tuple[int, int]], int]]:
    """Deterministic painter order: (area desc, gray asc, start corner)."""
    height, width = quantized.shape
    loops: list[tuple[float, list[tuple[int, int]], int]] = []
    for level in sorted(int(value) for value in np.unique(quantized)):
        for corners in trace_cell_contours(quantized == level):
            corner_x, corner_y = corners[0]
            sampled = int(
                quantized[min(corner_y, height - 1), min(corner_x, width - 1)]
            )
            gray = (
                int(round(float(opacity_by_level.get(sampled, 0.0)) * 255))
                if sampled > 0
                else 0
            )
            loops.append((abs(loop_signed_area(corners)), corners, gray))
    loops.sort(key=lambda item: (-item[0], item[2], item[1][0][1], item[1][0][0]))
    return loops


def _painter_polygon_children(
    loops: list[tuple[float, list[tuple[int, int]], int]],
    qname,
) -> list[ET.Element]:
    """One opaque grayscale ``<polygon>`` per contour loop (overpaint order)."""
    children: list[ET.Element] = []
    for _area, corners, gray in loops:
        children.append(
            ET.Element(
                qname("polygon"),
                {
                    "points": " ".join(f"{x},{y}" for x, y in corners),
                    "fill": f"rgb({gray},{gray},{gray})",
                },
            )
        )
    return children


def _painter_rect_children(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    qname,
) -> tuple[list[ET.Element], int]:
    """Run-length ``<rect>`` decomposition of each level's cells, grouped by gray.

    Levels partition the grid, so painting each level's exact merged rectangles at
    its own gray over the opaque black base reproduces the quantized alpha plane —
    render-identical to the polygon overpaint but often far fewer bytes when a mask
    has tens of thousands of jagged contour loops. ``<rect>`` (like ``<polygon>``)
    is not counted by ``path_count``, so the candidate identity invariant holds.
    The level-0 (transparent) cells are left to the black base, exactly as the
    polygon encoding paints them black.
    """
    from app.alpha_svg_mask import _merged_rectangles_by_level  # noqa: PLC0415

    children: list[ET.Element] = []
    rect_count = 0
    rectangles = _merged_rectangles_by_level(quantized)
    for level in sorted(rectangles):
        level_rectangles = rectangles[level]
        if not level_rectangles:
            continue
        gray = (
            int(round(float(opacity_by_level.get(level, 0.0)) * 255))
            if level > 0
            else 0
        )
        group = ET.Element(qname("g"), {"fill": f"rgb({gray},{gray},{gray})"})
        for x, y, width, height in level_rectangles:
            if width <= 0 or height <= 0:
                continue
            ET.SubElement(
                group,
                qname("rect"),
                {
                    "x": str(x),
                    "y": str(y),
                    "width": str(width),
                    "height": str(height),
                },
            )
        if len(group):
            children.append(group)
            rect_count += len(group)
    return children, rect_count


def _rectilinear_subpaths(
    loops: list[list[tuple[int, int]]],
) -> tuple[str, int]:
    """Hücre-kenarı döngülerini kompakt dikdörtgensel `d` (relatif h/v) olarak kodla.

    ``trace_cell_contours`` hücre KÖŞELERİni verir (ardışık köşeler eksen-hizalı),
    bu yüzden dolgu piksel kenarına tam oturur (polygon kodlaması gibi alfa-tam) —
    piksel MERKEZİni izleyen (yarım-piksel eksik dolan) kontur kodlayıcısından
    farkı budur. Her döngü ``M x0 y0`` + relatif ``h``/``v`` + ``Z``; tek path
    içinde birleşir, delikler even-odd ile çözülür. Tek koordinatlı h/v komutları
    polygon'un "x,y" çiftlerinin ~yarısı byte tutar.
    """
    parts: list[str] = []
    nodes = 0
    for corners in loops:
        if len(corners) < 3:
            continue
        x0, y0 = corners[0]
        segment = [f"M{x0} {y0}"]
        previous_x, previous_y = x0, y0
        for x, y in corners[1:]:
            if y == previous_y and x != previous_x:
                segment.append(f"h{x - previous_x}")
            elif x == previous_x and y != previous_y:
                segment.append(f"v{y - previous_y}")
            else:
                segment.append(f"L{x} {y}")
            previous_x, previous_y = x, y
        segment.append("Z")
        parts.append("".join(segment))
        nodes += len(corners) + 1
    return "".join(parts), nodes


def _painter_contour_children(
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    qname,
) -> tuple[list[ET.Element], dict[str, int]] | None:
    """Seviye başına tek grouped-evenodd `<path>` (hücre-kenarı union) maske çocuğu.

    Painter'ın polygon kodlamasıyla AYNI hücre-kenarı geometrisini (``trace_cell_
    contours``) kullanır → dolgu alfa-tam; ama döngü başına ayrı ``<polygon>``
    yerine SEVİYE başına tek even-odd ``<path>`` (kompakt relatif h/v). İç kenar
    yoktur (rect'in aksine) → journal değerlendirme çözünürlüğüne ölçeklenince AA
    dikişi üretmez. `<path>` maske geometrisi maske alt-ağacındadır
    (ROLE_MASK_DEFINITION → sanat parmak izinden hariç); toplam path/node/byte
    değişmemiş journal sınırlarınca bağlanır.
    """
    children: list[ET.Element] = []
    total_loops = 0
    total_nodes = 0
    for level in sorted(int(value) for value in opacity_by_level):
        if level <= 0:
            continue
        gray = int(round(float(opacity_by_level[level]) * 255))
        if gray <= 0:
            continue
        loops = trace_cell_contours(quantized == level)
        if not loops:
            continue
        path_data, nodes = _rectilinear_subpaths(loops)
        if not path_data:
            continue
        children.append(
            ET.Element(
                qname("path"),
                {
                    "fill": f"rgb({gray},{gray},{gray})",
                    "fill-rule": "evenodd",
                    "d": path_data,
                },
            )
        )
        total_loops += len(loops)
        total_nodes += nodes
    if not children:
        return None
    return children, {
        "contour_path_count": int(len(children)),
        "contour_command_count": int(total_nodes),
        "contour_loop_count": int(total_loops),
    }


def _requantize_alpha(
    alpha: np.ndarray, max_levels: int
) -> tuple[np.ndarray, dict[int, float]]:
    """Kaynak alfayı ≤ ``max_levels`` düzeye deterministik yeniden nicele.

    Şeffaf (alfa=0) her zaman düzey 0'dır. Pozitif alfalar [1,255] üzerinde
    (max_levels-1) düzgün kovaya bölünür; her kovanın temsili opaklığı, o kovadaki
    alfaların ORTALAMASIdır (round-trip gri = round(mean_alpha)). Daha kaba
    nicemleme daha küçük kodlama verir; kabul YALNIZ değişmemiş alfa IoU/MAE
    kapıları geçerse — kalite düşürerek testi geçme yoktur.
    """
    alpha = np.asarray(alpha, dtype=np.uint8)
    quantized = np.zeros(alpha.shape, dtype=np.int32)
    opacity_by_level: dict[int, float] = {0: 0.0}
    positive = alpha > 0
    if not bool(positive.any()):
        return quantized, opacity_by_level
    bucket_count = max(1, int(max_levels) - 1)
    buckets = np.clip(
        ((alpha.astype(np.int32) - 1) * bucket_count) // 255 + 1, 1, bucket_count
    )
    buckets[~positive] = 0
    for bucket in range(1, bucket_count + 1):
        selected = buckets == bucket
        if not bool(selected.any()):
            continue
        opacity_by_level[bucket] = float(alpha[selected].mean()) / 255.0
        quantized[selected] = bucket
    return quantized, opacity_by_level


def _serialized_children_size(children: list[ET.Element]) -> int:
    return sum(len(ET.tostring(child)) for child in children)


def build_painter_reconstruction_tree(
    original_root: ET.Element,
    canvas_element: ET.Element | None,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    stroke_width: float,
    mask_encoding: str = "polygon",
    transaction_id: str = "",
) -> tuple[ET.Element, dict[str, int]]:
    """Mask candidate paint once, retaining a proven canvas only as underpaint.

    ``canvas_element`` is a border-connected comparison background proven by
    :func:`app.alpha_candidate_background.classify_comparison_background`. It is
    moved into the same source-alpha-masked paint group and tagged transform-owned.
    This preserves full source-positive alpha support without leaving an unmasked
    opaque canvas. When it is ``None`` the candidate paint is preserved as-is.
    Colour is never inspected here.
    """
    from app.alpha_candidate_knockout import _local_name, _unique_id, _viewbox  # noqa: PLC0415
    from app.alpha_candidate_support import (  # noqa: PLC0415
        _PROTECTED_ROOT_TAGS,
        _expand_candidate_paint,
        _strip_content_alpha,
    )

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    root = copy.deepcopy(original_root)
    target_canvas = None
    if canvas_element is not None:
        canvas_index = list(original_root).index(canvas_element)
        target_canvas = list(root)[canvas_index]

    defs = next(
        (
            child
            for child in list(root)
            if _local_name(str(child.tag)).lower() == "defs"
        ),
        None,
    )
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)

    paint_id = _unique_id(root, "vektoryum-alpha-candidate-paint")
    paint = ET.SubElement(
        defs,
        qname("g"),
        {"id": paint_id, "data-vektoryum-alpha-candidate-paint": "preserved-v1"},
    )
    tag_transform_node(paint, ROLE_ARTWORK_CONTAINER, transaction_id)
    movable = [
        child
        for child in list(root)
        if child is not defs
        and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
    ]
    if not movable:
        raise RuntimeError("source_alpha_candidate_painter_no_paint")
    for child in movable:
        root.remove(child)
        if child is target_canvas:
            tag_transform_node(child, ROLE_CANVAS_UNDERPAINT, transaction_id)
            child.set(
                "data-vektoryum-candidate-geometry-underpaint",
                "comparison-canvas-v1",
            )
        _strip_content_alpha(child)
        paint.append(child)
    expanded_count = _expand_candidate_paint(paint, stroke_width)
    if expanded_count <= 0:
        raise RuntimeError("source_alpha_candidate_painter_no_fill_geometry")

    loops = _painter_loops(quantized, opacity_by_level)
    if not loops:
        raise RuntimeError("source_alpha_candidate_painter_empty_source")

    raster_height, raster_width = quantized.shape
    view_x, view_y, view_width, view_height = _viewbox(root)
    mask_id = _unique_id(root, "vektoryum-alpha-painter")
    mask = ET.SubElement(
        defs,
        qname("mask"),
        {
            "id": mask_id,
            "maskUnits": "userSpaceOnUse",
            "x": f"{view_x:g}",
            "y": f"{view_y:g}",
            "width": f"{view_width:g}",
            "height": f"{view_height:g}",
            "data-vektoryum-source-alpha-reconstruction": "painter-luminance-v1",
        },
    )
    tag_transform_node(mask, ROLE_MASK_DEFINITION, transaction_id)
    content = ET.SubElement(mask, qname("g"))
    content.set(
        "transform",
        f"translate({view_x:.12g} {view_y:.12g}) "
        f"scale({view_width / float(raster_width):.12g} "
        f"{view_height / float(raster_height):.12g})",
    )
    # Opak siyah taban: maske kanvasının alfası her yerde 1 kalır; sınır
    # pikselleri komşu gri değerlerinin lineer kapsama karışımına düşer.
    # Griler standart ``rgb(v,v,v)`` gösterimiyle yazılır: bunlar sanat
    # paletinin renkleri değil maske-içi luminance kodlamasıdır ve palet
    # sayımı sanat eserinin hex fill'lerini ölçmeye devam eder.
    ET.SubElement(
        content,
        qname("rect"),
        {
            "x": "0",
            "y": "0",
            "width": str(raster_width),
            "height": str(raster_height),
            "fill": "rgb(0,0,0)",
        },
    )
    # Luminance mask iki path_count-güvenli (aday kimliği korunur), aynı
    # native-grid alfa düzlemini üreten kodlamadan biriyle yazılır:
    #   * "polygon" — döngü başına union-kontur <polygon>. İç kenarı yoktur, bu
    #     yüzden mask ölçeklendiğinde (journal'ın değerlendirme çözünürlüğü)
    #     dikiş üretmez; VARSAYILAN ve render-güvenli kodlamadır.
    #   * "rect" — seviye başına gruplanmış run-length <rect>. Çok daha kompakt
    #     ama tuğlalar arasında çok sayıda İÇ kenar taşır; ölçeklenince bu
    #     kenarlarda AA dikişleri oluşup seam/ssim/edge kapılarını düşürebilir.
    #     Bu yüzden yalnızca polygon byte bütçesine SIĞMADIĞINDA fallback olarak
    #     çağrılır (çağıran katman kontrol eder), geçen vakaları bozmaz.
    chosen_encoding = "polygon"
    rect_count = 0
    contour_stats: dict[str, int] = {}
    if mask_encoding == "contour":
        # Grouped-evenodd: seviye başına tek even-odd <path> (bridge-walk union).
        # İç kenar yok → ölçekli dikiş yok (polygon gibi), ama karmaşık maskede
        # çok daha kompakt. FAZ 3A kimlik modeli maske alt-ağacını sanat parmak
        # izinden dışladığı için bu <path>'ler artık kabul edilebilir.
        contour_result = _painter_contour_children(
            quantized, opacity_by_level, qname
        )
        if contour_result is not None:
            mask_children, contour_stats = contour_result
            chosen_encoding = "contour"
        else:  # kontur üretilemezse asla boş maske bırakma; polygon'a düş
            mask_children = _painter_polygon_children(loops, qname)
    elif mask_encoding == "rect":
        mask_children, rect_count = _painter_rect_children(
            quantized, opacity_by_level, qname
        )
        if mask_children:
            chosen_encoding = "rect"
        else:  # rect üretilemezse asla boş maske bırakma; polygon'a düş
            mask_children = _painter_polygon_children(loops, qname)
    else:
        mask_children = _painter_polygon_children(loops, qname)
    for child in mask_children:
        content.append(child)

    layer = ET.SubElement(
        root,
        qname("g"),
        {"data-vektoryum-source-alpha-reconstruction": "painter-luminance-v1"},
    )
    tag_transform_node(layer, ROLE_MASK_APPLICATION, transaction_id)
    ET.SubElement(
        layer,
        qname("use"),
        {"href": f"#{paint_id}", "mask": f"url(#{mask_id})"},
    )
    return root, {
        "reconstruction_loop_count": int(len(loops)),
        "reconstruction_mask_encoding": chosen_encoding,
        "reconstruction_rect_count": int(rect_count),
        "candidate_support_expanded_geometry_count": int(expanded_count),
        "comparison_canvas_knocked_out": False,
        "comparison_canvas_retained_under_mask": bool(target_canvas is not None),
        **contour_stats,
    }


_PAINTER_ASSESS_STATUSES = (
    "byte_rejected",
    "render_rejected",
    "native_alpha_rejected",
    "bounded_alpha_rejected",
    "evaluator_rejected",
    "structure_rejected",
    "identity_rejected",
    "geometry_rejected",
    "accepted",
)


def _assess_painter_candidate(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    grid_alpha: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
    transaction_id: str = "",
    parent_artwork_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Fail-closed dual-scale alfa değerlendirmesi — YAPILANDIRILMIŞ sonuç döner.

    Kapı hatasında exception ATMAZ; her aşamanın gerçek sayısal metriğini ve
    kesin red aşamasını/kodunu kaydeder ki FAZ 3B.1 encoding attempt ledger'ı
    gerçek red nedenini taşısın (son denemenin ötekini ezmesi engellenir). Sıra
    ve EŞİKLER değişmemiştir: native alfa → bounded alfa → evaluator alfa →
    structure → artwork identity. Görünüm/topoloji/seam TransformJournal'a aittir.
    """
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS  # noqa: PLC0415
    from app.alpha_candidate_knockout import _source_rgb_on_white  # noqa: PLC0415
    from app.final_artifact_evaluator import (  # noqa: PLC0415
        _structure_check,
        _thresholds,
        evaluate_final_svg,
    )

    image_class = _MODE_IMAGE_CLASS.get(mode, "clean_logo")
    thresholds = _thresholds(image_class, None)
    iou_min = float(thresholds["alpha_iou_min"])
    mae_max = float(thresholds["alpha_mae_max"])

    result: dict[str, Any] = {
        "status": None,
        "validation_stage": None,
        "exact_error_code": "",
        "native_alpha_iou": None,
        "native_alpha_mae": None,
        "bounded_alpha_iou": None,
        "bounded_alpha_mae": None,
        "evaluator_alpha_iou": None,
        "evaluator_alpha_mae": None,
        "artwork_fingerprint_match": None,
        "actual_path_count": None,
        "actual_node_count": None,
        "report": None,
    }

    grid_height, grid_width = grid_alpha.shape
    native = render_svg_to_rgba(candidate_path, grid_width, grid_height)
    if native is None:
        result["status"] = "render_rejected"
        result["validation_stage"] = "render"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_render_unmeasured"
        )
        return result
    if native.shape[:2] != (grid_height, grid_width):
        native = resize_rgba(native, grid_width, grid_height)
    native_metrics = alpha_plane_metrics(grid_alpha, native[:, :, 3])
    result["native_alpha_iou"] = float(native_metrics["alpha_iou"])
    result["native_alpha_mae"] = float(native_metrics["alpha_mae"])
    if float(native_metrics["alpha_iou"]) < iou_min:
        result["status"] = "native_alpha_rejected"
        result["validation_stage"] = "native_alpha"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_native_iou_gate_failed:"
            f"{native_metrics['alpha_iou']:.6f}<{iou_min}"
        )
        return result
    if float(native_metrics["alpha_mae"]) > mae_max:
        result["status"] = "native_alpha_rejected"
        result["validation_stage"] = "native_alpha"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_native_mae_gate_failed:"
            f"{native_metrics['alpha_mae']:.6f}>{mae_max}"
        )
        return result

    source_height, source_width = source_rgba_full.shape[:2]
    eval_scale = min(1.0, _PAINTER_EVAL_SIDE / float(max(source_width, source_height)))
    eval_width = max(1, int(round(source_width * eval_scale)))
    eval_height = max(1, int(round(source_height * eval_scale)))
    source_eval = resize_rgba(source_rgba_full, eval_width, eval_height)
    rendered_eval_alpha = cv2.resize(
        native[:, :, 3], (eval_width, eval_height), interpolation=cv2.INTER_AREA
    )
    direct_metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered_eval_alpha)
    result["bounded_alpha_iou"] = float(direct_metrics["alpha_iou"])
    result["bounded_alpha_mae"] = float(direct_metrics["alpha_mae"])
    if float(direct_metrics["alpha_iou"]) < iou_min:
        result["status"] = "bounded_alpha_rejected"
        result["validation_stage"] = "bounded_alpha"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_iou_gate_failed:"
            f"{direct_metrics['alpha_iou']:.6f}<{iou_min}"
        )
        return result
    if float(direct_metrics["alpha_mae"]) > mae_max:
        result["status"] = "bounded_alpha_rejected"
        result["validation_stage"] = "bounded_alpha"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_mae_gate_failed:"
            f"{direct_metrics['alpha_mae']:.6f}>{mae_max}"
        )
        return result

    report = evaluate_final_svg(
        candidate_path,
        _source_rgb_on_white(source_rgba_full),
        source_alpha=source_rgba_full[:, :, 3],
        image_class=image_class,
        required_metrics={"alpha_fidelity"},
    )
    alpha_group = report.metrics.get("G_gradient_alpha") or {}
    evaluator_alpha_iou = alpha_group.get("alpha_iou")
    evaluator_alpha_mae = alpha_group.get("alpha_mae")
    if evaluator_alpha_iou is not None:
        result["evaluator_alpha_iou"] = float(evaluator_alpha_iou)
    if evaluator_alpha_mae is not None:
        result["evaluator_alpha_mae"] = float(evaluator_alpha_mae)
    if evaluator_alpha_iou is None or evaluator_alpha_mae is None:
        result["status"] = "evaluator_rejected"
        result["validation_stage"] = "evaluator"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_evaluator_rejected:alpha_plane_unmeasured"
        )
        return result
    plane_failure_codes = [
        code for code in report.hard_fail_codes if code in _ALPHA_PLANE_FAILURE_CODES
    ]
    if float(evaluator_alpha_iou) < iou_min and "alpha_iou_below_min" not in plane_failure_codes:
        plane_failure_codes.append("alpha_iou_below_min")
    if float(evaluator_alpha_mae) > mae_max and "alpha_mae_above_max" not in plane_failure_codes:
        plane_failure_codes.append("alpha_mae_above_max")
    if plane_failure_codes:
        result["status"] = "evaluator_rejected"
        result["validation_stage"] = "evaluator"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_evaluator_rejected:"
            + ",".join(plane_failure_codes)
        )
        return result

    structure, _messages, structure_codes, root = _structure_check(
        Path(candidate_path).read_bytes()
    )
    if structure_codes or root is None:
        result["status"] = "structure_rejected"
        result["validation_stage"] = "structure"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_structure_failed:"
            + ",".join(structure_codes or ["parse_failed"])
        )
        return result
    after_counts = (
        int(structure.get("path_count") or 0),
        int(structure.get("node_count") or 0),
    )
    result["actual_path_count"] = int(after_counts[0])
    result["actual_node_count"] = int(after_counts[1])
    # FAZ 3A — kör toplam-sayım eşitliği yerine provenance-farkında sanat kimliği.
    if parent_artwork_fingerprint is not None:
        candidate_fingerprint = artwork_fingerprint(root, transaction_id)
        result["artwork_fingerprint_match"] = bool(
            candidate_fingerprint == parent_artwork_fingerprint
        )
        if not result["artwork_fingerprint_match"]:
            result["status"] = "identity_rejected"
            result["validation_stage"] = "identity"
            result["exact_error_code"] = (
                "source_alpha_candidate_painter_artwork_identity_changed:"
                f"{parent_artwork_fingerprint[:12]}!={candidate_fingerprint[:12]}"
            )
            return result
    elif after_counts != parent_counts:
        result["artwork_fingerprint_match"] = False
        result["status"] = "identity_rejected"
        result["validation_stage"] = "identity"
        result["exact_error_code"] = (
            "source_alpha_candidate_painter_candidate_geometry_changed:"
            f"{parent_counts[0]}/{parent_counts[1]}->"
            f"{after_counts[0]}/{after_counts[1]}"
        )
        return result
    else:
        result["artwork_fingerprint_match"] = True

    result["status"] = "accepted"
    result["validation_stage"] = "accepted"
    result["report"] = {
        "painter_native_alpha_iou": float(native_metrics["alpha_iou"]),
        "painter_native_alpha_mae": float(native_metrics["alpha_mae"]),
        "source_truth_alpha_iou": float(direct_metrics["alpha_iou"]),
        "source_truth_alpha_mae": float(direct_metrics["alpha_mae"]),
        "source_truth_source_coverage": float(direct_metrics["source_coverage"]),
        "source_truth_render_coverage": float(direct_metrics["render_coverage"]),
        "source_truth_alignment": "native_render_inter_area_downscale",
        "final_evaluator_verdict": report.verdict,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": float(evaluator_alpha_iou),
        "final_evaluator_alpha_mae": float(evaluator_alpha_mae),
        "final_evaluator_alpha_source_resolution": f"{source_width}x{source_height}",
        "appearance_regression_authority": "transform_journal_parent_delta",
        "non_alpha_regression_authority": "transform_journal_parent_delta",
        "preserved_path_count": int(after_counts[0]),
        "preserved_node_count": int(after_counts[1]),
        "parent_path_count": int(parent_counts[0]),
        "parent_node_count": int(parent_counts[1]),
        "artwork_identity_preserved": True,
        "artwork_identity_authority": (
            "provenance_fingerprint"
            if parent_artwork_fingerprint is not None
            else "total_count_equality"
        ),
    }
    return result


def validate_painter_reconstruction(
    candidate_path: Path,
    source_rgba_full: np.ndarray,
    grid_alpha: np.ndarray,
    mode: str,
    parent_counts: tuple[int, int],
    transaction_id: str = "",
    parent_artwork_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Fail-closed dual-scale alpha validation with unchanged thresholds.

    Thin sarmalayıcı: yapılandırılmış değerlendirmeyi çağırır ve kabul edilmeyen
    adayı aynı kesin hata koduyla reddeder (dış sözleşme korunur).
    """
    assessment = _assess_painter_candidate(
        candidate_path,
        source_rgba_full,
        grid_alpha,
        mode,
        parent_counts,
        transaction_id,
        parent_artwork_fingerprint,
    )
    if assessment["status"] != "accepted":
        raise RuntimeError(assessment["exact_error_code"])
    return dict(assessment["report"] or {})


_PAINTER_ERROR_PREFIX = "source_alpha_candidate_painter_"


def _run_painter_geometry_journal(
    parent_path: Path,
    candidate_path: Path,
    journal_source_rgb: np.ndarray,
    image_class: str,
    transform_report: Any,
) -> tuple[bool, list[str]]:
    """Adayı, aşağı-akış ile AYNI değişmemiş TransformJournal geometri kapılarından
    (SSIM/edge/seam/topology/node/byte) geçir. Kabul edilirse (True, []); aksi halde
    (False, reason_codes). Böylece turnuva 'en küçük byte' seçimini YALNIZ bütün
    kapıları geçen adaylar arasında yapar; contour gibi byte-küçük ama seam/node
    reddi alan aday seçilmez (class_reklam regresyonunu önler). Journal DEĞİŞTİRİLMEZ,
    yalnızca kullanılır."""
    from app.transform_journal import TransformJournal  # noqa: PLC0415

    journal = TransformJournal(
        Path(parent_path),
        journal_source_rgb,
        image_class=image_class,
        required_metrics=set(),
    )
    accepted_path, stage = journal.consider_candidate(
        "source_alpha_painter_candidate",
        Path(parent_path),
        Path(candidate_path),
        transform_report=transform_report,
    )
    if Path(accepted_path) == Path(candidate_path):
        return True, []
    return False, [str(code) for code in (stage.get("reason_codes") or ["unknown"])]


def _short_error_code(error_code: str) -> str:
    if error_code.startswith(_PAINTER_ERROR_PREFIX):
        return error_code[len(_PAINTER_ERROR_PREFIX):]
    return error_code


def _emit_painter_attempts(attempts: list[dict[str, Any]]) -> str:
    """Güvenli sayısal telemetriyi deterministik JSON satırı olarak stderr'e yaz.

    Ham SVG / path ``d`` / kaynak byte'ı İÇERMEZ; yalnız sayısal metrik, enum ve
    hata kodu. Deterministik: sort_keys, sabit liste sırası, locale-bağımsız
    sayı formatı, rastgele ID yok → aynı girdi aynı SHA.
    """
    payload = json.dumps(attempts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(
        f"source_alpha_candidate_painter_attempts={payload}",
        file=sys.stderr,
        flush=True,
    )
    return digest


def _painter_primary_error(
    attempts: list[dict[str, Any]], attempts_sha: str
) -> str:
    """Öncelikli ana hata — son deneme (rect byte) daha anlamlıyı EZMEZ.

    Öncelik: (1) bütçeye girip validation başlatan İLK exact aday hatası,
    (2) İLK quantized validation hatası, (3) en küçük byte'lı byte-rejected exact,
    (4) en küçük byte'lı byte-rejected quantized, (5) no_admissible_reconstruction.
    """
    def _validated(family: str) -> dict[str, Any] | None:
        for entry in attempts:
            if (
                entry.get("exact_or_quantized") == family
                and entry.get("validation_started")
                and entry.get("status") != "accepted"
            ):
                return entry
        return None

    def _smallest_byte_rejected(family: str) -> dict[str, Any] | None:
        candidates = [
            entry
            for entry in attempts
            if entry.get("exact_or_quantized") == family
            and entry.get("status") == "byte_rejected"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: int(e["actual_serialized_bytes"]))

    primary = (
        _validated("exact")
        or _validated("quantized")
        or _smallest_byte_rejected("exact")
        or _smallest_byte_rejected("quantized")
    )
    if primary is not None:
        label = primary["encoding_label"]
        code = _short_error_code(str(primary["exact_error_code"]))
        return (
            "source_alpha_candidate_painter_no_admissible_reconstruction:"
            f"primary={label}:{code};attempts_sha256={attempts_sha}"
        )
    return (
        "source_alpha_candidate_painter_no_admissible_reconstruction:"
        f"primary=none:no_candidate;attempts_sha256={attempts_sha}"
    )


def apply_candidate_painter_reconstruction(
    svg_path: Path,
    source_path: Path,
    mode: str,
) -> dict[str, Any]:
    """Build, validate and atomically publish the painter reconstruction."""
    from app.alpha_candidate_background import (  # noqa: PLC0415
        classify_comparison_background,
    )
    from app.alpha_candidate_knockout import (  # noqa: PLC0415
        _local_name,
        _path_node_counts,
        _render_root,
        _viewbox,
        _write_tree_to_temp,
    )
    from app.alpha_mask_budget import _journal_limits  # noqa: PLC0415
    from app.alpha_svg_mask import (  # noqa: PLC0415
    _painter_retry_eligible,
    _quantize_alpha,
)

    target = Path(svg_path)
    source = Path(source_path)
    before_bytes = target.read_bytes()
    before_size = len(before_bytes)
    ET.register_namespace("", _SVG_NS)
    original_root = ET.fromstring(before_bytes)
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_candidate_painter_root_not_svg")
    parent_counts = _path_node_counts(original_root)
    _view_x, _view_y, view_width, view_height = _viewbox(original_root)

    grid_scale = min(
        1.0, _PAINTER_GRID_MAX_SIDE / float(max(view_width, view_height))
    )
    grid_width = max(1, int(round(view_width * grid_scale)))
    grid_height = max(1, int(round(view_height * grid_scale)))
    grid_rgba = _rgba_from_source_at_size(source, (grid_width, grid_height))
    grid_alpha = np.asarray(grid_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(grid_alpha == 255)):
        raise RuntimeError("source_alpha_candidate_painter_opaque_source")
    if not np.any(grid_alpha > 0):
        raise RuntimeError("source_alpha_candidate_painter_empty_source")
    quantized, opacity_by_level = _quantize_alpha(grid_alpha)

    collapsed = _render_root(original_root, grid_width, grid_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_painter_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_painter_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    # Renk-agnostik sınıflandırma: A (kanıtlı border-connected background → yalnız
    # onu knockout), B (background yok → canvas_not_proven yerine knockout'suz
    # reconstruction), C (belirsiz → tahmin etme, çağıran orijinali birebir geri
    # yükler). "canvas_not_proven" crash'i artık üretilmez.
    background_status, canvas = classify_comparison_background(
        original_root, grid_rgba, grid_width, grid_height
    )
    if background_status == "ambiguous":
        raise RuntimeError("source_alpha_candidate_painter_background_ambiguous")
    limits = _journal_limits(original_root, before_size)

    # FAZ 3A — deterministik işlem kimliği + parent sanat parmak izi. Kimlik,
    # parent SVG SHA + kaynak alfa SHA + mod + kodlamadan türetilir (rastgele
    # UUID YOK → artifact SHA deterministik). Parent parmak izi, knockout
    # edilecek kanıtlı karşılaştırma-tuvalini dışlar (meşru arka-plan çıkarımı
    # kimlik ihlali sayılmaz); aday tarafta o tuval arşive taşınıp maske olarak
    # işaretlidir.
    parent_sha256 = hashlib.sha256(before_bytes).hexdigest()
    source_alpha_sha256 = hashlib.sha256(
        np.ascontiguousarray(grid_alpha).tobytes()
    ).hexdigest()
    excluded_from_parent = (canvas,) if canvas is not None else ()

    with Image.open(source) as source_image:
        source_size = source_image.size
    source_rgba_full = _rgba_from_source_at_size(source, source_size)

    # Aşağı-akış TransformJournal ile AYNI beyaz-kompozit kaynak RGB'si + aynı
    # image_class: turnuva her alfa-geçen adayı, seçmeden ÖNCE değişmemiş geometri
    # kapılarından (SSIM/edge/seam/topology/node/byte) geçirir. Parent bytes ayrı
    # bir geçici dosyaya yazılır (target kazanan ile ezilecek).
    from app.alpha_svg_mask import _MODE_IMAGE_CLASS as _JOURNAL_MODE_CLASS  # noqa: PLC0415

    _src_alpha = source_rgba_full[:, :, 3:4].astype(np.float32) / 255.0
    journal_source_rgb = np.clip(
        source_rgba_full[:, :, :3].astype(np.float32) * _src_alpha
        + 255.0 * (1.0 - _src_alpha),
        0,
        255,
    ).astype(np.uint8)
    journal_image_class = _JOURNAL_MODE_CLASS.get(mode, "clean_logo")
    parent_journal_path = target.parent / f".{target.name}.painter-parent.svg"
    parent_journal_path.write_bytes(before_bytes)

    byte_limit = int(limits["byte_limit"])

    # FAZ 3B / 3B.1 — painter kompakt encoding turnuvası + attempt ledger.
    # ÜÇ KADEMELİ seçim (render-güvenli varsayılan politikası):
    #   Kademe 1 (sayı-koruyan, render-güvenli): polygon + rect. Maskeleri
    #     <polygon>/<rect> ile yazılır → path_count'a SAYILMAZ (aday kimliği ham
    #     sayıyla da korunur); iç-kenarsız polygon yukarı-ölçekte referans-sadıktır,
    #     rect ise journal seam kapısıyla bağlıdır. Bu kademede bütçeye SIĞAN ve TÜM
    #     değişmemiş kapıları geçen adaylardan kazanan: en küçük byte → en az path →
    #     en az node → sabit sıra.
    #   Kademe 2 (kompakt, sayı-şişiren): contour. Maskesi <path> ile yazılır →
    #     path/node SAYILIR ve kesirli-ölçekte polygon'dan az sadıktır; YALNIZ hiçbir
    #     Kademe-1 adayı bütçeye sığmadığında (ör. public-05: polygon+rect >bütçe)
    #     devreye girer. Journal geometri kapısı burada da nihai yetkidir.
    #   Kademe 3 (quantized): contour-q128/64/32 — yalnız hiçbir exact geçmezse.
    # Her (stroke, encoding) denemesi yapılandırılmış ledger'a yazılır; hiçbir aday
    # kabul edilmezse ana hata öncelikle seçilir (son rect byte hatası daha anlamlıyı
    # EZMEZ).
    source_level_count = len([lvl for lvl in opacity_by_level if int(lvl) > 0])
    count_preserving_specs: list[tuple[str, str, str, np.ndarray, dict[int, float]]] = [
        ("polygon", "polygon", "exact", quantized, opacity_by_level),
        ("rect", "rect", "exact", quantized, opacity_by_level),
    ]
    compact_specs: list[tuple[str, str, str, np.ndarray, dict[int, float]]] = [
        ("contour", "contour", "exact", quantized, opacity_by_level),
    ]
    quantized_specs: list[tuple[str, str, str, np.ndarray, dict[int, float]]] = []
    for target_levels in (128, 64, 32):
        requant, requant_opacity = _requantize_alpha(grid_alpha, target_levels)
        quantized_specs.append(
            (f"contour-q{target_levels}", "contour", "quantized", requant, requant_opacity)
        )

    attempts: list[dict[str, Any]] = []

    def _evaluate_phase(
        specs: list[tuple[str, str, str, np.ndarray, dict[int, float]]],
    ) -> list[Any] | None:
        best: list[Any] | None = None
        for order, (label, mask_encoding, family, spec_quant, spec_opacity) in enumerate(specs):
            encoded_levels = len([lvl for lvl in spec_opacity if int(lvl) > 0])
            accepted: list[Any] | None = None
            for stroke_width in _PAINTER_STROKE_PIXELS:
                txn = alpha_transaction_id(
                    parent_sha256, source_alpha_sha256, mode, label
                )
                probe_root, probe_geometry = build_painter_reconstruction_tree(
                    original_root,
                    canvas,
                    spec_quant,
                    spec_opacity,
                    stroke_width,
                    mask_encoding=mask_encoding,
                    transaction_id=txn,
                )
                probe_temp = _write_tree_to_temp(probe_root, target)
                probe_size = int(probe_temp.stat().st_size)
                added_paths = int(probe_geometry.get("contour_path_count", 0))
                added_nodes = int(probe_geometry.get("contour_command_count", 0))
                entry: dict[str, Any] = {
                    "stroke_width": float(stroke_width),
                    "encoding_label": label,
                    "encoding_family": mask_encoding,
                    "exact_or_quantized": family,
                    "source_alpha_level_count": int(source_level_count),
                    "encoded_alpha_level_count": int(encoded_levels),
                    "actual_serialized_bytes": probe_size,
                    "byte_limit": int(byte_limit),
                    "projected_path_count": int(parent_counts[0] + added_paths),
                    "actual_path_count": None,
                    "path_limit": int(limits["path_limit"]),
                    "projected_node_count": (
                        int(parent_counts[1] + added_nodes) if added_nodes else None
                    ),
                    "actual_node_count": None,
                    "node_limit": int(limits["node_limit"]),
                    "preflight_status": None,
                    "validation_started": False,
                    "validation_stage": None,
                    "status": None,
                    "exact_error_code": "",
                    "native_alpha_iou": None,
                    "native_alpha_mae": None,
                    "bounded_alpha_iou": None,
                    "bounded_alpha_mae": None,
                    "evaluator_alpha_iou": None,
                    "evaluator_alpha_mae": None,
                    "artwork_fingerprint_match": None,
                    "journal_gate_started": False,
                    "journal_passed": None,
                    "journal_reason_codes": [],
                }
                if probe_size > byte_limit:
                    entry["preflight_status"] = "over_budget"
                    entry["status"] = "byte_rejected"
                    entry["exact_error_code"] = (
                        "source_alpha_candidate_painter_byte_budget_rejected:"
                        f"{label}:{probe_size}>{byte_limit}"
                    )
                    attempts.append(entry)
                    probe_temp.unlink(missing_ok=True)
                    continue
                entry["preflight_status"] = "within_budget"
                if (
                    canvas is not None
                    and float(stroke_width)
                    < _PAINTER_UNDERPAINT_MIN_STROKE_PIXELS
                ):
                    entry["validation_stage"] = "underpaint_support"
                    entry["status"] = "geometry_rejected"
                    entry["exact_error_code"] = (
                        "source_alpha_candidate_painter_underpaint_support_insufficient:"
                        f"{stroke_width:g}<"
                        f"{_PAINTER_UNDERPAINT_MIN_STROKE_PIXELS:g}"
                    )
                    attempts.append(entry)
                    probe_temp.unlink(missing_ok=True)
                    continue
                entry["validation_started"] = True
                parent_artwork_fp = artwork_fingerprint(
                    original_root, txn, excluded_from_parent
                )
                assessment = _assess_painter_candidate(
                    probe_temp,
                    source_rgba_full,
                    grid_alpha,
                    mode,
                    parent_counts,
                    transaction_id=txn,
                    parent_artwork_fingerprint=parent_artwork_fp,
                )
                for field in (
                    "validation_stage",
                    "status",
                    "exact_error_code",
                    "native_alpha_iou",
                    "native_alpha_mae",
                    "bounded_alpha_iou",
                    "bounded_alpha_mae",
                    "evaluator_alpha_iou",
                    "evaluator_alpha_mae",
                    "artwork_fingerprint_match",
                    "actual_path_count",
                    "actual_node_count",
                ):
                    entry[field] = assessment[field]
                if assessment["status"] == "accepted":
                    # Alfa geçti → seçmeden ÖNCE aşağı-akışla AYNI geometri
                    # kapılarından (seam/node/topology/SSIM/edge/byte) geçir.
                    entry["journal_gate_started"] = True
                    journal_passed, journal_codes = _run_painter_geometry_journal(
                        parent_journal_path,
                        probe_temp,
                        journal_source_rgb,
                        journal_image_class,
                        assessment["report"],
                    )
                    entry["journal_passed"] = bool(journal_passed)
                    entry["journal_reason_codes"] = list(journal_codes)
                    if journal_passed:
                        attempts.append(entry)
                        accepted = [
                            probe_size,
                            int(assessment["actual_path_count"] or 0),
                            int(assessment["actual_node_count"] or 0),
                            order,
                            probe_temp,
                            probe_geometry,
                            dict(assessment["report"] or {}),
                            label,
                            float(stroke_width),
                        ]
                        break  # bu encoding için en küçük geçen stroke yeterli
                    # Alfa geçti ama journal geometri kapısı reddetti (ör. contour'un
                    # seam/node patlaması) → seçilmez; en küçük byte politikası YALNIZ
                    # tüm kapıları geçen adaylar arasında çalışır (class_reklam korunur).
                    entry["status"] = "geometry_rejected"
                    entry["validation_stage"] = "journal_geometry"
                    entry["exact_error_code"] = (
                        "source_alpha_candidate_painter_journal_geometry_rejected:"
                        + ",".join(journal_codes)
                    )
                    attempts.append(entry)
                    probe_temp.unlink(missing_ok=True)
                    # FAZ 3C sözleşmesi: topology/seam/SSIM/edge-F1 reddi
                    # destek genişliğine ve ölçekli AA'ya bağlı olabilir. TÜM journal
                    # kodları retry-eligible ise aynı encoding'in bir sonraki mevcut
                    # stroke adayını dene. Node/path/byte/palet gibi kapsam dışı tek
                    # kod varsa erken kesme korunur; fail-open veya eşik değişikliği yok.
                    if _painter_retry_eligible(journal_codes):
                        continue
                    break
                attempts.append(entry)
                probe_temp.unlink(missing_ok=True)
            if accepted is not None:
                if best is None or tuple(accepted[:4]) < tuple(best[:4]):
                    if best is not None:
                        Path(best[4]).unlink(missing_ok=True)
                    best = accepted
                else:
                    Path(accepted[4]).unlink(missing_ok=True)
        return best

    try:
        # Kademe 1 → Kademe 2 → Kademe 3: render-güvenli sayı-koruyan kodlamalar
        # bütçeye sığdığında tercih; contour yalnız onlar sığmadığında; quantized
        # yalnız hiçbir exact geçmezse.
        winner = _evaluate_phase(count_preserving_specs)
        if winner is None:
            winner = _evaluate_phase(compact_specs)
        if winner is None:
            winner = _evaluate_phase(quantized_specs)
    finally:
        parent_journal_path.unlink(missing_ok=True)

    if winner is not None:
        (
            _w_bytes,
            _w_path,
            _w_node,
            _w_order,
            w_temp,
            w_geometry,
            w_report,
            w_label,
            w_stroke,
        ) = winner
        os.replace(w_temp, target)
        attempts_sha = _emit_painter_attempts(attempts)
        return {
            "status": "accepted",
            "applied": True,
            "schema": "rfv3d2-candidate-painter-reconstruction-v1",
            "before_byte_size": int(before_size),
            "after_byte_size": int(target.stat().st_size),
            "mask_encoding": "candidate_painter_luminance_mask",
            "painter_encoding_label": w_label,
            "painter_attempts_sha256": attempts_sha,
            "painter_attempt_count": int(len(attempts)),
            "comparison_background_status": background_status,
            "comparison_background_color_agnostic": True,
            "candidate_support_stroke_width_pixels": float(w_stroke),
            "candidate_geometry_preserved": True,
            "candidate_path_data_preserved": True,
            "trace_rgb_bytes_preserved": True,
            "candidate_identity_preserved": True,
            "painter_grid_width": int(grid_width),
            "painter_grid_height": int(grid_height),
            "preflight_parent_path_count": int(parent_counts[0]),
            "preflight_parent_node_count": int(parent_counts[1]),
            "preflight_path_limit": int(limits["path_limit"]),
            "preflight_node_limit": int(limits["node_limit"]),
            "preflight_byte_limit": int(limits["byte_limit"]),
            **w_geometry,
            **w_report,
        }

    attempts_sha = _emit_painter_attempts(attempts)
    raise RuntimeError(_painter_primary_error(attempts, attempts_sha))
