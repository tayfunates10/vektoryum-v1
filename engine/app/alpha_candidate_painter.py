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
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.alpha_artwork_identity import (
    ROLE_ARTWORK_CONTAINER,
    ROLE_CANVAS_KNOCKOUT,
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
    """Mask the stroked paint once, knocking out a comparison canvas if given.

    ``canvas_element`` is a border-connected comparison background proven by
    :func:`app.alpha_candidate_background.classify_comparison_background`. When it
    is ``None`` (no background trace) the candidate paint is preserved as-is and
    source alpha is reconstructed over it without any knockout (canvas-independent
    path). Colour is never inspected here.
    """
    from app.alpha_candidate_knockout import _local_name, _unique_id, _viewbox  # noqa: PLC0415
    from app.alpha_candidate_support import (  # noqa: PLC0415
        _PROTECTED_ROOT_TAGS,
        _expand_candidate_paint,
        _strip_content_alpha,
    )

    qname = lambda name: f"{{{_SVG_NS}}}{name}"
    root = copy.deepcopy(original_root)
    knocked_out_canvas = False
    if canvas_element is not None:
        canvas_index = list(original_root).index(canvas_element)
        target_canvas = list(root)[canvas_index]
        root.remove(target_canvas)

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
    if canvas_element is not None:
        archive = ET.SubElement(
            defs,
            qname("g"),
            {
                "data-vektoryum-candidate-geometry-knockout": "comparison-canvas-v1",
                "display": "none",
            },
        )
        tag_transform_node(archive, ROLE_CANVAS_KNOCKOUT, transaction_id)
        archive.append(target_canvas)
        knocked_out_canvas = True

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
    if mask_encoding == "rect":
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
        "comparison_canvas_knocked_out": bool(knocked_out_canvas),
    }


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

    The direct gate renders once at the reconstruction's native grid — where
    the geometry is pixel-aligned and must reproduce the staged alpha plane —
    and compares the bounded evaluation view through the same INTER_AREA
    downscale the source reference uses, so both sides of the comparison pass
    through identical filters. Thresholds are the existing image-class hard
    gates; nothing is loosened. FinalArtifactEvaluator then independently
    confirms the alpha plane against the full-resolution source, exactly like
    the established dual contract, and appearance/topology/seam stay owned by
    the following real TransformJournal parent-delta stage.
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

    grid_height, grid_width = grid_alpha.shape
    native = render_svg_to_rgba(candidate_path, grid_width, grid_height)
    if native is None:
        raise RuntimeError("source_alpha_candidate_painter_render_unmeasured")
    if native.shape[:2] != (grid_height, grid_width):
        native = resize_rgba(native, grid_width, grid_height)
    native_metrics = alpha_plane_metrics(grid_alpha, native[:, :, 3])
    if float(native_metrics["alpha_iou"]) < iou_min:
        raise RuntimeError(
            "source_alpha_candidate_painter_native_iou_gate_failed:"
            f"{native_metrics['alpha_iou']:.6f}<{iou_min}"
        )
    if float(native_metrics["alpha_mae"]) > mae_max:
        raise RuntimeError(
            "source_alpha_candidate_painter_native_mae_gate_failed:"
            f"{native_metrics['alpha_mae']:.6f}>{mae_max}"
        )

    source_height, source_width = source_rgba_full.shape[:2]
    eval_scale = min(1.0, _PAINTER_EVAL_SIDE / float(max(source_width, source_height)))
    eval_width = max(1, int(round(source_width * eval_scale)))
    eval_height = max(1, int(round(source_height * eval_scale)))
    source_eval = resize_rgba(source_rgba_full, eval_width, eval_height)
    rendered_eval_alpha = cv2.resize(
        native[:, :, 3], (eval_width, eval_height), interpolation=cv2.INTER_AREA
    )
    direct_metrics = alpha_plane_metrics(source_eval[:, :, 3], rendered_eval_alpha)
    if float(direct_metrics["alpha_iou"]) < iou_min:
        raise RuntimeError(
            "source_alpha_candidate_painter_iou_gate_failed:"
            f"{direct_metrics['alpha_iou']:.6f}<{iou_min}"
        )
    if float(direct_metrics["alpha_mae"]) > mae_max:
        raise RuntimeError(
            "source_alpha_candidate_painter_mae_gate_failed:"
            f"{direct_metrics['alpha_mae']:.6f}>{mae_max}"
        )

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
    if evaluator_alpha_iou is None or evaluator_alpha_mae is None:
        raise RuntimeError(
            "source_alpha_candidate_painter_evaluator_rejected:alpha_plane_unmeasured"
        )
    plane_failure_codes = [
        code for code in report.hard_fail_codes if code in _ALPHA_PLANE_FAILURE_CODES
    ]
    if float(evaluator_alpha_iou) < iou_min and "alpha_iou_below_min" not in plane_failure_codes:
        plane_failure_codes.append("alpha_iou_below_min")
    if float(evaluator_alpha_mae) > mae_max and "alpha_mae_above_max" not in plane_failure_codes:
        plane_failure_codes.append("alpha_mae_above_max")
    if plane_failure_codes:
        raise RuntimeError(
            "source_alpha_candidate_painter_evaluator_rejected:"
            + ",".join(plane_failure_codes)
        )

    structure, _messages, structure_codes, root = _structure_check(
        Path(candidate_path).read_bytes()
    )
    if structure_codes or root is None:
        raise RuntimeError(
            "source_alpha_candidate_painter_structure_failed:"
            + ",".join(structure_codes or ["parse_failed"])
        )
    after_counts = (
        int(structure.get("path_count") or 0),
        int(structure.get("node_count") or 0),
    )
    # FAZ 3A — kör toplam-sayım eşitliği yerine provenance-farkında sanat kimliği.
    # Transform-owned maske/destek geometrisi (bu transaction'a etiketli) sanat
    # parmak izinden dışlanır; sanat eserinin geometri+renk kimliği parent ile
    # BİREBİR aynı olmalı. Toplam karmaşıklık (byte/path/node) ve görünüm
    # (SSIM/edge/seam/topology) değişmemiş journal kapılarınca bağlanmaya devam
    # eder — bu bir bypass değil, kimliğin iki ayrı sözleşmeye bölünmesidir.
    if parent_artwork_fingerprint is not None:
        candidate_fingerprint = artwork_fingerprint(root, transaction_id)
        if candidate_fingerprint != parent_artwork_fingerprint:
            raise RuntimeError(
                "source_alpha_candidate_painter_artwork_identity_changed:"
                f"{parent_artwork_fingerprint[:12]}!={candidate_fingerprint[:12]}"
            )
    elif after_counts != parent_counts:
        # Geriye dönük güvenli varsayılan: parmak izi verilmediyse eski sözleşme.
        raise RuntimeError(
            "source_alpha_candidate_painter_candidate_geometry_changed:"
            f"{parent_counts[0]}/{parent_counts[1]}->"
            f"{after_counts[0]}/{after_counts[1]}"
        )

    return {
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
    from app.alpha_svg_mask import _quantize_alpha  # noqa: PLC0415

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

    byte_limit = int(limits["byte_limit"])
    last_error: RuntimeError | None = None
    for stroke_width in _PAINTER_STROKE_PIXELS:
        # Render-güvenli polygon önce denenir; yalnız polygon byte bütçesini
        # aşarsa daha kompakt rect fallback'i denenir. Böylece polygon'un sığdığı
        # (ve geçen) vakalarda rect'in ölçekli dikişleri devreye girmez.
        candidate_root = None
        geometry = None
        temporary = None
        selected_txn = ""
        for encoding in ("polygon", "rect"):
            txn = alpha_transaction_id(
                parent_sha256, source_alpha_sha256, mode, encoding
            )
            probe_root, probe_geometry = build_painter_reconstruction_tree(
                original_root,
                canvas,
                quantized,
                opacity_by_level,
                stroke_width,
                mask_encoding=encoding,
                transaction_id=txn,
            )
            probe_temp = _write_tree_to_temp(probe_root, target)
            probe_size = probe_temp.stat().st_size
            if probe_size > byte_limit:
                last_error = RuntimeError(
                    "source_alpha_candidate_painter_byte_budget_rejected:"
                    f"{probe_size}>{byte_limit}"
                )
                probe_temp.unlink(missing_ok=True)
                # polygon sığmadıysa rect'i dene; rect kodlamasıysa bu stroke biter
                continue
            candidate_root, geometry, temporary, selected_txn = (
                probe_root,
                probe_geometry,
                probe_temp,
                txn,
            )
            break
        if temporary is None:
            continue  # her iki kodlama da bu stroke'ta bütçeyi aştı
        parent_artwork_fp = artwork_fingerprint(
            original_root, selected_txn, excluded_from_parent
        )
        try:
            try:
                validation = validate_painter_reconstruction(
                    temporary,
                    source_rgba_full,
                    grid_alpha,
                    mode,
                    parent_counts,
                    transaction_id=selected_txn,
                    parent_artwork_fingerprint=parent_artwork_fp,
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return {
            "status": "accepted",
            "applied": True,
            "schema": "rfv3d2-candidate-painter-reconstruction-v1",
            "before_byte_size": int(before_size),
            "after_byte_size": int(target.stat().st_size),
            "mask_encoding": "candidate_painter_luminance_mask",
            "comparison_background_status": background_status,
            "comparison_background_color_agnostic": True,
            "candidate_support_stroke_width_pixels": float(stroke_width),
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
            **geometry,
            **validation,
        }

    raise last_error or RuntimeError(
        "source_alpha_candidate_painter_no_admissible_reconstruction"
    )
