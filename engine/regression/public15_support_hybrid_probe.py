from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.alpha_candidate_painter import _rectilinear_subpaths
from app.alpha_mask_contour import trace_cell_contours
from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_support_topology_probe import _full_topology
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _qname(root: ET.Element, name: str) -> str:
    text = str(root.tag)
    if text.startswith("{") and "}" in text:
        return text.split("}", 1)[0] + "}" + name
    return name


def _grid_size_from_support(support: ET.Element) -> tuple[int, int]:
    max_x = 0
    max_y = 0
    for node in support.iter():
        if _local(node.tag) != "rect":
            continue
        x = int(float(node.attrib.get("x", "0")))
        y = int(float(node.attrib.get("y", "0")))
        w = int(float(node.attrib.get("width", "0")))
        h = int(float(node.attrib.get("height", "0")))
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    if max_x <= 0 or max_y <= 0:
        raise RuntimeError("public15 hybrid probe found no support rectangles")
    return max_x, max_y


def _replace_support(root: ET.Element, threshold: int) -> dict[str, int]:
    support = next(
        node for node in root.iter()
        if _local(node.tag) == "g"
        and node.attrib.get("data-vektoryum-paint-deficit") == "source-palette-v1"
    )
    width, height = _grid_size_from_support(support)
    qpath = _qname(root, "path")
    qgroup = _qname(root, "g")
    qrect = _qname(root, "rect")

    original_groups = list(support)
    for child in original_groups:
        support.remove(child)

    kept_rects = 0
    compacted_components = 0
    compacted_pixels = 0
    compacted_nodes = 0
    path_count = 0

    for group in original_groups:
        fill = group.attrib.get("fill", "rgb(0,0,0)")
        mask = np.zeros((height, width), dtype=np.uint8)
        rects: list[tuple[int, int, int, int]] = []
        for node in list(group):
            if _local(node.tag) != "rect":
                continue
            x = int(float(node.attrib.get("x", "0")))
            y = int(float(node.attrib.get("y", "0")))
            w = int(float(node.attrib.get("width", "0")))
            h = int(float(node.attrib.get("height", "0")))
            if w <= 0 or h <= 0:
                continue
            rects.append((x, y, w, h))
            mask[y:y+h, x:x+w] = 1

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        large_ids = {
            component_id
            for component_id in range(1, int(count))
            if int(stats[component_id, cv2.CC_STAT_AREA]) >= int(threshold)
        }

        small_group = ET.Element(qgroup, {"fill": fill})
        for x, y, w, h in rects:
            component_id = int(labels[y, x]) if y < height and x < width else 0
            if component_id in large_ids:
                continue
            ET.SubElement(
                small_group,
                qrect,
                {"x": str(x), "y": str(y), "width": str(w), "height": str(h)},
            )
            kept_rects += 1
        if len(small_group):
            support.append(small_group)

        loops: list[list[tuple[int, int]]] = []
        for component_id in sorted(large_ids):
            component = labels == component_id
            component_loops = trace_cell_contours(component)
            loops.extend(component_loops)
            compacted_components += 1
            compacted_pixels += int(stats[component_id, cv2.CC_STAT_AREA])
        if loops:
            path_data, nodes = _rectilinear_subpaths(loops)
            if path_data:
                ET.SubElement(
                    support,
                    qpath,
                    {"fill": fill, "fill-rule": "evenodd", "d": path_data},
                )
                path_count += 1
                compacted_nodes += int(nodes)

    return {
        "threshold_pixels": int(threshold),
        "kept_rects": int(kept_rects),
        "compacted_components": int(compacted_components),
        "compacted_pixels": int(compacted_pixels),
        "hybrid_path_count": int(path_count),
        "hybrid_path_nodes": int(compacted_nodes),
    }


def _native_support_signature(path: Path, source_size: tuple[int, int]) -> dict[str, int]:
    width, height = source_size
    rendered = render_svg_to_rgba(path, width, height)
    if rendered is None:
        return {"render_failed": 1}
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)
    binary = (np.asarray(rendered, dtype=np.uint8)[:, :, 3] > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    holes = 0
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        holes = sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
    return {
        "occupied_pixels": int(np.count_nonzero(binary)),
        "components": max(0, int(count) - 1),
        "holes": int(holes),
    }


def main() -> None:
    corpus = Path(os.environ["RFV_CORPUS"])
    out = Path(os.environ["PROOF_OUT"])
    engine_version = os.environ["ENGINE_VERSION"]
    out.mkdir(parents=True, exist_ok=True)

    v3 = run_probe(corpus, out / "v3", engine_version)
    selected = Path(str(v3["selected_candidate"]["path"]))
    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source = np.asarray(Image.open(corpus / case.source_path).convert("RGBA"), dtype=np.uint8)
    height, width = source.shape[:2]

    variants: list[dict[str, object]] = []
    for threshold in (4, 16, 64, 256, 1024):
        root = ET.parse(selected).getroot()
        stats = _replace_support(root, threshold)
        candidate = out / f"candidate-hybrid-{threshold}.svg"
        ET.ElementTree(root).write(candidate, encoding="utf-8", xml_declaration=True)
        variants.append({
            **stats,
            "bytes": int(candidate.stat().st_size),
            "support_native": _native_support_signature(candidate, (width, height)),
            "candidate_512": _full_topology(candidate, source, 512),
            "candidate_1024": _full_topology(candidate, source, 1024),
            "path": str(candidate),
        })

    evidence = {
        "schema": "vektoryum-public15-support-hybrid-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "baseline_bytes": int(selected.stat().st_size),
        "baseline_512": v3["complete_candidate_512"],
        "baseline_1024": v3["complete_candidate_1024"],
        "support_expected": v3["support_only"].get("expected"),
        "variants": variants,
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-support-hybrid-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_SUPPORT_HYBRID=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
