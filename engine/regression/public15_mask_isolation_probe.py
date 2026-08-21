from __future__ import annotations

import copy
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if raw:
        values = [float(value) for value in re.split(r"[ ,]+", raw.strip()) if value]
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]
    width = float(str(root.attrib.get("width", "1")).replace("px", ""))
    height = float(str(root.attrib.get("height", "1")).replace("px", ""))
    return 0.0, 0.0, width, height


def _coverage(source_alpha: np.ndarray, rendered_alpha: np.ndarray) -> dict[str, int | float]:
    source_u8 = np.asarray(source_alpha, dtype=np.uint8)
    render_u8 = np.asarray(rendered_alpha, dtype=np.uint8)
    source_positive = source_u8 > 0
    render_positive = render_u8 > 0
    abs_error = np.abs(source_u8.astype(np.int16) - render_u8.astype(np.int16))
    intersection = int(np.count_nonzero(source_positive & render_positive))
    union = int(np.count_nonzero(source_positive | render_positive))
    return {
        "source_positive_pixels": int(np.count_nonzero(source_positive)),
        "render_positive_pixels": int(np.count_nonzero(render_positive)),
        "false_negative_pixels": int(np.count_nonzero(source_positive & ~render_positive)),
        "false_positive_pixels": int(np.count_nonzero(render_positive & ~source_positive)),
        "binary_iou": float(intersection / union) if union else 1.0,
        "alpha_mae_normalized": float(abs_error.mean() / 255.0),
        "alpha_abs_error_mean": float(abs_error.mean()),
        "alpha_abs_error_p95": float(np.percentile(abs_error, 95)),
        "alpha_abs_error_max": int(abs_error.max()),
    }


def _component_count(mask: np.ndarray, max_side: int) -> dict[str, int]:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    h0, w0 = binary.shape
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    width = max(1, int(round(w0 * scale)))
    height = max(1, int(round(h0 * scale)))
    if (width, height) != (w0, h0):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(1, round(0.00004 * width * height))
    kept = sum(1 for idx in range(1, int(count)) if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area)
    return {"width": width, "height": height, "components": int(kept), "min_area": int(min_area)}


def _mask_only_root(candidate: Path) -> ET.Element:
    original = ET.parse(candidate).getroot()
    layer = next(
        element
        for element in original.iter()
        if _local(element.tag) == "g"
        and element.attrib.get("data-vektoryum-source-alpha-reconstruction") == "paint-deficit-q24-v1"
        and "mask" in element.attrib
    )
    mask_ref = layer.attrib["mask"]
    defs = next(element for element in list(original) if _local(element.tag) == "defs")
    root = ET.Element(original.tag, dict(original.attrib))
    root.append(copy.deepcopy(defs))
    group = ET.SubElement(root, f"{{http://www.w3.org/2000/svg}}g", {"mask": mask_ref})
    x, y, width, height = _viewbox(original)
    ET.SubElement(
        group,
        f"{{http://www.w3.org/2000/svg}}rect",
        {
            "x": f"{x:g}",
            "y": f"{y:g}",
            "width": f"{width:g}",
            "height": f"{height:g}",
            "fill": "rgb(255,255,255)",
        },
    )
    return root


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

    root = _mask_only_root(selected)
    mask_path = out / "mask-only.svg"
    ET.ElementTree(root).write(mask_path, encoding="utf-8", xml_declaration=True)
    rendered = render_svg_to_rgba(mask_path, width, height)
    if rendered is None:
        raise RuntimeError("public15 mask isolation render failed")
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)
    rendered = np.asarray(rendered, dtype=np.uint8)

    source_alpha = source[:, :, 3]
    mask_alpha = rendered[:, :, 3]
    evidence = {
        "schema": "vektoryum-public15-mask-isolation-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "selected_candidate_bytes": int(selected.stat().st_size),
        "mask_only_bytes": int(mask_path.stat().st_size),
        "coverage": _coverage(source_alpha, mask_alpha),
        "components_512": {
            "source": _component_count(source_alpha, 512),
            "mask_render": _component_count(mask_alpha, 512),
        },
        "components_1024": {
            "source": _component_count(source_alpha, 1024),
            "mask_render": _component_count(mask_alpha, 1024),
        },
        "support_expected": v3["support_only"].get("expected"),
        "support_alpha_ge_1": v3["support_only"].get("render_alpha_ge_1"),
        "complete_candidate_512": v3["complete_candidate_512"],
        "complete_candidate_1024": v3["complete_candidate_1024"],
        "evaluator_hard_fail_codes": v3["final_artifact_evaluator"].get("hard_fail_codes"),
        "journal_reason_codes": v3["transform_journal"]["stage"].get("reason_codes"),
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-mask-isolation-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_MASK_ISOLATION=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
