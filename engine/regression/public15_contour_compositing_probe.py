from __future__ import annotations

import copy
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app import alpha_candidate_painter as painter
from app.alpha_candidate_knockout import _path_node_counts
from app.final_artifact_evaluator import evaluate_final_svg
from app.pipeline_entry import run_pipeline
from app.source_truth import alpha_plane_metrics, composite_rgba, render_svg_to_rgba, resize_rgba
from app.transform_journal import TransformJournal
from benchmark.pipeline_results import run_case
from engine.regression.public15_support_topology_probe import _full_topology
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(name: str) -> str:
    return str(name).rsplit("}", 1)[-1].lower()


def _capture_q32(corpus: Path, out: Path, engine_version: str) -> tuple[Path, Path, np.ndarray]:
    case = next(item for item in load_qualification_cases(corpus) if item.case_id == "qualification-public-15")
    source_path = corpus / case.source_path
    source_rgba = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
    candidates: list[dict[str, Any]] = []
    parent_copy: Path | None = None
    original_apply = painter.apply_candidate_painter_reconstruction
    original_assess = painter._assess_painter_candidate

    def wrapped_assess(candidate_path, *args, **kwargs):
        result = original_assess(candidate_path, *args, **kwargs)
        try:
            path = Path(candidate_path)
            root = ET.parse(path).getroot()
            path_count, node_count = _path_node_counts(root)
            mask_contours = [
                element for element in root.iter()
                if _local(element.tag) == "path"
                and element.attrib.get("fill-rule") == "evenodd"
                and element.attrib.get("fill", "").startswith("rgb(")
            ]
            saved = out / f"captured-{len(candidates):02d}.svg"
            shutil.copy2(path, saved)
            candidates.append({
                "path": str(saved),
                "bytes": int(saved.stat().st_size),
                "path_count": int(path_count),
                "node_count": int(node_count),
                "mask_contour_count": int(len(mask_contours)),
            })
        except Exception:
            pass
        return result

    def wrapped_apply(svg_path, source_path_arg, mode):
        nonlocal parent_copy
        if parent_copy is None:
            parent_copy = out / "parent.svg"
            shutil.copy2(svg_path, parent_copy)
        return original_apply(svg_path, source_path_arg, mode)

    painter._assess_painter_candidate = wrapped_assess
    painter.apply_candidate_painter_reconstruction = wrapped_apply
    try:
        try:
            run_case(
                case,
                corpus_root=corpus,
                work_root=out / "pipeline-job",
                pipeline=run_pipeline,
                engine_version=engine_version,
                trace_mode="auto",
                peak_rss_mb=None,
            )
        except BaseException:
            pass
    finally:
        painter._assess_painter_candidate = original_assess
        painter.apply_candidate_painter_reconstruction = original_apply

    if parent_copy is None or not parent_copy.exists():
        raise RuntimeError("public15 compositing probe did not capture painter parent")
    eligible = [
        item for item in candidates
        if int(item["mask_contour_count"]) == 1
        and int(item["path_count"]) <= 40
        and int(item["bytes"]) < 200000
    ]
    if not eligible:
        raise RuntimeError("public15 compositing probe did not capture contour-q32")
    baseline = min(eligible, key=lambda item: int(item["bytes"]))
    return Path(str(baseline["path"])), parent_copy, source_rgba


def _find_mask_application(root: ET.Element) -> tuple[ET.Element, str]:
    for element in root.iter():
        value = element.attrib.get("mask", "")
        if value.startswith("url(#") and value.endswith(")"):
            return element, value[5:-1]
    raise RuntimeError("public15 compositing probe found no mask application")


def _svg_dims(root: ET.Element) -> tuple[str, str]:
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width and height:
        return width, height
    viewbox = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(viewbox) == 4:
        return viewbox[2], viewbox[3]
    return "1600", "1600"


def _write_variants(baseline: Path, out: Path) -> dict[str, Path]:
    tree = ET.parse(baseline)
    root = tree.getroot()
    application, mask_id = _find_mask_application(root)
    mask_attr = application.attrib["mask"]
    variants: dict[str, Path] = {"baseline": baseline}

    unmasked_root = copy.deepcopy(root)
    unmasked_app, _ = _find_mask_application(unmasked_root)
    unmasked_app.attrib.pop("mask", None)
    unmasked_path = out / "contour-q32-artwork-unmasked.svg"
    ET.ElementTree(unmasked_root).write(unmasked_path, encoding="utf-8", xml_declaration=True)
    variants["artwork_unmasked"] = unmasked_path

    mask_root = copy.deepcopy(root)
    width, height = _svg_dims(mask_root)
    defs = [child for child in list(mask_root) if _local(child.tag) == "defs"]
    for child in list(mask_root):
        mask_root.remove(child)
    for child in defs:
        mask_root.append(child)
    ns = mask_root.tag.split("}", 1)[0].removeprefix("{") if "}" in mask_root.tag else "http://www.w3.org/2000/svg"
    rect = ET.SubElement(mask_root, f"{{{ns}}}rect", {
        "x": "0", "y": "0", "width": width, "height": height,
        "fill": "white", "mask": mask_attr,
        "data-public15-diagnostic": "mask-on-solid-white",
    })
    _ = rect
    mask_path = out / "contour-q32-mask-on-solid.svg"
    ET.ElementTree(mask_root).write(mask_path, encoding="utf-8", xml_declaration=True)
    variants["mask_on_solid"] = mask_path

    no_defs_root = copy.deepcopy(root)
    app2, _ = _find_mask_application(no_defs_root)
    app2.attrib.pop("mask", None)
    for defs_node in [e for e in no_defs_root.iter() if _local(e.tag) == "defs"]:
        for child in list(defs_node):
            if child.attrib.get("id") == mask_id:
                defs_node.remove(child)
    no_defs_path = out / "contour-q32-artwork-only.svg"
    ET.ElementTree(no_defs_root).write(no_defs_path, encoding="utf-8", xml_declaration=True)
    variants["artwork_only"] = no_defs_path
    return variants


def _measure_variant(name: str, path: Path, parent: Path, source_rgba: np.ndarray) -> dict[str, Any]:
    source_rgb = composite_rgba(source_rgba, 255)
    rendered = render_svg_to_rgba(path, source_rgba.shape[1], source_rgba.shape[0])
    alpha_metrics: dict[str, float] | None = None
    if rendered is not None:
        if rendered.shape[:2] != source_rgba.shape[:2]:
            rendered = resize_rgba(rendered, source_rgba.shape[1], source_rgba.shape[0])
        alpha_metrics = alpha_plane_metrics(source_rgba[:, :, 3], rendered[:, :, 3])
    journal = TransformJournal(
        parent,
        source_rgb,
        image_class="clean_logo",
        required_metrics=set(),
        budget_seconds=180.0,
        stage_timeout_seconds=600.0,
    )
    accepted_path, stage = journal.consider_candidate(
        f"public15_contour_compositing_{name}",
        parent,
        path,
        transform_report={"diagnostic_only": True, "variant": name},
    )
    evaluator = evaluate_final_svg(
        path,
        source_rgb,
        source_alpha=source_rgba[:, :, 3],
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    ).to_dict()
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "native_alpha": alpha_metrics,
        "topology_512": _full_topology(path, source_rgba, 512),
        "topology_1024": _full_topology(path, source_rgba, 1024),
        "journal_accepted": Path(accepted_path) == path,
        "journal_reason_codes": stage.get("reason_codes") or [],
        "journal_after": stage.get("after") or {},
        "evaluator_hard_fail_codes": evaluator.get("hard_fail_codes") or [],
        "evaluator_metrics": evaluator.get("metrics") or {},
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, parent, source_rgba = _capture_q32(corpus, out, engine_version)
    variants = _write_variants(baseline, out)
    measured = {name: _measure_variant(name, path, parent, source_rgba) for name, path in variants.items()}
    evidence = {
        "schema": "vektoryum-public15-contour-compositing-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "variants": measured,
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed": False,
            "raster_embedding": False,
            "fixture_bypass": False,
        },
    }
    (out / "public15-contour-compositing-proof.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    evidence = run_probe(Path(os.environ["RFV_CORPUS"]), Path(os.environ["PROOF_OUT"]), os.environ["ENGINE_VERSION"])
    print("PUBLIC15_CONTOUR_COMPOSITING=" + json.dumps({
        name: {
            "bytes": item["bytes"],
            "native_alpha": item["native_alpha"],
            "topology_512": item["topology_512"],
            "topology_1024": item["topology_1024"],
            "journal_accepted": item["journal_accepted"],
            "journal_reason_codes": item["journal_reason_codes"],
            "evaluator_hard_fail_codes": item["evaluator_hard_fail_codes"],
        }
        for name, item in evidence["variants"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
