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
from app.source_truth import composite_rgba
from app.transform_journal import TransformJournal
from benchmark.pipeline_results import run_case
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(name: str) -> str:
    return str(name).rsplit("}", 1)[-1].lower()


def _seal_mask_paths(path: Path, width: float, out: Path) -> Path:
    tree = ET.parse(path)
    root = tree.getroot()
    touched = 0
    for element in root.iter():
        if _local(element.tag) != "path":
            continue
        parent_is_mask = False
        # ElementTree has no parent pointers; contour mask paths are the paths whose
        # fill is rgb(v,v,v) and whose fill-rule is evenodd. Artwork paths use the
        # preserved source styling and are intentionally untouched.
        fill = element.attrib.get("fill", "")
        if element.attrib.get("fill-rule") == "evenodd" and fill.startswith("rgb("):
            values = fill.removeprefix("rgb(").removesuffix(")").split(",")
            if len(values) == 3 and values[0] == values[1] == values[2]:
                parent_is_mask = True
        if not parent_is_mask:
            continue
        element.set("stroke", fill)
        element.set("stroke-width", f"{width:g}")
        element.set("stroke-linejoin", "miter")
        element.set("stroke-linecap", "butt")
        touched += 1
    if touched <= 0:
        raise RuntimeError("public15 contour seam probe found no mask contour paths")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    case = next(item for item in load_qualification_cases(corpus) if item.case_id == "qualification-public-15")
    source_path = corpus / case.source_path
    source_rgba = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
    source_rgb = composite_rgba(source_rgba, 255)

    candidates: list[dict[str, Any]] = []
    parent_copy: Path | None = None
    original_apply = painter.apply_candidate_painter_reconstruction
    original_assess = painter._assess_painter_candidate

    def wrapped_assess(candidate_path, *args, **kwargs):
        result = original_assess(candidate_path, *args, **kwargs)
        try:
            size = int(Path(candidate_path).stat().st_size)
            tree = ET.parse(candidate_path)
            root = tree.getroot()
            path_count, node_count = _path_node_counts(root)
            mask_contours = [
                element for element in root.iter()
                if _local(element.tag) == "path"
                and element.attrib.get("fill-rule") == "evenodd"
                and element.attrib.get("fill", "").startswith("rgb(")
            ]
            saved = out / f"captured-{len(candidates):02d}.svg"
            shutil.copy2(candidate_path, saved)
            candidates.append({
                "path": str(saved),
                "bytes": size,
                "path_count": int(path_count),
                "node_count": int(node_count),
                "mask_contour_count": int(len(mask_contours)),
                "assessment": {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None), list))},
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
        raise RuntimeError("public15 contour seam probe did not capture painter parent")

    # q32 is the smallest count-preserving contour candidate: one mask contour,
    # total path count 34 and size well below the live 335503-byte budget.
    eligible = [
        item for item in candidates
        if int(item["mask_contour_count"]) == 1
        and int(item["path_count"]) <= 40
        and int(item["bytes"]) < 200000
    ]
    if not eligible:
        raise RuntimeError("public15 contour seam probe did not capture q32 candidate")
    baseline = min(eligible, key=lambda item: int(item["bytes"]))
    baseline_path = Path(str(baseline["path"]))

    variants: list[dict[str, Any]] = []
    for width in (0.125, 0.25, 0.5, 0.75, 1.0):
        variant_path = out / f"contour-q32-mask-sealed-{width:g}.svg"
        _seal_mask_paths(baseline_path, width, variant_path)
        journal = TransformJournal(
            parent_copy,
            source_rgb,
            image_class="clean_logo",
            required_metrics=set(),
            budget_seconds=180.0,
            stage_timeout_seconds=600.0,
        )
        accepted_path, stage = journal.consider_candidate(
            f"public15_contour_q32_mask_seal_{width:g}",
            parent_copy,
            variant_path,
            transform_report={"diagnostic_only": True, "mask_seal_width": width},
        )
        evaluator = evaluate_final_svg(
            variant_path,
            source_rgb,
            source_alpha=source_rgba[:, :, 3],
            image_class="clean_logo",
            required_metrics={"alpha_fidelity"},
        ).to_dict()
        variants.append({
            "width": width,
            "bytes": int(variant_path.stat().st_size),
            "journal_accepted": Path(accepted_path) == variant_path,
            "journal_reason_codes": stage.get("reason_codes") or [],
            "journal_after": stage.get("after") or {},
            "evaluator_hard_fail_codes": evaluator.get("hard_fail_codes") or [],
            "evaluator_metrics": evaluator.get("metrics") or {},
        })

    evidence = {
        "schema": "vektoryum-public15-contour-seam-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "baseline": baseline,
        "variants": variants,
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "raster_embedding": False,
            "fixture_bypass": False,
        },
    }
    (out / "public15-contour-seam-proof.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    evidence = run_probe(Path(os.environ["RFV_CORPUS"]), Path(os.environ["PROOF_OUT"]), os.environ["ENGINE_VERSION"])
    print("PUBLIC15_CONTOUR_SEAM=" + json.dumps({
        "baseline_bytes": evidence["baseline"]["bytes"],
        "variants": [{
            "width": item["width"],
            "bytes": item["bytes"],
            "journal_accepted": item["journal_accepted"],
            "journal_reason_codes": item["journal_reason_codes"],
            "evaluator_hard_fail_codes": item["evaluator_hard_fail_codes"],
        } for item in evidence["variants"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
