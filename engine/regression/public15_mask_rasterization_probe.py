from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_mask_isolation_probe import _coverage, _mask_only_root
from engine.regression.public15_support_topology_probe import _full_topology
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _crisp_mask(root: ET.Element) -> None:
    for mask in root.iter():
        if _local(mask.tag) != "mask":
            continue
        for child in list(mask):
            if _local(child.tag) == "g":
                child.set("shape-rendering", "crispEdges")


def _render_alpha(path: Path, width: int, height: int) -> np.ndarray:
    rendered = render_svg_to_rgba(path, width, height)
    if rendered is None:
        raise RuntimeError(f"render failed: {path}")
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)
    return np.asarray(rendered, dtype=np.uint8)[:, :, 3]


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

    baseline_mask_root = _mask_only_root(selected)
    baseline_mask = out / "mask-baseline.svg"
    ET.ElementTree(baseline_mask_root).write(baseline_mask, encoding="utf-8", xml_declaration=True)

    crisp_mask_root = copy.deepcopy(baseline_mask_root)
    _crisp_mask(crisp_mask_root)
    crisp_mask = out / "mask-crisp.svg"
    ET.ElementTree(crisp_mask_root).write(crisp_mask, encoding="utf-8", xml_declaration=True)

    full_root = ET.parse(selected).getroot()
    _crisp_mask(full_root)
    full_crisp = out / "candidate-crisp-mask.svg"
    ET.ElementTree(full_root).write(full_crisp, encoding="utf-8", xml_declaration=True)

    evidence = {
        "schema": "vektoryum-public15-mask-rasterization-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "baseline_candidate_bytes": int(selected.stat().st_size),
        "crisp_candidate_bytes": int(full_crisp.stat().st_size),
        "baseline_mask": _coverage(source[:, :, 3], _render_alpha(baseline_mask, width, height)),
        "crisp_mask": _coverage(source[:, :, 3], _render_alpha(crisp_mask, width, height)),
        "baseline_512": v3["complete_candidate_512"],
        "baseline_1024": v3["complete_candidate_1024"],
        "crisp_512": _full_topology(full_crisp, source, 512),
        "crisp_1024": _full_topology(full_crisp, source, 1024),
        "support_expected": v3["support_only"].get("expected"),
        "support_alpha_ge_1": v3["support_only"].get("render_alpha_ge_1"),
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-mask-rasterization-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_MASK_RASTERIZATION=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
