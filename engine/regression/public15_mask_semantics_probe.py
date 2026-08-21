from __future__ import annotations

import copy
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_mask_isolation_probe import _coverage, _mask_only_root
from engine.regression.public15_support_topology_probe import _full_topology
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases

_RGB_RE = re.compile(
    r"^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$",
    re.IGNORECASE,
)


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _alpha_mask_semantics(root: ET.Element) -> int:
    """Convert gray luminance-mask paint into equivalent alpha-mask paint.

    Geometry is intentionally untouched.  A gray value ``g`` that previously
    supplied mask luminance becomes white with ``fill-opacity=g/255`` under an
    explicit alpha mask.  This probe therefore isolates renderer mask semantics
    from the already-proven support geometry and from every acceptance threshold.
    """
    converted = 0
    for mask in root.iter():
        if _local(mask.tag) != "mask":
            continue
        mask.set("mask-type", "alpha")
        for node in mask.iter():
            fill = node.get("fill")
            if not fill:
                continue
            match = _RGB_RE.match(fill.strip())
            if match is None:
                continue
            r, g, b = (int(match.group(i)) for i in range(1, 4))
            if not (r == g == b):
                continue
            value = max(0, min(255, r))
            node.set("fill", "rgb(255,255,255)")
            node.set("fill-opacity", f"{value / 255.0:.12g}")
            converted += 1
    return converted


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
    source = np.asarray(
        Image.open(corpus / case.source_path).convert("RGBA"),
        dtype=np.uint8,
    )
    height, width = source.shape[:2]

    baseline_mask_root = _mask_only_root(selected)
    baseline_mask = out / "mask-luminance.svg"
    ET.ElementTree(baseline_mask_root).write(
        baseline_mask,
        encoding="utf-8",
        xml_declaration=True,
    )

    alpha_mask_root = copy.deepcopy(baseline_mask_root)
    mask_nodes_converted = _alpha_mask_semantics(alpha_mask_root)
    if mask_nodes_converted <= 0:
        raise RuntimeError("public15 mask semantics probe converted no mask paint")
    alpha_mask = out / "mask-alpha.svg"
    ET.ElementTree(alpha_mask_root).write(
        alpha_mask,
        encoding="utf-8",
        xml_declaration=True,
    )

    full_alpha_root = ET.parse(selected).getroot()
    full_nodes_converted = _alpha_mask_semantics(full_alpha_root)
    if full_nodes_converted != mask_nodes_converted:
        raise RuntimeError(
            "public15 mask semantics conversion mismatch:"
            f"{full_nodes_converted}!={mask_nodes_converted}"
        )
    full_alpha = out / "candidate-alpha-mask.svg"
    ET.ElementTree(full_alpha_root).write(
        full_alpha,
        encoding="utf-8",
        xml_declaration=True,
    )

    evidence = {
        "schema": "vektoryum-public15-mask-semantics-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "mask_nodes_converted": int(mask_nodes_converted),
        "baseline_candidate_bytes": int(selected.stat().st_size),
        "alpha_candidate_bytes": int(full_alpha.stat().st_size),
        "baseline_mask": _coverage(
            source[:, :, 3],
            _render_alpha(baseline_mask, width, height),
        ),
        "alpha_mask": _coverage(
            source[:, :, 3],
            _render_alpha(alpha_mask, width, height),
        ),
        "baseline_512": v3["complete_candidate_512"],
        "baseline_1024": v3["complete_candidate_1024"],
        "alpha_512": _full_topology(full_alpha, source, 512),
        "alpha_1024": _full_topology(full_alpha, source, 1024),
        "support_expected": v3["support_only"].get("expected"),
        "support_alpha_ge_1": v3["support_only"].get("render_alpha_ge_1"),
        "invariants": {
            "geometry_changed": False,
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-mask-semantics-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PUBLIC15_MASK_SEMANTICS=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
