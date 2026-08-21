from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from app import alpha_candidate_painter as painter
from app.alpha_candidate_knockout import _path_node_counts
from engine.regression.public15_contour_compositing_probe import (
    _capture_q32,
    _find_mask_application,
    _local,
    _measure_variant,
)


def _find_mask(root: ET.Element, mask_id: str) -> ET.Element:
    for element in root.iter():
        if _local(element.tag) == "mask" and element.attrib.get("id") == mask_id:
            return element
    raise RuntimeError(f"public15 alpha ladder probe missing mask {mask_id}")


def _replace_mask_ladder(
    baseline_root: ET.Element,
    mask_id: str,
    source_alpha,
    *,
    levels: int,
    cumulative: bool,
) -> ET.Element:
    root = copy.deepcopy(baseline_root)
    mask = _find_mask(root, mask_id)
    content = next((child for child in list(mask) if _local(child.tag) == "g"), None)
    if content is None:
        raise RuntimeError("public15 alpha ladder probe missing mask content group")

    black_base = next(
        (
            child for child in list(content)
            if _local(child.tag) == "rect" and child.attrib.get("fill") == "rgb(0,0,0)"
        ),
        None,
    )
    if black_base is None:
        raise RuntimeError("public15 alpha ladder probe missing black mask base")
    for child in list(content):
        if child is not black_base:
            content.remove(child)

    quantized, opacity_by_level = painter._requantize_alpha(
        source_alpha,
        levels,
        allow_silhouette_shortcut=False,
    )
    ns = mask.tag.split("}", 1)[0].removeprefix("{") if "}" in mask.tag else "http://www.w3.org/2000/svg"
    qname = lambda name: f"{{{ns}}}{name}"
    if cumulative:
        generated = painter._cumulative_threshold_children(quantized, opacity_by_level, qname)
    else:
        generated = painter._painter_contour_children(quantized, opacity_by_level, qname)
    if generated is None:
        raise RuntimeError("public15 alpha ladder probe generated no mask geometry")
    children, _stats = generated
    for child in children:
        content.append(child)
    return root


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, parent, source_rgba = _capture_q32(corpus, out, engine_version)
    baseline_root = ET.parse(baseline).getroot()
    _application, mask_id = _find_mask_application(baseline_root)

    variants: dict[str, Path] = {"baseline_silhouette_q32": baseline}
    specs = (
        (8, True),
        (16, True),
        (32, True),
        (32, False),
    )
    for levels, cumulative in specs:
        label = f"graded_{'cumulative' if cumulative else 'discrete'}_q{levels}"
        root = _replace_mask_ladder(
            baseline_root,
            mask_id,
            source_rgba[:, :, 3],
            levels=levels,
            cumulative=cumulative,
        )
        path = out / f"contour-q32-{label}.svg"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        variants[label] = path

    measured: dict[str, Any] = {}
    for name, path in variants.items():
        item = _measure_variant(name, path, parent, source_rgba)
        root = ET.parse(path).getroot()
        path_count, node_count = _path_node_counts(root)
        item["path_count"] = int(path_count)
        item["node_count"] = int(node_count)
        measured[name] = item

    evidence = {
        "schema": "vektoryum-public15-contour-alpha-ladder-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "variants": measured,
        "interpretation_contract": {
            "graded_variant_reduces_seam_and_keeps_alpha": (
                "binary silhouette shortcut is the seam mechanism; graded mask encoding is the repair direction"
            ),
            "graded_variant_does_not_reduce_seam": (
                "silhouette shortcut is not sufficient to explain the seam"
            ),
        },
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
    (out / "public15-contour-alpha-ladder-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    evidence = run_probe(
        Path(os.environ["RFV_CORPUS"]),
        Path(os.environ["PROOF_OUT"]),
        os.environ["ENGINE_VERSION"],
    )
    print("PUBLIC15_CONTOUR_ALPHA_LADDER=" + json.dumps({
        name: {
            "bytes": item["bytes"],
            "path_count": item["path_count"],
            "node_count": item["node_count"],
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
