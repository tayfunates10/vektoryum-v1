from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

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
    raise RuntimeError(f"public15 mask multiplication probe missing mask {mask_id}")


def _replace_mask_with_white(root: ET.Element, mask_id: str) -> None:
    mask = _find_mask(root, mask_id)
    for child in list(mask):
        mask.remove(child)
    ns = mask.tag.split("}", 1)[0].removeprefix("{") if "}" in mask.tag else "http://www.w3.org/2000/svg"
    width = root.attrib.get("width", "1600")
    height = root.attrib.get("height", "1600")
    ET.SubElement(mask, f"{{{ns}}}rect", {
        "x": "0", "y": "0", "width": width, "height": height, "fill": "white",
        "data-public15-diagnostic": "neutral-white-mask",
    })


def _scale_mask_gray(root: ET.Element, mask_id: str, factor: float) -> int:
    mask = _find_mask(root, mask_id)
    touched = 0
    for element in mask.iter():
        fill = element.attrib.get("fill", "")
        if not fill.startswith("rgb("):
            continue
        parts = fill.removeprefix("rgb(").removesuffix(")").split(",")
        if len(parts) != 3 or not (parts[0] == parts[1] == parts[2]):
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        scaled = max(0, min(255, int(round(value * factor))))
        element.set("fill", f"rgb({scaled},{scaled},{scaled})")
        touched += 1
    if touched <= 0:
        raise RuntimeError("public15 mask multiplication probe found no grayscale mask fills")
    return touched


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, parent, source_rgba = _capture_q32(corpus, out, engine_version)
    base_root = ET.parse(baseline).getroot()
    _application, mask_id = _find_mask_application(base_root)

    variants: dict[str, Path] = {"baseline": baseline}

    neutral_root = copy.deepcopy(base_root)
    _replace_mask_with_white(neutral_root, mask_id)
    neutral_path = out / "contour-q32-neutral-white-mask.svg"
    ET.ElementTree(neutral_root).write(neutral_path, encoding="utf-8", xml_declaration=True)
    variants["neutral_white_mask"] = neutral_path

    for factor in (0.5, 0.75, 1.25, 1.5):
        root = copy.deepcopy(base_root)
        touched = _scale_mask_gray(root, mask_id, factor)
        path = out / f"contour-q32-mask-gray-x{factor:g}.svg"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        variants[f"mask_gray_x{factor:g}"] = path

    measured = {
        name: _measure_variant(name, path, parent, source_rgba)
        for name, path in variants.items()
    }
    evidence = {
        "schema": "vektoryum-public15-mask-multiplication-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "variants": measured,
        "interpretation_contract": {
            "neutral_white_matches_unmasked": "seam is caused by non-unity mask alpha multiplication, not mask application plumbing",
            "neutral_white_keeps_baseline_seam": "seam is caused by mask application/compositing plumbing independent of mask alpha values",
            "gray_scale_monotonic": "supports multiplicative attenuation as the seam mechanism",
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
    (out / "public15-mask-multiplication-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    evidence = run_probe(
        Path(os.environ["RFV_CORPUS"]),
        Path(os.environ["PROOF_OUT"]),
        os.environ["ENGINE_VERSION"],
    )
    print("PUBLIC15_MASK_MULTIPLICATION=" + json.dumps({
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
