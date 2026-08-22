from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from engine.regression.public15_contour_compositing_probe import (
    _capture_q32,
    _local,
    _measure_variant,
)


def _find_mask(root: ET.Element) -> ET.Element:
    mask_ids: set[str] = set()
    for element in root.iter():
        value = element.attrib.get("mask", "")
        if value.startswith("url(#") and value.endswith(")"):
            mask_ids.add(value[5:-1])
    if len(mask_ids) != 1:
        raise RuntimeError(f"expected one applied mask, found {sorted(mask_ids)}")
    mask_id = next(iter(mask_ids))
    for element in root.iter():
        if _local(element.tag) == "mask" and element.attrib.get("id") == mask_id:
            return element
    raise RuntimeError(f"mask definition {mask_id!r} not found")


def _shift_mask_geometry(baseline: Path, dx: float, dy: float, out: Path) -> Path:
    tree = ET.parse(baseline)
    root = tree.getroot()
    mask = _find_mask(root)
    children = list(mask)
    if not children:
        raise RuntimeError("public15 mask phase probe found empty mask")
    ns = mask.tag.split("}", 1)[0].removeprefix("{") if "}" in mask.tag else "http://www.w3.org/2000/svg"
    wrapper = ET.Element(f"{{{ns}}}g", {
        "transform": f"translate({dx:g} {dy:g})",
        "data-public15-diagnostic": "mask-phase-shift",
    })
    for child in children:
        mask.remove(child)
        wrapper.append(child)
    mask.append(wrapper)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, parent, source_rgba = _capture_q32(corpus, out, engine_version)

    # Source-space offsets. Public-15 is rendered down for the journal/evaluator,
    # so +/-1 and +/-2 source pixels sample sub-device phase changes at 512/1024.
    offsets = [
        (0.0, 0.0),
        (-2.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, -2.0), (0.0, -1.0), (0.0, 1.0), (0.0, 2.0),
        (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0),
    ]

    variants: dict[str, dict[str, Any]] = {}
    for dx, dy in offsets:
        label = f"dx{dx:g}_dy{dy:g}".replace("-", "m").replace(".", "p")
        if dx == 0.0 and dy == 0.0:
            path = baseline
        else:
            path = out / f"contour-q32-mask-phase-{label}.svg"
            _shift_mask_geometry(baseline, dx, dy, path)
        measured = _measure_variant(label, path, parent, source_rgba)
        measured["dx"] = dx
        measured["dy"] = dy
        variants[label] = measured

    baseline_item = variants["dx0_dy0"]
    ranked = sorted(
        variants.items(),
        key=lambda item: (
            "seam_regression" in item[1]["journal_reason_codes"],
            len(item[1]["journal_reason_codes"]),
            float((item[1].get("journal_after") or {}).get("seam_ratio") or 1.0),
        ),
    )
    evidence = {
        "schema": "vektoryum-public15-mask-phase-shift-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "baseline": baseline_item,
        "variants": variants,
        "ranking": [name for name, _ in ranked],
        "decision_contract": {
            "phase_causal": "a non-zero mask-only translation removes seam_regression or materially lowers seam while preserving alpha/topology gates",
            "phase_disproven": "all non-zero translations worsen or leave seam unchanged and/or break alpha/topology",
            "not_proven": "mixed results without a gate-preserving improvement",
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
    (out / "public15-mask-phase-shift-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    evidence = run_probe(
        Path(os.environ["RFV_CORPUS"]),
        Path(os.environ["PROOF_OUT"]),
        os.environ["ENGINE_VERSION"],
    )
    print("PUBLIC15_MASK_PHASE_SHIFT=" + json.dumps({
        "ranking": evidence["ranking"],
        "variants": {
            name: {
                "dx": item["dx"],
                "dy": item["dy"],
                "bytes": item["bytes"],
                "native_alpha": item["native_alpha"],
                "topology_512": item["topology_512"],
                "topology_1024": item["topology_1024"],
                "journal_accepted": item["journal_accepted"],
                "journal_reason_codes": item["journal_reason_codes"],
                "journal_after": item["journal_after"],
                "evaluator_hard_fail_codes": item["evaluator_hard_fail_codes"],
            }
            for name, item in evidence["variants"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
