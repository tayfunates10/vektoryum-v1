from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.final_artifact_evaluator import (
    _classify,
    _derive_palette,
    _seam_ratio,
    _topology_signature,
)
from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _render(path: Path, width: int, height: int) -> np.ndarray:
    rgba = render_svg_to_rgba(path, width, height)
    if rgba is None:
        raise RuntimeError(f"composition isolation render failed: {path.name}")
    if rgba.shape[:2] != (height, width):
        rgba = resize_rgba(rgba, width, height)
    return np.asarray(rgba, dtype=np.uint8)


def _coverage(source: np.ndarray, rendered: np.ndarray) -> dict[str, int | float]:
    source_positive = source[:, :, 3] > 0
    render_positive = rendered[:, :, 3] > 0
    alpha_abs = np.abs(
        source[:, :, 3].astype(np.int16) - rendered[:, :, 3].astype(np.int16)
    )
    return {
        "source_positive_pixels": int(np.count_nonzero(source_positive)),
        "render_positive_pixels": int(np.count_nonzero(render_positive)),
        "false_negative_pixels": int(np.count_nonzero(source_positive & ~render_positive)),
        "false_positive_pixels": int(np.count_nonzero(render_positive & ~source_positive)),
        "alpha_abs_error_mean": float(alpha_abs.mean()),
        "alpha_abs_error_p95": float(np.percentile(alpha_abs, 95)),
        "alpha_abs_error_max": int(alpha_abs.max()),
    }


def _topology(source: np.ndarray, rendered: np.ndarray, max_side: int) -> dict[str, object]:
    h0, w0 = source.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    width = max(1, int(round(w0 * scale)))
    height = max(1, int(round(h0 * scale)))
    src = resize_rgba(source, width, height)
    rnd = resize_rgba(rendered, width, height)
    src_rgb = composite_rgba(src, 255)
    rnd_rgb = composite_rgba(rnd, 255)
    palette = _derive_palette(src_rgb)
    src_classes = _classify(src_rgb, palette)
    rnd_classes = _classify(rnd_rgb, palette)
    min_area = max(6, round(0.00004 * width * height))
    src_sig = _topology_signature(src_classes, len(palette), min_area)
    rnd_sig = _topology_signature(rnd_classes, len(palette), min_area)
    return {
        "width": width,
        "height": height,
        "source": src_sig,
        "render": rnd_sig,
        "component_delta": abs(src_sig["components"] - rnd_sig["components"]),
        "hole_delta": abs(src_sig["holes"] - rnd_sig["holes"]),
        "seam_ratio": float(_seam_ratio(src_rgb, rnd_rgb)),
    }


def _write_variant(root: ET.Element, path: Path) -> None:
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _variant_roots(candidate: Path) -> dict[str, ET.Element]:
    original = ET.parse(candidate).getroot()

    def locate(root: ET.Element) -> tuple[ET.Element, ET.Element]:
        layer = next(
            element
            for element in root.iter()
            if _local(element.tag) == "g"
            and element.attrib.get("data-vektoryum-source-alpha-reconstruction")
            == "paint-deficit-q24-v1"
            and "mask" in element.attrib
        )
        support = next(
            element
            for element in list(layer)
            if _local(element.tag) == "g"
            and element.attrib.get("data-vektoryum-paint-deficit") == "source-palette-v1"
        )
        return layer, support

    variants: dict[str, ET.Element] = {"full": copy.deepcopy(original)}

    masked_paint = copy.deepcopy(original)
    layer, support = locate(masked_paint)
    layer.remove(support)
    variants["masked_paint_only"] = masked_paint

    unmasked_paint = copy.deepcopy(masked_paint)
    layer, _ = next(
        (element, None)
        for element in unmasked_paint.iter()
        if _local(element.tag) == "g"
        and element.attrib.get("data-vektoryum-source-alpha-reconstruction")
        == "paint-deficit-q24-v1"
        and "mask" in element.attrib
    )
    layer.attrib.pop("mask", None)
    variants["unmasked_paint_only"] = unmasked_paint

    support_outside = copy.deepcopy(original)
    layer, support = locate(support_outside)
    support_copy = copy.deepcopy(support)
    layer.remove(support)
    # Diagnostic counterfactual only: keep the preserved paint masked, but move
    # the exact deficit support outside that mask. This must never be accepted
    # by the probe itself; it only identifies whether parent mask attenuation is
    # the first composition divergence.
    children = list(support_outside)
    layer_index = children.index(layer)
    support_outside.insert(layer_index + 1, support_copy)
    variants["masked_paint_plus_unmasked_support"] = support_outside

    return variants


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
        Image.open(corpus / case.source_path).convert("RGBA"), dtype=np.uint8
    )
    height, width = source.shape[:2]

    variants = _variant_roots(selected)
    evidence_variants: dict[str, object] = {}
    for name, root in variants.items():
        path = out / f"composition-{name}.svg"
        _write_variant(root, path)
        rendered = _render(path, width, height)
        evidence_variants[name] = {
            "bytes": int(path.stat().st_size),
            "coverage": _coverage(source, rendered),
            "topology_512": _topology(source, rendered, 512),
            "topology_1024": _topology(source, rendered, 1024),
        }

    evidence = {
        "schema": "vektoryum-public15-composition-isolation-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "selected_candidate_bytes": int(selected.stat().st_size),
        "support_expected": v3["support_only"].get("expected"),
        "support_alpha_ge_1": v3["support_only"].get("render_alpha_ge_1"),
        "variants": evidence_variants,
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-composition-isolation-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_COMPOSITION_ISOLATION=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
