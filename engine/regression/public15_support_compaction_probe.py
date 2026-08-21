from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from app.final_artifact_evaluator import _structure_check, evaluate_final_svg
from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from app.transform_journal import TransformJournal, _measure_svg_bytes
from engine.regression.public15_support_topology_probe import _full_topology
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _local(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _rect_path(rects: list[ET.Element]) -> str:
    commands: list[str] = []
    for rect in rects:
        x = rect.attrib.get("x", "0")
        y = rect.attrib.get("y", "0")
        width = rect.attrib.get("width", "0")
        height = rect.attrib.get("height", "0")
        commands.append(f"M{x} {y}h{width}v{height}h-{width}z")
    return "".join(commands)


def compact_support_rects(root: ET.Element) -> dict[str, int]:
    support = next(
        node for node in root.iter()
        if _local(node.tag) == "g"
        and node.attrib.get("data-vektoryum-paint-deficit") == "source-palette-v1"
    )
    rect_count = 0
    path_count = 0
    for group in list(support):
        if _local(group.tag) != "g":
            continue
        rects = [child for child in list(group) if _local(child.tag) == "rect"]
        if not rects:
            continue
        d = _rect_path(rects)
        for rect in rects:
            group.remove(rect)
        namespace = str(group.tag).split("}", 1)[0] + "}" if "}" in str(group.tag) else ""
        ET.SubElement(group, f"{namespace}path", {"d": d})
        rect_count += len(rects)
        path_count += 1
    return {"compacted_rect_count": rect_count, "support_path_count": path_count}


def _support_only(candidate: Path, output: Path) -> None:
    tree = ET.parse(candidate)
    root = tree.getroot()
    support = next(
        node for node in root.iter()
        if _local(node.tag) == "g"
        and node.attrib.get("data-vektoryum-paint-deficit") == "source-palette-v1"
    )
    support_root = ET.Element(root.tag, dict(root.attrib))
    support_root.append(copy.deepcopy(support))
    ET.ElementTree(support_root).write(output, encoding="utf-8", xml_declaration=True)


def _render_equal(left: Path, right: Path, width: int, height: int) -> dict[str, object]:
    a = render_svg_to_rgba(left, width, height)
    b = render_svg_to_rgba(right, width, height)
    if a is None or b is None:
        return {"measured": False}
    if a.shape[:2] != (height, width):
        a = resize_rgba(a, width, height)
    if b.shape[:2] != (height, width):
        b = resize_rgba(b, width, height)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "measured": True,
        "byte_identical_rgba": bool(np.array_equal(a, b)),
        "max_channel_delta": int(delta.max()),
        "mean_channel_delta": float(delta.mean()),
        "different_channel_values": int(np.count_nonzero(delta)),
    }


def main() -> None:
    corpus = Path(os.environ["RFV_CORPUS"])
    out = Path(os.environ["PROOF_OUT"])
    engine_version = os.environ["ENGINE_VERSION"]
    out.mkdir(parents=True, exist_ok=True)

    v3_out = out / "v3"
    v3 = run_probe(corpus, v3_out, engine_version)
    selected = Path(str(v3["selected_candidate"]["path"]))
    parent = v3_out / "parent.svg"

    compact = out / "candidate-support-path.svg"
    tree = ET.parse(selected)
    stats = compact_support_rects(tree.getroot())
    tree.write(compact, encoding="utf-8", xml_declaration=True)

    original_support = out / "support-original.svg"
    compact_support = out / "support-compact.svg"
    _support_only(selected, original_support)
    _support_only(compact, compact_support)

    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source_rgba = np.asarray(Image.open(corpus / case.source_path).convert("RGBA"), dtype=np.uint8)
    source_rgb = composite_rgba(source_rgba, 255)

    struct, _messages, codes, _ = _structure_check(compact.read_bytes())
    evaluator = evaluate_final_svg(
        compact,
        source_rgb,
        source_alpha=source_rgba[:, :, 3],
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    ).to_dict()
    journal = TransformJournal(
        parent,
        source_rgb,
        image_class="clean_logo",
        required_metrics=set(),
        budget_seconds=180.0,
        stage_timeout_seconds=600.0,
    )
    accepted, stage = journal.consider_candidate(
        "public15_support_path_compaction_diagnostic",
        parent,
        compact,
        transform_report={"diagnostic_only": True, "support_encoding": "compound-path-v1"},
    )

    gh, gw = v3["support_only"]["expected"]["shape"]
    evidence = {
        "schema": "vektoryum-public15-support-path-compaction-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "baseline_candidate_bytes": int(selected.stat().st_size),
        "compact_candidate_bytes": int(compact.stat().st_size),
        "saved_bytes": int(selected.stat().st_size - compact.stat().st_size),
        "compaction": stats,
        "support_render_equivalence": _render_equal(original_support, compact_support, int(gw), int(gh)),
        "complete_candidate_512": _full_topology(compact, source_rgba, 512),
        "complete_candidate_1024": _full_topology(compact, source_rgba, 1024),
        "structure": {
            "codes": list(codes),
            "byte_size": int(compact.stat().st_size),
            "path_count": int(struct.get("path_count") or 0),
            "node_count": int(struct.get("node_count") or 0),
        },
        "transform_journal": {
            "accepted_candidate": Path(accepted) == compact,
            "stage": stage,
            "candidate_measurement": _measure_svg_bytes(compact.read_bytes(), source_rgb, max_side=512),
        },
        "final_artifact_evaluator": evaluator,
    }
    (out / "public15-support-path-compaction-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_SUPPORT_PATH_COMPACTION=" + json.dumps({
        "baseline_bytes": evidence["baseline_candidate_bytes"],
        "compact_bytes": evidence["compact_candidate_bytes"],
        "support_equivalence": evidence["support_render_equivalence"],
        "candidate_512": evidence["complete_candidate_512"],
        "candidate_1024": evidence["complete_candidate_1024"],
        "journal_reasons": stage.get("reason_codes"),
        "evaluator_hard_fail_codes": evaluator.get("hard_fail_codes"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
