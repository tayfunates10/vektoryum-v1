from __future__ import annotations

import copy
import hashlib
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import alpha_candidate_paint_deficit as deficit
from app.final_artifact_evaluator import (
    _classify,
    _derive_palette,
    _seam_ratio,
    _structure_check,
    _topology_signature,
    evaluate_final_svg,
)
from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from app.transform_journal import TransformJournal, _measure_svg_bytes
from app.pipeline_entry import run_pipeline
from benchmark.pipeline_results import run_case
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _label_signature(labels: np.ndarray, palette_count: int) -> dict[str, int]:
    arr = np.asarray(labels, dtype=np.int32)
    components = 0
    holes = 0
    for label in range(1, int(palette_count) + 1):
        mask = (arr == label).astype(np.uint8)
        if not mask.any():
            continue
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        components += sum(int(stats[i, cv2.CC_STAT_AREA]) >= 1 for i in range(1, n))
        inv = (1 - mask).astype(np.uint8)
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        nh, labh, statsh, _ = cv2.connectedComponentsWithStats(padded, connectivity=4)
        outer = int(labh[0, 0])
        holes += sum(
            i != outer and int(statsh[i, cv2.CC_STAT_AREA]) >= 1
            for i in range(1, nh)
        )
    return {"components": int(components), "holes": int(max(0, holes))}


def _full_topology(path: Path, source_rgba: np.ndarray, max_side: int) -> dict[str, object]:
    h0, w0 = source_rgba.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    w = max(1, int(round(w0 * scale)))
    h = max(1, int(round(h0 * scale)))
    src_rgba = resize_rgba(source_rgba, w, h)
    src_rgb = composite_rgba(src_rgba, 255)
    rnd_rgba = render_svg_to_rgba(path, w, h)
    if rnd_rgba is None:
        return {"status": "render_failed", "width": w, "height": h}
    if rnd_rgba.shape[:2] != (h, w):
        rnd_rgba = resize_rgba(rnd_rgba, w, h)
    rnd_rgb = composite_rgba(rnd_rgba, 255)
    palette = _derive_palette(src_rgb)
    co = _classify(src_rgb, palette)
    cr = _classify(rnd_rgb, palette)
    min_area = max(6, round(0.00004 * w * h))
    src_sig = _topology_signature(co, len(palette), min_area)
    rnd_sig = _topology_signature(cr, len(palette), min_area)
    return {
        "status": "measured",
        "width": w,
        "height": h,
        "min_area": min_area,
        "palette_count": int(len(palette)),
        "source": src_sig,
        "render": rnd_sig,
        "component_delta": abs(src_sig["components"] - rnd_sig["components"]),
        "hole_delta": abs(src_sig["holes"] - rnd_sig["holes"]),
        "seam_ratio": float(_seam_ratio(src_rgb, rnd_rgb)),
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source_path = corpus / case.source_path
    source_rgba_full = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)

    captures: list[dict[str, object]] = []
    label_capture: dict[str, object] = {}
    original_labels = deficit._paint_deficit_labels
    original_builder = deficit.build_paint_deficit_reconstruction_tree

    def wrapped_labels(source_rgba, artwork_rgba):
        labels, palette, stats = original_labels(source_rgba, artwork_rgba)
        label_capture.clear()
        label_capture.update({
            "shape": list(labels.shape),
            "palette": np.asarray(palette, dtype=np.uint8).tolist(),
            "stats": dict(stats),
            "signature": _label_signature(labels, len(palette)),
            "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
            "labels": np.asarray(labels, dtype=np.int32).copy(),
        })
        return labels, palette, stats

    def wrapped_builder(
        original_root,
        canvas_element,
        source_rgba_grid,
        transaction_id,
        mask_encoding="polygon",
        levels=24,
    ):
        root, geometry = original_builder(
            original_root,
            canvas_element,
            source_rgba_grid,
            transaction_id,
            mask_encoding=mask_encoding,
            levels=levels,
        )
        index = len(captures)
        candidate = out / f"candidate-{index:02d}-{mask_encoding}-q{levels}.svg"
        ET.ElementTree(root).write(candidate, encoding="utf-8", xml_declaration=True)
        if index == 0:
            ET.ElementTree(copy.deepcopy(original_root)).write(
                out / "parent.svg", encoding="utf-8", xml_declaration=True
            )
        captures.append({
            "index": index,
            "mask_encoding": str(mask_encoding),
            "levels": int(levels),
            "path": str(candidate),
            "bytes": int(candidate.stat().st_size),
            "geometry": {
                key: value for key, value in geometry.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
        })
        return root, geometry

    deficit._paint_deficit_labels = wrapped_labels
    deficit.build_paint_deficit_reconstruction_tree = wrapped_builder
    try:
        try:
            result = run_case(
                case,
                corpus_root=corpus,
                work_root=out / "pipeline-job",
                pipeline=run_pipeline,
                engine_version=engine_version,
                trace_mode="auto",
                peak_rss_mb=None,
            )
        except BaseException as exc:
            pipeline: dict[str, object] = {
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
                "traceback_tail": traceback.format_exc()[-2500:],
            }
        else:
            pipeline = {"status": "completed", "result": result.to_dict()}
    finally:
        deficit._paint_deficit_labels = original_labels
        deficit.build_paint_deficit_reconstruction_tree = original_builder

    if not captures:
        raise RuntimeError("public15 diagnostic captured no paint-deficit candidates")
    q8 = [
        item for item in captures
        if item["mask_encoding"] == "cumulative" and item["levels"] == 8
    ]
    if not q8:
        raise RuntimeError("public15 diagnostic captured no cumulative q8 candidate")
    selected = min(q8, key=lambda item: int(item["bytes"]))
    candidate_path = Path(str(selected["path"]))
    parent_path = out / "parent.svg"

    tree = ET.parse(candidate_path)
    root = tree.getroot()
    local = lambda name: str(name).rsplit("}", 1)[-1].lower()
    support = next(
        element for element in root.iter()
        if local(element.tag) == "g"
        and element.attrib.get("data-vektoryum-paint-deficit") == "source-palette-v1"
    )
    support_root = ET.Element(root.tag, dict(root.attrib))
    support_root.append(copy.deepcopy(support))
    support_path = out / "support-only.svg"
    ET.ElementTree(support_root).write(support_path, encoding="utf-8", xml_declaration=True)

    support_evidence: dict[str, object] = {
        "expected": {
            "shape": label_capture.get("shape"),
            "signature": label_capture.get("signature"),
            "labels_sha256": label_capture.get("labels_sha256"),
        }
    }
    labels = label_capture.get("labels")
    palette_list = label_capture.get("palette")
    if isinstance(labels, np.ndarray) and isinstance(palette_list, list):
        gh, gw = labels.shape
        rendered = render_svg_to_rgba(support_path, gw, gh)
        if rendered is not None:
            if rendered.shape[:2] != (gh, gw):
                rendered = resize_rgba(rendered, gw, gh)
            palette = np.asarray(palette_list, dtype=np.uint8)
            rgb = rendered[:, :, :3].astype(np.int32)
            distances = (
                (rgb[:, :, None, :] - palette.astype(np.int32)[None, None, :, :]) ** 2
            ).sum(axis=3)
            nearest = np.argmin(distances, axis=2).astype(np.int32) + 1
            for threshold in (1, 128):
                observed = np.zeros((gh, gw), dtype=np.int32)
                occupied = rendered[:, :, 3] >= threshold
                observed[occupied] = nearest[occupied]
                support_evidence[f"render_alpha_ge_{threshold}"] = {
                    "signature": _label_signature(observed, len(palette)),
                    "occupied_pixels": int(np.count_nonzero(occupied)),
                    "label_match_ratio": float(np.mean(observed == labels)),
                }
        else:
            support_evidence["render_status"] = "render_failed"

    source_rgb_full = composite_rgba(source_rgba_full, 255)
    evaluator = evaluate_final_svg(
        candidate_path,
        source_rgb_full,
        source_alpha=source_rgba_full[:, :, 3],
        image_class="clean_logo",
        required_metrics={"alpha_fidelity"},
    ).to_dict()

    journal = TransformJournal(
        parent_path,
        source_rgb_full,
        image_class="clean_logo",
        required_metrics=set(),
        budget_seconds=180.0,
        stage_timeout_seconds=600.0,
    )
    accepted_path, stage = journal.consider_candidate(
        "public15_support_topology_diagnostic",
        parent_path,
        candidate_path,
        transform_report={"diagnostic_only": True},
    )
    parent_measure = _measure_svg_bytes(parent_path.read_bytes(), source_rgb_full, max_side=512)
    candidate_measure = _measure_svg_bytes(candidate_path.read_bytes(), source_rgb_full, max_side=512)
    struct, _messages, codes, _ = _structure_check(candidate_path.read_bytes())

    evidence: dict[str, object] = {
        "schema": "vektoryum-public15-support-topology-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "source_size": [int(source_rgba_full.shape[1]), int(source_rgba_full.shape[0])],
        "pipeline": pipeline,
        "captured_candidates": captures,
        "selected_candidate": selected,
        "label_capture": {key: value for key, value in label_capture.items() if key != "labels"},
        "support_only": support_evidence,
        "complete_candidate_512": _full_topology(candidate_path, source_rgba_full, 512),
        "complete_candidate_1024": _full_topology(candidate_path, source_rgba_full, 1024),
        "structure": {
            "codes": list(codes),
            "byte_size": int(candidate_path.stat().st_size),
            "path_count": int(struct.get("path_count") or 0),
            "node_count": int(struct.get("node_count") or 0),
        },
        "transform_journal": {
            "accepted_candidate": Path(accepted_path) == candidate_path,
            "stage": stage,
            "parent_measurement": parent_measure,
            "candidate_measurement": candidate_measure,
        },
        "final_artifact_evaluator": evaluator,
    }
    (out / "public15-support-topology-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    corpus = Path(os.environ["RFV_CORPUS"])
    out = Path(os.environ["PROOF_OUT"])
    engine_version = os.environ["ENGINE_VERSION"]
    evidence = run_probe(corpus, out, engine_version)
    stage = evidence["transform_journal"]["stage"]
    evaluator = evidence["final_artifact_evaluator"]
    print("PUBLIC15_PROOF=" + json.dumps({
        "pipeline_status": evidence["pipeline"]["status"],
        "candidate_bytes": evidence["selected_candidate"]["bytes"],
        "support_expected": evidence["support_only"].get("expected"),
        "candidate_512": evidence["complete_candidate_512"],
        "candidate_1024": evidence["complete_candidate_1024"],
        "journal_reasons": stage.get("reason_codes"),
        "evaluator_hard_fail_codes": evaluator.get("hard_fail_codes"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
