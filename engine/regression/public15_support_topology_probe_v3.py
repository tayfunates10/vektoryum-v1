from __future__ import annotations

import copy
import hashlib
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app import alpha_candidate_paint_deficit as deficit
from app.final_artifact_evaluator import _structure_check, evaluate_final_svg
from app.pipeline_entry import run_pipeline
from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from app.transform_journal import TransformJournal, _measure_svg_bytes
from benchmark.pipeline_results import run_case
from engine.regression.public15_support_topology_probe import _full_topology, _label_signature
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _snapshot_labels(labels: np.ndarray, palette: np.ndarray, stats: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(labels, dtype=np.int32)
    palette_array = np.asarray(palette, dtype=np.uint8)
    return {
        "shape": list(array.shape),
        "palette": palette_array.tolist(),
        "stats": dict(stats),
        "signature": _label_signature(array, len(palette_array)),
        "labels_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        "labels": array.copy(),
    }


def _bind_selected_capture(
    selected: dict[str, Any],
    label_captures: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    index = int(selected["index"])
    snapshot = label_captures.get(index)
    if snapshot is None:
        raise RuntimeError(f"public15 diagnostic missing label binding for candidate {index}")

    geometry = selected.get("geometry") or {}
    expected_pixels = geometry.get("paint_deficit_pixel_count")
    observed_pixels = (snapshot.get("stats") or {}).get("paint_deficit_pixel_count")
    if (
        expected_pixels is not None
        and observed_pixels is not None
        and int(expected_pixels) != int(observed_pixels)
    ):
        raise RuntimeError(
            "public15 diagnostic selected-label binding mismatch:"
            f"candidate={index}:geometry={expected_pixels}:labels={observed_pixels}"
        )
    return snapshot


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source_path = corpus / case.source_path
    source_rgba_full = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)

    captures: list[dict[str, Any]] = []
    label_capture: dict[str, Any] = {}
    label_captures: dict[int, dict[str, Any]] = {}
    original_labels = deficit._paint_deficit_labels
    original_builder = deficit.build_paint_deficit_reconstruction_tree

    def wrapped_labels(source_rgba, artwork_rgba):
        labels, palette, stats = original_labels(source_rgba, artwork_rgba)
        label_capture.clear()
        label_capture.update(_snapshot_labels(labels, palette, stats))
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
        if "labels" not in label_capture:
            raise RuntimeError(
                f"public15 diagnostic builder {index} completed without a label snapshot"
            )
        label_captures[index] = {
            key: (value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))
            for key, value in label_capture.items()
        }

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
            "label_snapshot_sha256": label_captures[index]["labels_sha256"],
            "label_snapshot_deficit_pixels": int(
                label_captures[index]["stats"]["paint_deficit_pixel_count"]
            ),
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
    selected_labels = _bind_selected_capture(selected, label_captures)
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
            "shape": selected_labels["shape"],
            "signature": selected_labels["signature"],
            "labels_sha256": selected_labels["labels_sha256"],
            "deficit_pixels": int(selected_labels["stats"]["paint_deficit_pixel_count"]),
        }
    }
    labels = selected_labels["labels"]
    palette = np.asarray(selected_labels["palette"], dtype=np.uint8)
    gh, gw = labels.shape
    rendered = render_svg_to_rgba(support_path, gw, gh)
    if rendered is None:
        support_evidence["render_status"] = "render_failed"
    else:
        if rendered.shape[:2] != (gh, gw):
            rendered = resize_rgba(rendered, gw, gh)
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
        "public15_support_topology_diagnostic_v3",
        parent_path,
        candidate_path,
        transform_report={"diagnostic_only": True, "label_binding": "selected-candidate-v1"},
    )
    parent_measure = _measure_svg_bytes(parent_path.read_bytes(), source_rgb_full, max_side=512)
    candidate_measure = _measure_svg_bytes(candidate_path.read_bytes(), source_rgb_full, max_side=512)
    struct, _messages, codes, _ = _structure_check(candidate_path.read_bytes())

    evidence: dict[str, object] = {
        "schema": "vektoryum-public15-support-topology-proof-v2",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "source_size": [int(source_rgba_full.shape[1]), int(source_rgba_full.shape[0])],
        "pipeline": pipeline,
        "captured_candidates": captures,
        "selected_candidate": selected,
        "selected_label_binding": {
            "candidate_index": int(selected["index"]),
            "labels_sha256": selected_labels["labels_sha256"],
            "geometry_deficit_pixels": int(selected["geometry"]["paint_deficit_pixel_count"]),
            "label_deficit_pixels": int(selected_labels["stats"]["paint_deficit_pixel_count"]),
            "matched": True,
        },
        "label_capture": {
            key: value for key, value in selected_labels.items() if key != "labels"
        },
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
    (out / "public15-support-topology-proof-v3.json").write_text(
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
    print("PUBLIC15_PROOF_V3=" + json.dumps({
        "pipeline_status": evidence["pipeline"]["status"],
        "candidate_bytes": evidence["selected_candidate"]["bytes"],
        "selected_label_binding": evidence["selected_label_binding"],
        "support_expected": evidence["support_only"].get("expected"),
        "support_alpha_ge_1": evidence["support_only"].get("render_alpha_ge_1"),
        "candidate_512": evidence["complete_candidate_512"],
        "candidate_1024": evidence["complete_candidate_1024"],
        "journal_reasons": stage.get("reason_codes"),
        "evaluator_hard_fail_codes": evaluator.get("hard_fail_codes"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
