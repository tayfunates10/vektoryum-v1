from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.final_artifact_evaluator import _classify, _derive_palette
from app.source_truth import composite_rgba, render_svg_to_rgba, resize_rgba
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def _component_records(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=4
    )
    records: list[dict[str, int]] = []
    remap = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for component_id in range(1, int(count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        remap[labels == component_id] = next_id
        records.append({
            "id": next_id,
            "area": area,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })
        next_id += 1
    return remap, records


def _overlap_ids(labels: np.ndarray, component_mask: np.ndarray) -> list[int]:
    values = np.unique(labels[np.asarray(component_mask, dtype=bool)])
    return [int(value) for value in values if int(value) > 0]


def _bbox_alpha_stats(
    source_alpha: np.ndarray,
    render_alpha: np.ndarray,
    record: dict[str, int],
) -> dict[str, float | int]:
    x = int(record["x"])
    y = int(record["y"])
    width = int(record["width"])
    height = int(record["height"])
    src = source_alpha[y:y + height, x:x + width].astype(np.int16)
    rnd = render_alpha[y:y + height, x:x + width].astype(np.int16)
    diff = np.abs(src - rnd)
    return {
        "source_alpha_min": int(src.min()) if src.size else 0,
        "source_alpha_max": int(src.max()) if src.size else 0,
        "render_alpha_min": int(rnd.min()) if rnd.size else 0,
        "render_alpha_max": int(rnd.max()) if rnd.size else 0,
        "alpha_abs_error_mean": float(diff.mean()) if diff.size else 0.0,
        "alpha_abs_error_max": int(diff.max()) if diff.size else 0,
    }


def _analyze_scale(candidate: Path, source_rgba_full: np.ndarray, max_side: int) -> dict[str, object]:
    h0, w0 = source_rgba_full.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h0, w0)))
    width = max(1, int(round(w0 * scale)))
    height = max(1, int(round(h0 * scale)))
    source_rgba = resize_rgba(source_rgba_full, width, height)
    rendered_rgba = render_svg_to_rgba(candidate, width, height)
    if rendered_rgba is None:
        return {"status": "render_failed", "width": width, "height": height}
    if rendered_rgba.shape[:2] != (height, width):
        rendered_rgba = resize_rgba(rendered_rgba, width, height)

    source_rgb = composite_rgba(source_rgba, 255)
    rendered_rgb = composite_rgba(rendered_rgba, 255)
    palette = _derive_palette(source_rgb)
    source_classes = _classify(source_rgb, palette)
    render_classes = _classify(rendered_rgb, palette)
    min_area = max(6, round(0.00004 * width * height))

    labels_evidence: list[dict[str, object]] = []
    total_source = 0
    total_render = 0
    total_splits = 0
    total_merges = 0
    total_unmatched_render = 0

    for palette_label in range(1, int(len(palette)) + 1):
        src_labels, src_records = _component_records(source_classes == palette_label, min_area)
        rnd_labels, rnd_records = _component_records(render_classes == palette_label, min_area)
        total_source += len(src_records)
        total_render += len(rnd_records)
        if not src_records and not rnd_records:
            continue

        split_records: list[dict[str, object]] = []
        for record in src_records:
            component = src_labels == int(record["id"])
            overlaps = _overlap_ids(rnd_labels, component)
            if len(overlaps) > 1:
                split_records.append({**record, "render_component_ids": overlaps})

        merge_records: list[dict[str, object]] = []
        unmatched_render: list[dict[str, object]] = []
        for record in rnd_records:
            component = rnd_labels == int(record["id"])
            overlaps = _overlap_ids(src_labels, component)
            enriched = {
                **record,
                "source_component_ids": overlaps,
                "alpha": _bbox_alpha_stats(
                    source_rgba[:, :, 3], rendered_rgba[:, :, 3], record
                ),
            }
            if len(overlaps) > 1:
                merge_records.append(enriched)
            elif not overlaps:
                unmatched_render.append(enriched)

        total_splits += len(split_records)
        total_merges += len(merge_records)
        total_unmatched_render += len(unmatched_render)
        labels_evidence.append({
            "palette_label": palette_label,
            "palette_rgb": [int(v) for v in palette[palette_label - 1]],
            "source_component_count": len(src_records),
            "render_component_count": len(rnd_records),
            "component_delta_signed": len(rnd_records) - len(src_records),
            "source_splits": split_records,
            "render_merges": merge_records,
            "unmatched_render_components": unmatched_render,
        })

    return {
        "status": "measured",
        "width": width,
        "height": height,
        "min_area": min_area,
        "palette_count": int(len(palette)),
        "source_component_count": total_source,
        "render_component_count": total_render,
        "component_delta_signed": total_render - total_source,
        "split_source_component_count": total_splits,
        "merged_render_component_count": total_merges,
        "unmatched_render_component_count": total_unmatched_render,
        "labels": labels_evidence,
    }


def main() -> None:
    corpus = Path(os.environ["RFV_CORPUS"])
    out = Path(os.environ["PROOF_OUT"])
    engine_version = os.environ["ENGINE_VERSION"]
    out.mkdir(parents=True, exist_ok=True)

    v3 = run_probe(corpus, out / "v3", engine_version)
    candidate = Path(str(v3["selected_candidate"]["path"]))
    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source = np.asarray(Image.open(corpus / case.source_path).convert("RGBA"), dtype=np.uint8)

    evidence = {
        "schema": "vektoryum-public15-topology-component-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "candidate_bytes": int(candidate.stat().st_size),
        "scale_512": _analyze_scale(candidate, source, 512),
        "scale_1024": _analyze_scale(candidate, source, 1024),
        "invariants": {
            "thresholds_changed": False,
            "budgets_changed": False,
            "evaluator_changed": False,
            "journal_changed": False,
            "production_code_changed_by_probe": False,
        },
    }
    (out / "public15-topology-component-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PUBLIC15_TOPOLOGY_COMPONENT=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
