from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from app import alpha_candidate_paint_deficit as deficit
from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_support_topology_probe_v3 import run_probe
from engine.regression.rfv3_measurement_runner import load_qualification_cases


def main() -> None:
    corpus = Path(os.environ["RFV_CORPUS"])
    out = Path(os.environ["PROOF_OUT"])
    engine_version = os.environ["ENGINE_VERSION"]
    out.mkdir(parents=True, exist_ok=True)

    v3_out = out / "v3"
    v3 = run_probe(corpus, v3_out, engine_version)
    selected = Path(str(v3["selected_candidate"]["path"]))

    case = next(
        item for item in load_qualification_cases(corpus)
        if item.case_id == "qualification-public-15"
    )
    source = np.asarray(Image.open(corpus / case.source_path).convert("RGBA"), dtype=np.uint8)
    height, width = source.shape[:2]
    rendered = render_svg_to_rgba(selected, width, height)
    if rendered is None:
        raise RuntimeError("public15 alpha coverage probe render failed")
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)

    source_positive = source[:, :, 3] > 0
    render_positive = rendered[:, :, 3] > 0
    false_negative = source_positive & ~render_positive
    false_positive = render_positive & ~source_positive

    source_white = deficit._composite_on_white(source)
    source_visible = np.any(
        np.abs(source_white.astype(np.int16) - 255) > 12,
        axis=2,
    )
    near_white_source = source_positive & ~source_visible

    fn_count = int(np.count_nonzero(false_negative))
    fn_visible = int(np.count_nonzero(false_negative & source_visible))
    fn_near_white = int(np.count_nonzero(false_negative & near_white_source))

    alpha_abs = np.abs(
        source[:, :, 3].astype(np.int16) - rendered[:, :, 3].astype(np.int16)
    )
    evidence = {
        "schema": "vektoryum-public15-alpha-coverage-proof-v1",
        "diagnostic_only": True,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "candidate_bytes": int(selected.stat().st_size),
        "source_positive_pixels": int(np.count_nonzero(source_positive)),
        "render_positive_pixels": int(np.count_nonzero(render_positive)),
        "false_negative_pixels": fn_count,
        "false_positive_pixels": int(np.count_nonzero(false_positive)),
        "false_negative_visible_source_pixels": fn_visible,
        "false_negative_near_white_source_pixels": fn_near_white,
        "false_negative_near_white_ratio": (
            float(fn_near_white) / float(fn_count) if fn_count else 0.0
        ),
        "source_near_white_positive_pixels": int(np.count_nonzero(near_white_source)),
        "alpha_abs_error_mean": float(alpha_abs.mean()),
        "alpha_abs_error_p95": float(np.percentile(alpha_abs, 95)),
        "alpha_abs_error_max": int(alpha_abs.max()),
        "support_expected": v3["support_only"].get("expected"),
        "support_alpha_ge_1": v3["support_only"].get("render_alpha_ge_1"),
        "complete_candidate_512": v3["complete_candidate_512"],
        "complete_candidate_1024": v3["complete_candidate_1024"],
        "evaluator_hard_fail_codes": v3["final_artifact_evaluator"].get("hard_fail_codes"),
        "journal_reason_codes": v3["transform_journal"]["stage"].get("reason_codes"),
    }
    (out / "public15-alpha-coverage-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PUBLIC15_ALPHA_COVERAGE=" + json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
