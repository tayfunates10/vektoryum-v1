from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.source_truth import render_svg_to_rgba, resize_rgba
from engine.regression.public15_contour_compositing_probe import _capture_q32, _write_variants


def _render_alpha(path: Path, width: int, height: int) -> np.ndarray:
    rendered = render_svg_to_rgba(path, width, height)
    if rendered is None:
        raise RuntimeError(f"public15 double attenuation render failed: {path.name}")
    if rendered.shape[:2] != (height, width):
        rendered = resize_rgba(rendered, width, height)
    return rendered[:, :, 3].astype(np.float32) / 255.0


def _summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(flat)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def run_probe(corpus: Path, out: Path, engine_version: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    baseline, _parent, source_rgba = _capture_q32(corpus, out, engine_version)
    variants = _write_variants(baseline, out)
    height, width = source_rgba.shape[:2]

    source = source_rgba[:, :, 3].astype(np.float32) / 255.0
    final_alpha = _render_alpha(variants["baseline"], width, height)
    artwork_alpha = _render_alpha(variants["artwork_unmasked"], width, height)
    mask_alpha = _render_alpha(variants["mask_on_solid"], width, height)

    predicted = artwork_alpha * mask_alpha
    product_error = np.abs(final_alpha - predicted)
    source_error = np.abs(source - final_alpha)
    mask_source_error = np.abs(source - mask_alpha)

    aa = (artwork_alpha > (1.0 / 255.0)) & (artwork_alpha < (254.0 / 255.0)) & (mask_alpha > 0.0)
    expected_extra_attenuation = mask_alpha * (1.0 - artwork_alpha)
    observed_mask_to_final_drop = np.maximum(mask_alpha - final_alpha, 0.0)

    if bool(np.any(aa)):
        x = expected_extra_attenuation[aa].reshape(-1)
        y = observed_mask_to_final_drop[aa].reshape(-1)
        correlation = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and float(np.std(x)) > 0 and float(np.std(y)) > 0 else 0.0
        aa_product_error = _summary(product_error[aa])
        aa_expected_drop = _summary(expected_extra_attenuation[aa])
        aa_observed_drop = _summary(observed_mask_to_final_drop[aa])
        aa_count = int(np.count_nonzero(aa))
    else:
        correlation = 0.0
        aa_product_error = _summary(np.empty((0,), dtype=np.float32))
        aa_expected_drop = _summary(np.empty((0,), dtype=np.float32))
        aa_observed_drop = _summary(np.empty((0,), dtype=np.float32))
        aa_count = 0

    product_mae = float(np.mean(product_error))
    confirmed = bool(product_mae <= (1.5 / 255.0) and aa_count > 0 and correlation >= 0.98)

    evidence: dict[str, Any] = {
        "schema": "vektoryum-public15-double-attenuation-proof-v1",
        "diagnostic_only": True,
        "case_id": "qualification-public-15",
        "engine_version": engine_version,
        "mechanism": "artwork_coverage_multiplied_by_source_alpha_mask",
        "mechanism_confirmed": confirmed,
        "full_frame": {
            "product_error": _summary(product_error),
            "product_mae": product_mae,
            "source_to_final_error": _summary(source_error),
            "source_to_mask_error": _summary(mask_source_error),
        },
        "artwork_aa_region": {
            "pixel_count": aa_count,
            "expected_vs_observed_drop_correlation": correlation,
            "product_error": aa_product_error,
            "expected_extra_attenuation": aa_expected_drop,
            "observed_mask_to_final_drop": aa_observed_drop,
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
    (out / "public15-double-attenuation-proof.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    evidence = run_probe(
        Path(os.environ["RFV_CORPUS"]),
        Path(os.environ["PROOF_OUT"]),
        os.environ["ENGINE_VERSION"],
    )
    print("PUBLIC15_DOUBLE_ATTENUATION=" + json.dumps({
        "mechanism_confirmed": evidence["mechanism_confirmed"],
        "full_frame": evidence["full_frame"],
        "artwork_aa_region": evidence["artwork_aa_region"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
