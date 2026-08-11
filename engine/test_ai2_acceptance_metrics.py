from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from ai2_acceptance_metrics import build_acceptance_report, validate_report  # noqa: E402


def test_issue_137_named_canonical_reproducers_meet_all_native_256_gates(tmp_path: Path) -> None:
    report = build_acceptance_report(tmp_path / "first")

    assert validate_report(report) == []
    assert set(report["cases"]) == {
        "qa-lowres-badge",
        "qa-gray-border-counter",
        "neutral-tone-steps",
    }
    for case in report["cases"].values():
        assert case["fixture_kind"] == "canonical_reproducer_not_historical_asset"
        assert case["native_size"] == [256, 256]
        assert case["neutral_band_count"] >= 3
        assert case["after"]["source_cc_recall"] == 1.0
        assert case["after"]["render_cc_precision"] == 1.0
        assert case["after"]["min_true_cc_iou"] >= 0.95
        assert case["after"]["visible_residual"] <= 0.01
        assert case["after"]["de00_p95"] <= 1.0
        assert case["after"]["boundary_p95_px"] <= 0.75


def test_issue_137_native_256_evidence_is_deterministic_across_two_loops(tmp_path: Path) -> None:
    first = build_acceptance_report(tmp_path / "first")
    second = build_acceptance_report(tmp_path / "second")

    assert first == second
