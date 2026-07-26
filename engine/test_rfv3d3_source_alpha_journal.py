from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _backgrounds(*, ssim: float, ms_ssim: float, mae: float, p95: float):
    return {
        name: {
            "ssim": ssim,
            "ms_ssim": ms_ssim,
            "rgb_mae": mae,
            "rgb_p95": p95,
        }
        for name in ("white", "black", "checker")
    }


def _report(*, schema: str = "rfv3d3-soft-ellipse-alpha-v1") -> dict:
    comparison = {
        background: {
            "ssim": {"parent": 0.95, "candidate": 0.97, "non_regressing": True},
            "ms_ssim": {"parent": 0.96, "candidate": 0.98, "non_regressing": True},
            "rgb_mae": {"parent": 0.03, "candidate": 0.02, "non_regressing": True},
            "rgb_p95": {"parent": 0.08, "candidate": 0.06, "non_regressing": True},
        }
        for background in ("white", "black", "checker")
    }
    report = {
        "schema": schema,
        "status": "accepted",
        "applied": True,
        "candidate_geometry_preserved": True,
        "candidate_path_data_preserved": True,
        "candidate_identity_preserved": True,
        "artwork_identity_preserved": True,
        "source_truth_alpha_iou": 0.999,
        "source_truth_alpha_mae": 0.001,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_iou": 0.999,
        "final_evaluator_alpha_mae": 0.001,
        "final_evaluator_alpha_plane_hard_fail_codes": [],
        "source_alpha_background_non_regression": True,
        "source_alpha_background_failure_codes": [],
        "source_alpha_background_comparison": comparison,
        "source_alpha_background_authority": (
            "final_artifact_evaluator_parent_candidate_exact_non_regression"
        ),
    }
    if schema == "rfv3d2-candidate-painter-reconstruction-v1":
        report["trace_rgb_bytes_preserved"] = True
    return report


def _metric(data: bytes, *, candidate: bool, ssim: float = 0.99, byte_size: int | None = None):
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_size": len(data) if byte_size is None else byte_size,
        "structural_safe": True,
        "structural_failure_codes": [],
        "path_count": 1,
        "node_count": 5,
        "gradient_definition_count": 0,
        "required_unmeasured": [],
        "ssim": ssim,
        "edge_f1_1px": 0.94 if candidate else 0.995,
        "seam_ratio": 0.02 if candidate else 0.001,
        "component_delta": 8 if candidate else 0,
        "hole_delta": 3 if candidate else 0,
    }


def test_background_non_regression_uses_exact_existing_metrics() -> None:
    from app.alpha_candidate_validation import alpha_background_non_regression

    parent = SimpleNamespace(
        metrics={"G_gradient_alpha": {"backgrounds": _backgrounds(
            ssim=0.95, ms_ssim=0.96, mae=0.03, p95=0.08
        )}}
    )
    candidate = SimpleNamespace(
        metrics={"G_gradient_alpha": {"backgrounds": _backgrounds(
            ssim=0.97, ms_ssim=0.98, mae=0.02, p95=0.06
        )}}
    )
    proof = alpha_background_non_regression(parent, candidate)
    assert proof["source_alpha_background_non_regression"] is True
    assert proof["source_alpha_background_failure_codes"] == []
    assert all(
        metric["non_regressing"] is True
        for background in proof["source_alpha_background_comparison"].values()
        for metric in background.values()
    )


def test_background_regression_fails_closed_without_tolerance() -> None:
    from app.alpha_candidate_validation import alpha_background_non_regression

    parent = SimpleNamespace(
        metrics={"G_gradient_alpha": {"backgrounds": _backgrounds(
            ssim=0.95, ms_ssim=0.96, mae=0.03, p95=0.08
        )}}
    )
    candidate = SimpleNamespace(
        metrics={"G_gradient_alpha": {"backgrounds": _backgrounds(
            ssim=0.949999, ms_ssim=0.98, mae=0.02, p95=0.06
        )}}
    )
    proof = alpha_background_non_regression(parent, candidate)
    assert proof["source_alpha_background_non_regression"] is False
    assert "source_alpha_white_ssim_regression" in proof[
        "source_alpha_background_failure_codes"
    ]


@pytest.mark.parametrize(
    "stage_id,schema",
    [
        ("source_alpha_vector_mask", "rfv3d3-soft-ellipse-alpha-v1"),
        (
            "source_alpha_painter_candidate",
            "rfv3d2-candidate-painter-reconstruction-v1",
        ),
    ],
)
def test_verified_source_alpha_skips_only_boundary_relative_vetoes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
    schema: str,
) -> None:
    import app.transform_journal as transform_journal
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"<svg>parent</svg>")
    candidate.write_bytes(b"<svg>candidate</svg>")
    source = np.zeros((8, 8, 3), dtype=np.uint8)

    def measure(data: bytes, _source: np.ndarray, **_kwargs):
        return _metric(data, candidate=b"candidate" in data)

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)
    journal = TransformJournal(parent, source, image_class="clean_logo")
    accepted, stage = journal.consider_candidate(
        stage_id,
        parent,
        candidate,
        transform_report=_report(schema=schema),
    )
    assert accepted == candidate
    assert stage["status"] == "accepted"
    assert stage["reason_codes"] == ["verified_source_alpha_contract"]
    assert stage["source_alpha_contract_verified"] is True


def test_unverified_source_alpha_keeps_existing_journal_vetoes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transform_journal as transform_journal
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"<svg>parent</svg>")
    candidate.write_bytes(b"<svg>candidate</svg>")

    def measure(data: bytes, _source: np.ndarray, **_kwargs):
        return _metric(data, candidate=b"candidate" in data)

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)
    journal = TransformJournal(parent, np.zeros((8, 8, 3), dtype=np.uint8))
    accepted, stage = journal.consider_candidate(
        "source_alpha_vector_mask", parent, candidate, transform_report={}
    )
    assert accepted == parent
    assert stage["source_alpha_contract_verified"] is False
    assert {
        "topology_component_regression",
        "topology_hole_regression",
        "edge_f1_regression",
        "seam_regression",
    }.issubset(set(stage["reason_codes"]))


def test_verified_source_alpha_still_rejects_ssim_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transform_journal as transform_journal
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"<svg>parent</svg>")
    candidate.write_bytes(b"<svg>candidate</svg>")

    def measure(data: bytes, _source: np.ndarray, **_kwargs):
        candidate_data = b"candidate" in data
        return _metric(
            data,
            candidate=candidate_data,
            ssim=0.97 if candidate_data else 0.99,
        )

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)
    journal = TransformJournal(parent, np.zeros((8, 8, 3), dtype=np.uint8))
    accepted, stage = journal.consider_candidate(
        "source_alpha_vector_mask",
        parent,
        candidate,
        transform_report=_report(),
    )
    assert accepted == parent
    assert "ssim_regression" in stage["reason_codes"]
    assert stage["source_alpha_contract_verified"] is False


def test_verified_source_alpha_still_rejects_complexity_explosion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transform_journal as transform_journal
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"<svg>parent</svg>")
    candidate.write_bytes(b"<svg>candidate</svg>")

    def measure(data: bytes, _source: np.ndarray, **_kwargs):
        candidate_data = b"candidate" in data
        metric = _metric(data, candidate=candidate_data)
        if candidate_data:
            metric["path_count"] = 600
        return metric

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)
    journal = TransformJournal(parent, np.zeros((8, 8, 3), dtype=np.uint8))
    accepted, stage = journal.consider_candidate(
        "source_alpha_vector_mask",
        parent,
        candidate,
        transform_report=_report(),
    )
    assert accepted == parent
    assert "path_complexity_explosion" in stage["reason_codes"]


def test_identity_or_background_proof_cannot_be_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transform_journal as transform_journal
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"<svg>parent</svg>")
    candidate.write_bytes(b"<svg>candidate</svg>")

    def measure(data: bytes, _source: np.ndarray, **_kwargs):
        return _metric(data, candidate=b"candidate" in data)

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)
    report = _report()
    report["artwork_identity_preserved"] = False
    journal = TransformJournal(parent, np.zeros((8, 8, 3), dtype=np.uint8))
    accepted, stage = journal.consider_candidate(
        "source_alpha_vector_mask", parent, candidate, transform_report=report
    )
    assert accepted == parent
    assert stage["source_alpha_contract_verified"] is False
    assert "seam_regression" in stage["reason_codes"]
