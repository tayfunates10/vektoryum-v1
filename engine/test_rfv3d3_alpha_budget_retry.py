"""RFV-3D3 knockout değerlendirme bütçesi retry sözleşmesi."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import alpha_budget_retry as retry


class _FakeJournal:
    created: list["_FakeJournal"] = []

    def __init__(
        self,
        baseline_path: Path,
        source_rgb: np.ndarray,
        *,
        image_class: str = "clean_logo",
        required_metrics: set[str] | None = None,
        budget_seconds: float = 45.0,
        stage_timeout_seconds: float = 180.0,
        max_side: int = 512,
    ) -> None:
        self.baseline_path = Path(baseline_path)
        self.source_rgb = np.asarray(source_rgb)
        self.image_class = image_class
        self.required_metrics = set(required_metrics or ())
        self.budget_seconds = budget_seconds
        self.stage_timeout_seconds = stage_timeout_seconds
        self.max_side = max_side
        self.evaluation_seconds = 0.0
        self.stages: list[dict] = []
        type(self).created.append(self)


def setup_function() -> None:
    retry._PENDING.set({})
    _FakeJournal.created.clear()


def test_explicit_quantization_reduces_levels_with_existing_alpha_margin() -> None:
    yy, xx = np.mgrid[0:256, 0:256]
    radius = np.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
    values = np.clip((110.0 - radius) * 16.0, 0.0, 255.0).astype(np.uint8)
    quantized, opacity = retry._quantize_alpha_levels(values, 32)
    assert len(opacity) <= 31
    restored = np.rint(
        quantized.astype(np.float32) * 255.0 / 31.0
    ).astype(np.uint8)
    source = values.astype(np.float32) / 255.0
    candidate = restored.astype(np.float32) / 255.0
    alpha_iou = float(
        np.minimum(source, candidate).sum()
        / np.maximum(source, candidate).sum()
    )
    alpha_mae = float(np.abs(source - candidate).mean())
    assert alpha_iou >= 0.995
    assert alpha_mae <= 0.005


def test_budget_only_retries_with_same_per_candidate_limits() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "parent.svg"
        candidate = root / "candidate.svg"
        source = root / "source.png"
        parent.write_bytes(b"parent")
        candidate.write_bytes(b"exact")
        source.write_bytes(b"source")
        exact_report = {
            "status": "accepted",
            "mask_encoding": "candidate_geometry_knockout",
            "reconstruction_clip_encoding": "rect-transform",
        }
        retry._register(
            candidate,
            retry._PendingKnockout(source, "logo_color", exact_report),
        )
        journal = _FakeJournal(
            parent,
            np.zeros((2, 2, 3), dtype=np.uint8),
            budget_seconds=45.0,
            stage_timeout_seconds=180.0,
            max_side=512,
        )
        calls: list[_FakeJournal] = []

        def original(
            current,
            stage_id,
            parent_path,
            candidate_path,
            *,
            transform_report=None,
        ):
            del stage_id, transform_report
            calls.append(current)
            if len(calls) == 1:
                current.evaluation_seconds = 45.1
                return Path(parent_path), {
                    "status": "budget_exhausted",
                    "reason_codes": ["evaluation_budget_exhausted"],
                }
            current.evaluation_seconds = 12.5
            current.stages = [{"status": "accepted"}]
            return Path(candidate_path), {
                "status": "accepted",
                "reason_codes": ["metrics_non_regressing"],
            }

        def compact(
            path,
            _source,
            _mode,
            *,
            alpha_levels,
            preferred_clip_encoding,
        ):
            Path(path).write_bytes(f"compact-{alpha_levels}".encode())
            return {
                "status": "accepted",
                "mask_encoding": "candidate_geometry_knockout",
                "reconstruction_clip_encoding": preferred_clip_encoding,
                "reconstruction_rectangle_count": 9000,
                "after_byte_size": Path(path).stat().st_size,
            }

        wrapped = retry._wrap_consider_candidate(original)
        with patch.object(retry, "_compact_knockout_candidate", side_effect=compact):
            accepted, stage = wrapped(
                journal,
                "source_alpha_vector_mask",
                parent,
                candidate,
                transform_report=exact_report,
            )

        assert accepted == candidate
        assert stage["status"] == "accepted"
        assert len(calls) == 2
        assert calls[1].budget_seconds == calls[0].budget_seconds == 45.0
        assert calls[1].stage_timeout_seconds == calls[0].stage_timeout_seconds == 180.0
        assert calls[1].max_side == calls[0].max_side == 512
        assert journal.evaluation_seconds == 12.5
        assert candidate.read_bytes() == b"compact-32"
        assert exact_report["mask_fallback_trigger"] == "evaluation_budget_exhausted"
        assert exact_report["compact_retry_attempts"][0]["alpha_levels"] == 32


def test_non_budget_rejection_is_not_retried() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "parent.svg"
        candidate = root / "candidate.svg"
        source = root / "source.png"
        parent.write_bytes(b"parent")
        candidate.write_bytes(b"exact")
        source.write_bytes(b"source")
        report = {
            "status": "accepted",
            "mask_encoding": "candidate_geometry_knockout",
        }
        retry._register(
            candidate,
            retry._PendingKnockout(source, "logo_color", report),
        )
        journal = _FakeJournal(parent, np.zeros((1, 1, 3), dtype=np.uint8))

        def original(
            _journal,
            _stage,
            parent_path,
            _candidate,
            *,
            transform_report=None,
        ):
            del transform_report
            return Path(parent_path), {
                "status": "rolled_back",
                "reason_codes": ["ssim_regression"],
            }

        wrapped = retry._wrap_consider_candidate(original)
        with patch.object(retry, "_compact_knockout_candidate") as compact:
            accepted, stage = wrapped(
                journal,
                "source_alpha_vector_mask",
                parent,
                candidate,
                transform_report=report,
            )
        assert accepted == parent
        assert stage["reason_codes"] == ["ssim_regression"]
        compact.assert_not_called()


def test_failed_compact_attempt_restores_exact_candidate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / "parent.svg"
        candidate = root / "candidate.svg"
        source = root / "source.png"
        parent.write_bytes(b"parent")
        candidate.write_bytes(b"exact-candidate")
        source.write_bytes(b"source")
        report = {
            "status": "accepted",
            "mask_encoding": "candidate_geometry_knockout",
        }
        retry._register(
            candidate,
            retry._PendingKnockout(source, "logo_color", report),
        )
        journal = _FakeJournal(parent, np.zeros((1, 1, 3), dtype=np.uint8))

        def original(
            _journal,
            _stage,
            parent_path,
            _candidate,
            *,
            transform_report=None,
        ):
            del transform_report
            return Path(parent_path), {
                "status": "budget_exhausted",
                "reason_codes": ["evaluation_budget_exhausted"],
            }

        wrapped = retry._wrap_consider_candidate(original)
        with patch.object(
            retry,
            "_compact_knockout_candidate",
            side_effect=RuntimeError(
                "source_alpha_candidate_knockout_iou_gate_failed"
            ),
        ):
            accepted, stage = wrapped(
                journal,
                "source_alpha_vector_mask",
                parent,
                candidate,
                transform_report=report,
            )
        assert accepted == parent
        assert stage["status"] == "budget_exhausted"
        assert candidate.read_bytes() == b"exact-candidate"
        assert report["compact_retry_attempts"][0]["status"] == "candidate_rejected"
