from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import painter_measurement_reuse as reuse


def setup_function() -> None:
    reuse._clear_scope()


def _journal(source: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        source_rgb=np.zeros((4, 4, 3), dtype=np.uint8) if source is None else source,
        image_class="clean_logo",
        required_metrics=set(),
        max_side=512,
        evaluation_seconds=0.0,
        _measurement_stage_id="source_alpha_vector_mask",
        _cache={},
        _cache_namespace="",
    )


def _proof_report() -> dict:
    comparisons = {
        background: {
            metric: {"non_regressing": True}
            for metric in ("ssim", "ms_ssim", "rgb_mae", "rgb_p95")
        }
        for background in ("white", "black", "checker")
    }
    return {
        "schema": "rfv3d2-candidate-painter-reconstruction-v1",
        "status": "accepted",
        "applied": True,
        "candidate_geometry_preserved": True,
        "candidate_path_data_preserved": True,
        "candidate_identity_preserved": True,
        "artwork_identity_preserved": True,
        "trace_rgb_bytes_preserved": True,
        "source_alpha_background_non_regression": True,
        "source_alpha_background_failure_codes": [],
        "source_alpha_background_comparison": comparisons,
        "source_alpha_background_authority": (
            "final_artifact_evaluator_parent_candidate_exact_non_regression"
        ),
        "source_alpha_paint_support_verified": True,
        "source_alpha_paint_support_authority": (
            "exact_component_overlap_with_non_canvas_rendered_paint"
        ),
        "source_alpha_component_count": 1,
        "source_alpha_unsupported_component_count": 0,
        "source_alpha_non_canvas_renderable_count": 1,
        "final_evaluator_alpha_plane_status": "passed",
        "final_evaluator_alpha_plane_hard_fail_codes": [],
        "source_truth_alpha_iou": 0.999,
        "source_truth_alpha_mae": 0.0001,
        "final_evaluator_alpha_iou": 0.999,
        "final_evaluator_alpha_mae": 0.0001,
    }


def _measured_cache(parent: Path, candidate: Path, source: np.ndarray) -> dict:
    namespace = reuse._namespace(source, 512, set())
    cache = {}
    for path, seconds in ((parent, 3.0), (candidate, 4.0)):
        sha = reuse._sha_path(path)
        key = reuse._metric_key(namespace, sha)
        cache[key] = {"sha256": sha}
        cache[f"{key}\x00seconds"] = seconds
    return cache


def test_verified_cache_hit_has_zero_increment_but_miss_is_charged() -> None:
    journal = _journal()
    journal._cache_namespace = reuse._namespace(journal.source_rgb, 512, set())
    data = b"<svg/>"
    key = reuse._journal_key(journal, data)
    journal._vektoryum_reuse_bound = True
    journal._cache[key] = {"sha256": reuse._sha_bytes(data)}
    journal._cache[f"{key}\x00seconds"] = 12.5

    def original(current, payload):
        cache_key = reuse._journal_key(current, payload)
        if cache_key in current._cache:
            current.evaluation_seconds += current._cache[f"{cache_key}\x00seconds"]
            return current._cache[cache_key]
        current.evaluation_seconds += 7.0
        return {"sha256": reuse._sha_bytes(payload)}

    measured = reuse._wrap_measure(original)
    assert measured(journal, data)["sha256"] == reuse._sha_bytes(data)
    assert journal.evaluation_seconds == 0.0
    assert journal._vektoryum_reuse_hits == 1

    measured(journal, b"<svg><path/></svg>")
    assert journal.evaluation_seconds == 7.0

    unbound = _journal()
    unbound._cache_namespace = reuse._namespace(unbound.source_rgb, 512, set())
    unbound._cache[key] = {"sha256": reuse._sha_bytes(data)}
    unbound._cache[f"{key}\x00seconds"] = 12.5
    measured(unbound, data)
    assert unbound.evaluation_seconds == 12.5


def test_exact_identity_and_proof_match_binds_cache_once(tmp_path: Path) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    cache = _measured_cache(parent, candidate, source)
    report = _proof_report()
    reuse._register(
        parent, candidate, source, "clean_logo", set(), 512, cache, report
    )

    journal = _journal(source.copy())
    seen = {}

    def original(current, stage_id, parent_path, candidate_path, *, transform_report=None):
        seen["cache"] = current._cache
        seen["namespace"] = current._cache_namespace
        current._vektoryum_reuse_hits = 2
        return Path(candidate_path), {"status": "accepted"}

    wrapped = reuse._wrap_consider(original)
    accepted, stage = wrapped(
        journal,
        "source_alpha_vector_mask",
        parent,
        candidate,
        transform_report=report,
    )
    assert accepted == candidate
    assert stage["status"] == "accepted"
    assert seen["cache"] is cache
    assert seen["namespace"] == reuse._namespace(source, 512, set())
    assert journal._vektoryum_reuse_bound is True
    assert stage["measurement_cache_reuse"] == "verified_request_scoped_exact_sha"
    assert stage["measurement_cache_hit_count"] == 2
    assert stage["measurement_cache_original_seconds"] == 7.0
    assert report["journal_measurement_cache_hit_count"] == 2

    second = _journal(source.copy())
    wrapped(second, "source_alpha_vector_mask", parent, candidate, transform_report=report)
    assert second._cache == {}
    assert not hasattr(second, "_vektoryum_reuse_bound")


@pytest.mark.parametrize(
    "mismatch",
    ["candidate", "parent", "source", "class", "metrics", "side", "proof"],
)
def test_identity_or_proof_mismatch_never_binds_cache(
    tmp_path: Path,
    mismatch: str,
) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    cache = _measured_cache(parent, candidate, source)
    registered_report = _proof_report()
    reuse._register(
        parent, candidate, source, "clean_logo", set(), 512, cache,
        registered_report,
    )

    journal = _journal(source.copy())
    outer_report = _proof_report()
    if mismatch == "candidate":
        candidate.write_bytes(b"changed")
    elif mismatch == "parent":
        parent.write_bytes(b"changed-parent")
    elif mismatch == "source":
        journal.source_rgb[0, 0, 0] = 1
    elif mismatch == "class":
        journal.image_class = "geometric"
    elif mismatch == "metrics":
        journal.required_metrics = {"alpha_fidelity"}
    elif mismatch == "side":
        journal.max_side = 256
    elif mismatch == "proof":
        outer_report["source_truth_alpha_iou"] = 0.998

    def original(current, _stage, parent_path, _candidate_path, *, transform_report=None):
        return Path(parent_path), {"status": "rolled_back"}

    wrapped = reuse._wrap_consider(original)
    wrapped(
        journal,
        "source_alpha_vector_mask",
        parent,
        candidate,
        transform_report=outer_report,
    )
    assert journal._cache == {}
    assert not hasattr(journal, "_vektoryum_reuse_bound")


def test_only_passing_painter_journal_with_complete_measurements_registers(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    report = _proof_report()

    def pass_and_measure(
        parent_path,
        candidate_path,
        journal_source_rgb,
        image_class,
        transform_report,
        measurement_cache=None,
    ):
        measurement_cache.update(
            _measured_cache(parent_path, candidate_path, journal_source_rgb)
        )
        return True, []

    passing = reuse._wrap_painter_journal(pass_and_measure)
    cache = {}
    passed, reasons = passing(
        parent, candidate, source, "clean_logo", report, cache
    )
    assert passed is True and reasons == []
    proof = reuse._take(
        _journal(source), parent, candidate, _proof_report()
    )
    assert proof is not None
    assert proof.cache is cache
    assert proof.measured_seconds == 7.0

    reuse._clear_scope()
    failing = reuse._wrap_painter_journal(
        lambda *args, **kwargs: (False, ["seam_regression"])
    )
    failed_cache = {}
    passed, reasons = failing(
        parent, candidate, source, "clean_logo", report, failed_cache
    )
    assert passed is False and reasons == ["seam_regression"]
    assert reuse._take(
        _journal(source), parent, candidate, _proof_report()
    ) is None

    reuse._clear_scope()
    incomplete = reuse._wrap_painter_journal(
        lambda *args, **kwargs: (True, [])
    )
    incomplete(parent, candidate, source, "clean_logo", report, {})
    assert reuse._take(
        _journal(source), parent, candidate, _proof_report()
    ) is None


def test_painter_exception_clears_request_scope(tmp_path: Path) -> None:
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(b"parent")
    candidate.write_bytes(b"candidate")
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    cache = _measured_cache(parent, candidate, source)
    reuse._register(
        parent, candidate, source, "clean_logo", set(), 512, cache,
        _proof_report(),
    )

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    wrapped = reuse._wrap_painter_apply(explode)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped()
    assert reuse._REGISTRY.get() == {}
