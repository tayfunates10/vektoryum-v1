"""FAZ 2 — transform journal accept/rollback/complexity regresyonları."""
from __future__ import annotations

import hashlib
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ENGINE_DIR / "regression"))


def _svg(body: str, n: int = 128) -> bytes:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}">'
        f'<rect width="{n}" height="{n}" fill="#fff"/>{body}</svg>'
    ).encode()


def _square_source(n: int = 128) -> np.ndarray:
    src = np.full((n, n, 3), 255, np.uint8)
    src[24:104, 24:104] = (227, 0, 11)
    return src


def _square_svg(extra: str = "") -> bytes:
    return _svg(f'<rect x="24" y="24" width="80" height="80" fill="#e3000b"/>{extra}')


def _donut_source(n: int = 128) -> np.ndarray:
    src = _square_source(n)
    src[50:78, 50:78] = 255
    return src


def _donut_svg() -> bytes:
    return _svg(
        '<path fill="#e3000b" fill-rule="evenodd" '
        'd="M24 24 H104 V104 H24 Z M50 50 H78 V78 H50 Z"/>'
    )


def test_safe_metadata_only_candidate_is_accepted(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>deterministic</metadata>"))
    journal = TransformJournal(parent, _square_source())
    accepted, stage = journal.consider_candidate("metadata", parent, candidate)
    assert accepted == candidate
    assert stage["status"] == "accepted"
    assert journal.to_dict()["final_accepted_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()


def test_hole_loss_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    parent = tmp_path / "donut.svg"
    candidate = tmp_path / "filled.svg"
    parent.write_bytes(_donut_svg())
    candidate.write_bytes(_square_svg())
    journal = TransformJournal(parent, _donut_source())
    accepted, stage = journal.consider_candidate("hole_breaker", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert set(stage["reason_codes"]) & {
        "topology_hole_regression", "ssim_regression", "edge_f1_regression"
    }


def test_path_explosion_is_rolled_back_even_when_render_is_same(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "exploded.svg"
    parent.write_bytes(_square_svg())
    invisible = "".join('<path d="M0 0" fill="none"/>' for _ in range(600))
    candidate.write_bytes(_square_svg(invisible))
    journal = TransformJournal(parent, _square_source())
    accepted, stage = journal.consider_candidate("path_explosion", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "path_complexity_explosion" in stage["reason_codes"]


def test_gradient_definition_loss_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    parent = tmp_path / "gradient.svg"
    candidate = tmp_path / "flattened.svg"
    parent.write_bytes(_svg(
        '<defs><linearGradient id="g"><stop offset="0" stop-color="#e3000b"/>'
        '<stop offset="1" stop-color="#554bad"/></linearGradient></defs>'
        '<rect x="24" y="24" width="80" height="80" fill="url(#g)"/>'
    ))
    candidate.write_bytes(_square_svg())
    journal = TransformJournal(parent, _square_source())
    accepted, stage = journal.consider_candidate("flatten_gradient", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "gradient_definition_loss" in stage["reason_codes"]


def test_real_geometry_cleanup_is_measured_and_accepted(tmp_path: Path) -> None:
    from app.geometry_cleanup import cleanup_svg_geometry
    from app.transform_journal import TransformJournal

    path = tmp_path / "redundant.svg"
    path.write_bytes(_svg(
        '<path fill="#e3000b" d="M24 24 L50 24 L78 24 L104 24 '
        'L104 64 L104 104 L64 104 L24 104 L24 64 Z"/>'
    ))
    journal = TransformJournal(path, _square_source())
    accepted, _report, stage = journal.run_in_place(
        "geometry_cleanup", path,
        lambda candidate: cleanup_svg_geometry(candidate, aggressiveness="standard"),
    )
    assert accepted
    assert stage["status"] in {"accepted", "no_op"}
    assert stage["accepted_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_t1_junction_blob_regression_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal
    from exact_corpus import t1_topology

    fixture = t1_topology(192)
    parent = tmp_path / "t1.svg"
    candidate = tmp_path / "t1_blob.svg"
    parent.write_bytes(fixture.oracle_svg)
    root = ET.fromstring(fixture.oracle_svg)
    ET.SubElement(root, "{http://www.w3.org/2000/svg}circle", {
        "cx": "96", "cy": "115", "r": "18", "fill": "#0a0a0a",
    })
    candidate.write_bytes(ET.tostring(root, encoding="utf-8"))
    journal = TransformJournal(parent, fixture.rgb, image_class="geometric")
    accepted, stage = journal.consider_candidate("junction_blob", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert set(stage["reason_codes"]) & {
        "topology_component_regression", "ssim_regression",
        "edge_f1_regression", "seam_regression",
    }


def test_t3_thin_ring_and_micro_component_loss_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal
    from exact_corpus import t3_micro_detail

    fixture = t3_micro_detail(192)
    parent = tmp_path / "t3.svg"
    candidate = tmp_path / "t3_missing_ring.svg"
    parent.write_bytes(fixture.oracle_svg)
    root = ET.fromstring(fixture.oracle_svg)
    for child in list(root):
        if child.tag.rsplit("}", 1)[-1] == "circle" and child.get("stroke-width") == "1":
            root.remove(child)
            break
    candidate.write_bytes(ET.tostring(root, encoding="utf-8"))
    journal = TransformJournal(parent, fixture.rgb, image_class="lineart")
    accepted, stage = journal.consider_candidate("micro_detail_loss", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert set(stage["reason_codes"]) & {
        "topology_component_regression", "ssim_regression", "edge_f1_regression",
    }


def test_coarse_topology_false_positive_is_refined_at_1024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coarse AA gürültüsü hard-veto olmaz; fine tur yine kanonik karardır."""
    import app.final_artifact_evaluator as evaluator
    from app.transform_journal import _measure_svg_bytes

    source = np.full((2048, 2048, 3), 255, np.uint8)
    source[384:1664, 384:1664] = (227, 0, 11)
    svg = _svg(
        '<rect x="24" y="24" width="80" height="80" fill="#e3000b"/>',
        n=128,
    )
    calls_by_side: dict[int, int] = {}

    def fake_topology(labels: np.ndarray, _colors: int, _min_area: int) -> dict[str, int]:
        side = max(labels.shape)
        call = calls_by_side.get(side, 0)
        calls_by_side[side] = call + 1
        if side <= 512:
            # source/render arasında yalnız coarse ölçümde sahte dört bileşen
            return {"components": 8 if call % 2 else 4, "holes": 0}
        return {"components": 4, "holes": 0}

    monkeypatch.setattr(evaluator, "_topology_signature", fake_topology)
    measured = _measure_svg_bytes(svg, source, max_side=512)
    assert measured["topology_refinement"] == {
        "coarse_max_side": 512,
        "refined_max_side": 1024,
        "coarse_component_delta": 4,
        "coarse_hole_delta": 0,
        "refined_component_delta": 0,
        "refined_hole_delta": 0,
    }
    assert measured["component_delta"] == 0
    assert measured["hole_delta"] == 0


def test_t5_q32_post_transform_complexity_explosion_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal
    from exact_corpus import t5_lowres_jpeg

    fixture = t5_lowres_jpeg(128)
    parent = tmp_path / "t5.svg"
    candidate = tmp_path / "t5_exploded.svg"
    parent.write_bytes(fixture.oracle_svg)
    root = ET.fromstring(fixture.oracle_svg)
    for index in range(600):
        ET.SubElement(root, "{http://www.w3.org/2000/svg}path", {
            "d": f"M{index % 128} {index % 128}", "fill": "none",
        })
    candidate.write_bytes(ET.tostring(root, encoding="utf-8"))
    journal = TransformJournal(parent, fixture.rgb, image_class="photo")
    accepted, stage = journal.consider_candidate("jpeg_path_explosion", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "path_complexity_explosion" in stage["reason_codes"]


def test_nonfinite_geometry_is_rolled_back(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "nan.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_svg('<path d="M0 0 LNaN 4"/>'))
    journal = TransformJournal(parent, _square_source())
    accepted, stage = journal.consider_candidate("nan", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "nonfinite_geometry" in stage["reason_codes"]


def test_source_dimension_restore_adds_viewbox_before_alignment(tmp_path: Path) -> None:
    from app.pipeline import _restore_source_dimensions
    from app.transform_journal import TransformJournal

    path = tmp_path / "no_viewbox.svg"
    path.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        b'<path fill="#e3000b" d="M12 12H52V52H12Z"/></svg>'
    )
    journal = TransformJournal(path, _square_source())
    accepted, _report, stage = journal.run_in_place(
        "restore_source_dimensions", path,
        lambda candidate: _restore_source_dimensions(
            candidate, {"width": 128, "height": 128},
        ),
    )
    assert accepted
    assert stage["status"] == "accepted"
    root = ET.fromstring(path.read_bytes())
    assert root.get("viewBox") == "0 0 128 128"
    assert root.get("width") == "128" and root.get("height") == "128"


def test_in_place_malformed_transform_restores_exact_parent_bytes(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    path = tmp_path / "artifact.svg"
    original = _square_svg()
    path.write_bytes(original)
    journal = TransformJournal(path, _square_source())

    def break_xml(target: Path):
        target.write_bytes(b"<svg><path>")
        return {"changed": True}

    accepted, _report, stage = journal.run_in_place("malformed", path, break_xml)
    assert not accepted
    assert stage["status"] == "rolled_back"
    assert "xml_parse_failed" in stage["reason_codes"]
    assert path.read_bytes() == original


def test_in_place_exception_restores_parent(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    path = tmp_path / "artifact.svg"
    original = _square_svg()
    path.write_bytes(original)
    journal = TransformJournal(path, _square_source())

    def explode(target: Path):
        target.write_bytes(b"broken")
        raise RuntimeError("boom")

    accepted, report, stage = journal.run_in_place("exception", path, explode)
    assert not accepted and report is None
    assert stage["status"] == "failed"
    assert "transform_exception" in stage["reason_codes"]
    assert path.read_bytes() == original


def test_mutator_never_receives_or_touches_accepted_path(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    path = tmp_path / "artifact.svg"
    original = _square_svg()
    changed = _square_svg("<metadata>candidate</metadata>")
    path.write_bytes(original)
    journal = TransformJournal(path, _square_source())

    def mutate_copy(target: Path):
        assert target != path
        assert path.read_bytes() == original
        target.write_bytes(changed)
        assert path.read_bytes() == original
        return {"changed": True}

    accepted, _report, stage = journal.run_in_place("immutable_parent", path, mutate_copy)
    assert accepted
    assert stage["status"] == "accepted"
    assert path.read_bytes() == changed


def test_budget_exceeded_during_transform_is_fail_closed(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    path = tmp_path / "artifact.svg"
    original = _square_svg()
    path.write_bytes(original)
    journal = TransformJournal(
        path, _square_source(), budget_seconds=5.0, stage_timeout_seconds=0.005,
    )

    def slow_change(target: Path):
        target.write_bytes(_square_svg("<metadata>late</metadata>"))
        time.sleep(0.01)
        return {"changed": True}

    accepted, _report, stage = journal.run_in_place("late", path, slow_change)
    assert not accepted
    assert stage["status"] == "budget_exhausted"
    assert stage["reason_codes"] == ["transform_stage_timeout"]
    assert path.read_bytes() == original


def test_noop_and_budget_exhaustion_do_not_mutate(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal

    path = tmp_path / "artifact.svg"
    original = _square_svg()
    path.write_bytes(original)
    journal = TransformJournal(path, _square_source())
    accepted, _report, stage = journal.run_in_place("noop", path, lambda _p: {"changed": False})
    assert accepted and stage["status"] == "no_op"

    exhausted = TransformJournal(path, _square_source(), budget_seconds=0.0)
    called = False

    def must_not_run(_target: Path):
        nonlocal called
        called = True

    accepted, _report, stage = exhausted.run_in_place("budget", path, must_not_run)
    assert not accepted and not called
    assert stage["status"] == "budget_exhausted"
    assert path.read_bytes() == original


def test_unmeasured_alpha_or_gradient_forces_changed_stage_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.source_truth as source_truth
    from app.transform_journal import TransformJournal

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", lambda *_args, **_kwargs: None)
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>changed</metadata>"))
    journal = TransformJournal(
        parent, _square_source(), required_metrics={"alpha_fidelity"}
    )
    accepted, stage = journal.consider_candidate("alpha_unknown", parent, candidate)
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]


def test_alpha_preserving_viewbox_restore_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    import app.source_truth as source_truth
    from app.pipeline import _restore_source_dimensions
    from app.transform_journal import TransformJournal

    source = _square_source()
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())

    def stable_alpha(_path: Path, width: int, height: int) -> np.ndarray:
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[24:104, 24:104, :3] = (227, 0, 11)
        rgba[24:104, 24:104, 3] = 255
        return rgba

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", stable_alpha)
    path = tmp_path / "transparent-no-viewbox.svg"
    path.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128">'
        b'<path fill="#e3000b" d="M24 24H104V104H24Z"/></svg>'
    )
    journal = TransformJournal(
        path, source, required_metrics={"alpha_fidelity"},
    )
    accepted, _report, stage = journal.run_in_place(
        "restore_source_dimensions",
        path,
        lambda candidate: _restore_source_dimensions(
            candidate, {"width": 128, "height": 128},
        ),
    )

    assert accepted
    assert stage["status"] == "accepted"
    assert stage["required_unmeasured"] == []
    assert stage["alpha_comparison"]["alpha_iou"] == pytest.approx(1.0)
    assert stage["alpha_comparison"]["alpha_mae"] == pytest.approx(0.0)
    assert "_render_alpha" not in stage["before_metrics"]
    assert "_render_alpha" not in stage["after_metrics"]
    assert ET.fromstring(path.read_bytes()).get("viewBox") == "0 0 128 128"


def test_alpha_plane_regression_is_rolled_back_even_when_rgb_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    import app.source_truth as source_truth
    from app.transform_journal import TransformJournal

    source = _square_source()
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())

    def rendered_alpha(path: Path, width: int, height: int) -> np.ndarray:
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        loss = b"alpha-loss" in Path(path).read_bytes()
        lo, hi = (36, 92) if loss else (24, 104)
        rgba[lo:hi, lo:hi, :3] = (227, 0, 11)
        rgba[lo:hi, lo:hi, 3] = 255
        return rgba

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", rendered_alpha)
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>alpha-loss</metadata>"))
    journal = TransformJournal(
        parent, source, required_metrics={"alpha_fidelity"},
    )
    accepted, stage = journal.consider_candidate(
        "restore_source_dimensions", parent, candidate,
    )

    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert set(stage["reason_codes"]) & {"alpha_iou_regression", "alpha_mae_regression"}
    assert stage["required_unmeasured"] == []
    assert stage["alpha_comparison"]["alpha_iou"] < 0.995


def test_alpha_measurement_is_scoped_to_source_dimension_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    import app.source_truth as source_truth
    from app.transform_journal import TransformJournal

    source = _square_source()
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())

    def stable_alpha(_path: Path, width: int, height: int) -> np.ndarray:
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[24:104, 24:104, :3] = (227, 0, 11)
        rgba[24:104, 24:104, 3] = 255
        return rgba

    monkeypatch.setattr(source_truth, "render_svg_to_rgba", stable_alpha)
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>downstream-change</metadata>"))
    journal = TransformJournal(parent, source, required_metrics={"alpha_fidelity"})
    accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)

    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]
    assert stage["required_unmeasured"] == ["alpha_fidelity"]
    assert stage["alpha_comparison"] is None


def test_alpha_restore_reuses_single_rgba_render_per_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    import app.source_truth as source_truth
    from app.transform_journal import TransformJournal

    source = _square_source()
    calls = {"rgb": 0, "rgba": 0}

    def rgb_render(_path: Path, _width: int, _height: int) -> np.ndarray:
        calls["rgb"] += 1
        return source.copy()

    def rgba_render(_path: Path, width: int, height: int) -> np.ndarray:
        calls["rgba"] += 1
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[24:104, 24:104, :3] = (227, 0, 11)
        rgba[24:104, 24:104, 3] = 255
        return rgba

    monkeypatch.setattr(fidelity, "render_svg_to_rgb", rgb_render)
    monkeypatch.setattr(source_truth, "render_svg_to_rgba", rgba_render)
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>same-alpha</metadata>"))
    journal = TransformJournal(
        parent, source, required_metrics={"alpha_fidelity"},
    )

    accepted, stage = journal.consider_candidate(
        "restore_source_dimensions", parent, candidate,
    )

    assert accepted == candidate
    assert stage["status"] == "accepted"
    assert stage["required_unmeasured"] == []
    assert stage["alpha_comparison"]["alpha_iou"] == pytest.approx(1.0)
    assert calls == {"rgb": 0, "rgba": 2}

def test_gradient_preserving_restore_and_render_equivalent_topology_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    import app.transform_journal as transform_journal
    from app.pipeline import _restore_source_dimensions
    from app.transform_journal import TransformJournal

    source = _square_source()
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())
    path = tmp_path / "gradient-no-viewbox.svg"
    path.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128">'
        b'<defs><linearGradient id="g"><stop offset="0" stop-color="#e3000b"/>'
        b'<stop offset="1" stop-color="#554bad"/></linearGradient></defs>'
        b'<rect x="24" y="24" width="80" height="80" fill="url(#g)"/></svg>'
    )
    journal = TransformJournal(path, source, required_metrics={"gradient_fidelity"})
    accepted, _report, stage = journal.run_in_place(
        "restore_source_dimensions", path,
        lambda candidate: _restore_source_dimensions(
            candidate, {"width": 128, "height": 128},
        ),
    )
    assert accepted
    assert stage["status"] == "accepted"
    assert stage["required_unmeasured"] == []
    assert stage["render_comparison"]["ssim"] == pytest.approx(1.0)
    assert "_render_rgb" not in stage["before_metrics"]
    assert "_render_rgb" not in stage["after_metrics"]
    assert ET.fromstring(path.read_bytes()).get("viewBox") == "0 0 128 128"

    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>candidate</metadata>"))

    def synthetic_measure(data: bytes, _source: np.ndarray, **_kwargs):
        changed = b"candidate" in data
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
            "structural_safe": True,
            "structural_failure_codes": [],
            "path_count": 1,
            "node_count": 5,
            "gradient_definition_count": 0,
            "required_unmeasured": [],
            "ssim": 0.99,
            "edge_f1_1px": 0.995,
            "seam_ratio": 0.0,
            "component_delta": 11 if changed else 10,
            "hole_delta": 0,
            "_render_rgb": source.copy(),
        }

    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", synthetic_measure)
    restore_journal = TransformJournal(parent, source)
    restored, restore_stage = restore_journal.consider_candidate(
        "restore_source_dimensions", parent, candidate,
    )
    assert restored == candidate
    assert restore_stage["status"] == "accepted"
    assert restore_stage["render_comparison"]["ssim"] == pytest.approx(1.0)

    downstream_journal = TransformJournal(parent, source)
    downstream, downstream_stage = downstream_journal.consider_candidate(
        "boundary_refit", parent, candidate,
    )
    assert downstream == parent
    assert "topology_component_regression" in downstream_stage["reason_codes"]


def test_gradient_render_regression_and_missing_render_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    from app.transform_journal import TransformJournal

    source = _square_source()

    def gradient_render(path: Path, _width: int, _height: int) -> np.ndarray:
        rendered = source.copy()
        if b"gradient-loss" in Path(path).read_bytes():
            rendered[:, 64:] = (0, 0, 0)
        return rendered

    monkeypatch.setattr(fidelity, "render_svg_to_rgb", gradient_render)
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>gradient-loss</metadata>"))
    journal = TransformJournal(
        parent, source, required_metrics={"gradient_fidelity"},
    )
    accepted, stage = journal.consider_candidate(
        "restore_source_dimensions", parent, candidate,
    )
    assert accepted == parent
    assert set(stage["reason_codes"]) & {
        "gradient_ssim_regression", "gradient_edge_regression",
        "gradient_rgb_mae_regression",
    }

    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: None)
    missing_journal = TransformJournal(
        parent, source, required_metrics={"gradient_fidelity"},
    )
    missing, missing_stage = missing_journal.consider_candidate(
        "restore_source_dimensions", parent, candidate,
    )
    assert missing == parent
    assert "required_metric_unmeasured" in missing_stage["reason_codes"]
    assert "gradient_stage_metrics_incomplete" in missing_stage["reason_codes"]


def test_gradient_measurement_is_scoped_to_source_dimension_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.fidelity as fidelity
    from app.transform_journal import TransformJournal

    source = _square_source()
    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())
    parent = tmp_path / "parent.svg"
    candidate = tmp_path / "candidate.svg"
    parent.write_bytes(_square_svg())
    candidate.write_bytes(_square_svg("<metadata>downstream-gradient</metadata>"))
    journal = TransformJournal(
        parent, source, required_metrics={"gradient_fidelity"},
    )
    accepted, stage = journal.consider_candidate(
        "boundary_refit", parent, candidate,
    )
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert stage["required_unmeasured"] == ["gradient_fidelity"]
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "gradient_stage_metrics_incomplete" in stage["reason_codes"]
    assert stage["render_comparison"] is None

def test_assertions_are_real() -> None:
    """Bu dosya global FAILS listesine değil gerçek pytest assertion'a dayanır."""
    with pytest.raises(AssertionError):
        assert False, "mutation proof"


def test_merge_detects_broken_sha_chain() -> None:
    from app.transform_journal import merge_journal_reports

    report = merge_journal_reports(
        {
            "schema_version": 1,
            "baseline_sha256": "a",
            "final_accepted_sha256": "b",
            "stages": [{"parent_sha256": "x", "accepted_sha256": "b"}],
            "budget": {},
        },
    )
    assert report is not None
    assert not report["chain_valid"]
    assert "stage_parent_hash_mismatch" in report["chain_failure_codes"]
