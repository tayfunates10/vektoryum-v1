"""T2/T3/T5 zor korpusunun gerçek production zinciri regresyonları."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ENGINE_DIR / "regression"))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import app.main as main

    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(main, "FEEDBACK_FILE", tmp_path / "data" / "feedback.jsonl")
    monkeypatch.setattr(main, "USERS_FILE", tmp_path / "data" / "users.json")
    monkeypatch.setattr(
        main, "_require_user",
        lambda _session: {"email": "corpus@example.com", "name": "Corpus", "role": "user"},
    )
    result = TestClient(main.app)
    result.cookies.set("session", "corpus-session")
    return result


def _run(client: TestClient, fixture, filename: str) -> dict:
    response = client.post(
        "/api/vectorize",
        files={"file": (filename, fixture.input_bytes, fixture.input_mime)},
        data={"trace_mode": "auto", "shape_stacking": "stacked", "edge_cleanup": "on"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    journal = body["transform_journal"]
    assert journal["chain_valid"] is True
    assert journal["final_accepted_sha256"] == body["final_svg_sha256"]
    assert all(stage["status"] in {
        "accepted", "rolled_back", "no_op", "failed", "budget_exhausted",
    } for stage in journal["stages"])
    if "svg" in body["download_links"]:
        download = client.get(body["download_links"]["svg"])
        assert download.status_code == 200
        assert hashlib.sha256(download.content).hexdigest() == body["final_svg_sha256"]
    return body


def _assert_no_accepted_post_transform_explosion(body: dict) -> None:
    for stage in body["transform_journal"]["stages"]:
        if stage["status"] != "accepted":
            continue
        before, after = stage.get("before_metrics"), stage.get("after_metrics")
        if not before or not after:
            continue
        before_paths = max(1, int(before.get("path_count") or 0))
        before_nodes = max(1, int(before.get("node_count") or 0))
        before_bytes = max(1, int(before.get("byte_size") or 0))
        assert int(after.get("path_count") or 0) <= max(before_paths * 4, before_paths + 500)
        assert int(after.get("node_count") or 0) <= max(before_nodes * 4, before_nodes + 2500)
        assert int(after.get("byte_size") or 0) <= max(before_bytes * 3, before_bytes + 250_000)


def test_t2_alpha_gradient_is_never_false_production_ready(client: TestClient) -> None:
    from exact_corpus import t2_gradient_alpha

    body = _run(client, t2_gradient_alpha(192), "t2.png")
    artifact = body["final_artifact"]
    assert body["analysis"]["has_gradient"] is True
    assert artifact["verdict"] != "production_ready"
    alpha = artifact["metrics"]["G_gradient_alpha"]
    assert alpha["source_has_alpha"] is True
    assert alpha["alpha_fidelity_status"] in {"passed", "failed", "measured", "unmeasured"}
    if alpha["alpha_fidelity_status"] == "unmeasured":
        assert "alpha_fidelity" in artifact["unmeasured_required"]
    else:
        assert "alpha_iou" in alpha and "alpha_mae" in alpha
    # Gradient field modeling is still a separate required metric; no false green.
    assert "gradient_fidelity" in artifact["unmeasured_required"]
    _assert_no_accepted_post_transform_explosion(body)


def test_t3_micro_detail_pipeline_keeps_journal_invariants(client: TestClient) -> None:
    from exact_corpus import t3_micro_detail

    body = _run(client, t3_micro_detail(192), "t3.png")
    for stage in body["transform_journal"]["stages"]:
        if stage["status"] != "accepted":
            continue
        before, after = stage.get("before_metrics"), stage.get("after_metrics")
        if not before or not after:
            continue
        assert after.get("component_delta", 0) <= before.get("component_delta", 0)
        assert after.get("hole_delta", 0) <= before.get("hole_delta", 0)
    _assert_no_accepted_post_transform_explosion(body)


def test_t5_pipeline_receives_q32_and_blocks_post_transform_explosion(client: TestClient) -> None:
    from exact_corpus import t5_lowres_jpeg

    fixture = t5_lowres_jpeg(128)
    assert fixture.input_bytes[:2] == b"\xff\xd8"
    body = _run(client, fixture, "t5.jpg")
    assert fixture.oracle["class"] == "low_res_logo"
    assert body["final_artifact"]["verdict"] in {
        "production_ready", "needs_review", "failed",
    }
    _assert_no_accepted_post_transform_explosion(body)
