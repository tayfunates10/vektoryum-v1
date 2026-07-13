"""Gerçek ASGI -> pipeline -> export -> evaluator -> download entegrasyonu."""
from __future__ import annotations

import hashlib
import sys
import xml.etree.ElementTree as ET
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
        lambda _session: {"email": "integration@example.com", "name": "IT", "role": "user"},
    )
    test_client = TestClient(main.app)
    test_client.cookies.set("session", "integration-session")
    return test_client


def _vectorize(client: TestClient, payload: bytes, mime: str, filename: str) -> dict:
    response = client.post(
        "/api/vectorize",
        files={"file": (filename, payload, mime)},
        data={"trace_mode": "auto", "shape_stacking": "stacked", "edge_cleanup": "on"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _xml_path_count(raw: bytes) -> int:
    root = ET.fromstring(raw)
    return sum(element.tag.rsplit("}", 1)[-1] == "path" for element in root.iter())


def _decision_signature(body: dict) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (
            str(stage["stage_id"]), str(stage["status"]),
            tuple(str(code) for code in stage.get("reason_codes", [])),
        )
        for stage in (body.get("transform_journal") or {}).get("stages", [])
    ]


def test_exact_sha_xml_metrics_and_real_download_body(client: TestClient) -> None:
    import app.main as main
    from exact_corpus import t1_topology

    fixture = t1_topology(256)
    body = _vectorize(client, fixture.input_bytes, fixture.input_mime, "t1.png")
    artifact = body["final_artifact"]
    journal = body["transform_journal"]
    response = client.get(body["download_links"]["svg"])
    assert response.status_code == 200, response.text
    downloaded = response.content
    actual_sha = hashlib.sha256(downloaded).hexdigest()

    assert body["quality_report"]["source"] == "final_artifact_evaluator"
    assert body["final_svg_sha256"] == actual_sha
    assert body["quality_report"]["final_svg_sha256"] == actual_sha
    assert artifact["final_svg_sha256"] == actual_sha
    assert artifact["quality_verdict"] == artifact["verdict"]
    assert artifact["quality_failure_codes"] == artifact["hard_fail_codes"]
    assert artifact["structural_safe"] is True
    assert artifact["artifacts"]["svg"]["sha256"] == actual_sha
    assert journal["chain_valid"] is True
    assert journal["final_accepted_sha256"] == actual_sha
    serializer_stage = next(
        stage for stage in journal["stages"]
        if stage["stage_id"] == "production_serializer"
    )
    assert body["refit_info"]["final_rescore"] == {
        "status": "measured",
        "fidelity_score": body["legacy_candidate_report"]["metrics"]["fidelity_score"],
        # Pipeline'ın kesin artifact'ı serializer'ın byte-parent'ıdır;
        # export sonrası exact hash yukarıda final evaluator ile doğrulanır.
        "svg_sha256": serializer_stage["parent_sha256"],
    }
    stage_ids = [stage["stage_id"] for stage in journal["stages"]]
    assert stage_ids.index("restore_source_dimensions") < stage_ids.index("component_align")
    assert artifact["exact_metrics"]["path_count"] == _xml_path_count(downloaded)
    assert b"<image" not in downloaded and b"data:image" not in downloaded
    assert set(body["download_links"]) == set(artifact["downloadable_formats"])
    assert not list((main.JOBS_ROOT / body["job_id"]).glob(".*.candidate.svg"))


def test_verdict_is_honest_and_legacy_score_is_isolated(client: TestClient) -> None:
    from exact_corpus import t1_topology

    body = _vectorize(client, t1_topology(256).input_bytes, "image/png", "t1.png")
    artifact = body["final_artifact"]
    assert artifact["verdict"] in {"production_ready", "needs_review", "failed"}
    if artifact["unmeasured_required"]:
        assert artifact["verdict"] != "production_ready"
    assert "legacy_candidate_report" in body
    assert body["quality_report"]["source"] == "final_artifact_evaluator"


def test_two_independent_runs_are_byte_and_decision_deterministic(client: TestClient) -> None:
    from exact_corpus import t1_topology

    fixture = t1_topology(192)
    first = _vectorize(client, fixture.input_bytes, fixture.input_mime, "same.png")
    second = _vectorize(client, fixture.input_bytes, fixture.input_mime, "same.png")
    assert first["job_id"] != second["job_id"]
    assert first["final_svg_sha256"] == second["final_svg_sha256"]
    assert _decision_signature(first) == _decision_signature(second)
    first_svg = client.get(first["download_links"]["svg"]).content
    second_svg = client.get(second["download_links"]["svg"]).content
    assert first_svg == second_svg
