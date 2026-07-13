"""FAZ 1.1 — ham-byte, exact XML ve server-side download sözleşmeleri."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def _svg_bytes(*, newline: bytes = b"\n", bom: bool = False) -> bytes:
    lines = [
        b'<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'.encode(),
        '  <metadata>Türkçe ölçüm</metadata>'.encode("utf-8"),
        b'  <rect width="16" height="16" fill="#ffffff"/>',
        b'</svg>',
    ]
    data = newline.join(lines)
    return (b"\xef\xbb\xbf" + data) if bom else data


@pytest.mark.parametrize(
    "raw",
    [_svg_bytes(), _svg_bytes(newline=b"\r\n"), _svg_bytes(newline=b"\r\n", bom=True)],
    ids=["lf", "crlf", "crlf-bom-nonascii"],
)
def test_sha_is_exact_raw_bytes(raw: bytes) -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    src = np.full((16, 16, 3), 255, np.uint8)
    rep = evaluate_final_svg_bytes(raw, src, image_class="clean_logo")
    assert rep.sha256 == hashlib.sha256(raw).hexdigest()
    assert rep.metrics["A_structure"]["byte_size"] == len(raw)
    assert rep.verdict in {"production_ready", "needs_review", "failed"}


def test_exact_xml_metrics_ignore_comments_and_references() -> None:
    from app.final_artifact_evaluator import _structure_check

    raw = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
      <!-- <path d="M0 0"/><linearGradient id="fake"/> -->
      <metadata>path gradient M L C Z</metadata>
      <defs><linearGradient id="g"><stop offset="0"/></linearGradient></defs>
      <path d="M0 0 L10 0 C10 1 11 2 12 3 Z" fill="url(#g)" stroke="url(#g)"/>
    </svg>'''
    metrics, failures, codes, _root = _structure_check(raw)
    assert not failures, codes
    assert metrics["path_count"] == 1
    assert metrics["node_count"] == 4
    assert metrics["linear_gradient_count"] == 1
    assert metrics["gradient_definition_count"] == 1
    assert metrics["gradient_reference_count"] == 2
    assert metrics["gradient_count"] == 1


def test_non_gradient_url_reference_is_not_counted_as_gradient() -> None:
    from app.final_artifact_evaluator import _structure_check

    raw = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
      <defs><pattern id="p" width="2" height="2" patternUnits="userSpaceOnUse">
        <rect width="2" height="2" fill="red"/>
      </pattern></defs><rect width="20" height="20" fill="url(#p)"/>
    </svg>'''
    metrics, failures, codes, _root = _structure_check(raw)
    assert not failures, codes
    assert metrics["gradient_definition_count"] == 0
    assert metrics["gradient_reference_count"] == 0


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b'<script>alert(1)</script>', "forbidden_script"),
        (b'<path d="M0 0 L NaN 2"/>', "nonfinite_geometry"),
        (b'<rect onload="alert(1)"/>', "event_handler"),
        (b'<image href="data:image/png;base64,AA=="/>', "embedded_raster"),
        (b'<use href="https://example.invalid/x.svg#x"/>', "external_reference"),
        (b'<rect style="fill:url(https://example.invalid/x.svg#g)"/>', "external_reference"),
        (b'<style>@import url(https://example.invalid/x.css)</style>', "unsafe_css"),
        (b'<feImage href="#local"/>', "forbidden_feimage"),
    ],
)
def test_unsafe_svg_is_failed_without_rendering(body: bytes, code: str) -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    raw = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">' + body + b'</svg>'
    src = np.full((10, 10, 3), 255, np.uint8)
    rep = evaluate_final_svg_bytes(raw, src)
    assert rep.verdict == "failed"
    assert code in rep.hard_fail_codes
    assert "fail" != rep.verdict


def test_alpha_and_photo_cannot_be_false_ready() -> None:
    from app.final_artifact_evaluator import evaluate_final_svg_bytes

    raw = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="white"/></svg>'
    src = np.full((16, 16, 3), 255, np.uint8)
    alpha = np.tile(np.arange(16, dtype=np.uint8) * 17, (16, 1))
    alpha_rep = evaluate_final_svg_bytes(raw, src, source_alpha=alpha)
    photo_rep = evaluate_final_svg_bytes(raw, src, image_class="photo")
    assert alpha_rep.verdict == "failed"
    alpha_metrics = alpha_rep.metrics["G_gradient_alpha"]
    assert alpha_metrics["source_has_alpha"] is True
    assert alpha_metrics["alpha_fidelity_status"] == "failed"
    assert "alpha_iou" in alpha_metrics and "alpha_mae" in alpha_metrics
    assert set(alpha_rep.hard_fail_codes) & {
        "alpha_iou_below_min", "alpha_mae_above_max",
        "alpha_black_ssim_below_min", "alpha_checker_ssim_below_min",
    }
    assert photo_rep.verdict != "production_ready"
    assert "photo_vector_fidelity" in photo_rep.unmeasured_required


def _write_job(root: Path, job_id: str, *, owner: str = "owner@example.com") -> tuple[Path, str]:
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    svg_path = job_dir / f"{job_id}.svg"
    raw = _svg_bytes()
    svg_path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    report = {
        "job_id": job_id,
        "user": {"email": owner},
        "final_artifact": {
            "downloadable_formats": ["svg"],
            "artifacts": {"svg": {"sha256": sha, "structural_safe": True}},
        },
    }
    (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return svg_path, sha


def test_download_gate_enforces_session_owner_format_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    job_id = "a" * 32
    svg_path, sha = _write_job(tmp_path, job_id)
    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)

    def require_user(session: str | None):
        users = {
            "owner": {"email": "owner@example.com", "role": "user"},
            "other": {"email": "other@example.com", "role": "user"},
            "admin": {"email": "admin@example.com", "role": "admin"},
        }
        if session not in users:
            raise HTTPException(status_code=401, detail="login")
        return users[session]

    monkeypatch.setattr(main, "_require_user", require_user)
    client = TestClient(main.app)
    url = f"/api/download/{job_id}/svg"
    assert client.get(url).status_code == 401
    assert client.get(url, cookies={"session": "other"}).status_code == 404
    owner = client.get(url, cookies={"session": "owner"})
    assert owner.status_code == 200
    assert hashlib.sha256(owner.content).hexdigest() == sha
    assert client.get(url, cookies={"session": "admin"}).status_code == 200
    pdf = client.get(f"/api/download/{job_id}/pdf", cookies={"session": "owner"})
    assert pdf.status_code == 409
    assert pdf.json()["detail"]["code"] == "artifact_not_downloadable"
    svg_path.write_bytes(svg_path.read_bytes() + b"\n")
    changed = client.get(url, cookies={"session": "owner"})
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "artifact_hash_mismatch"


def test_download_gate_fails_closed_on_missing_or_bad_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    monkeypatch.setattr(main, "JOBS_ROOT", tmp_path)
    monkeypatch.setattr(main, "_require_user", lambda _session: {"email": "owner@example.com", "role": "user"})
    client = TestClient(main.app)
    job_id = "b" * 32
    assert client.get(f"/api/download/{job_id}/svg", cookies={"session": "owner"}).status_code == 404
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "report.json").write_text("{not-json", encoding="utf-8")
    bad = client.get(f"/api/download/{job_id}/svg", cookies={"session": "owner"})
    assert bad.status_code == 409
    assert bad.json()["detail"]["code"] == "artifact_report_invalid"
