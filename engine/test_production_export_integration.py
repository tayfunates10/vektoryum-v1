from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import app.production_export_integration as integration
from app.pipeline_canonical_report import PipelineCanonicalSvgReport
from app.shadow_svg_document import ShadowSvgDocumentReport
from app.shadow_svg_promotion_gate import ShadowSvgPromotionGateReport


LEGACY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'
CANONICAL_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4" viewBox="0 0 4 4">'
    '<path id="face-1" d="M 0 0 L 4 0 L 4 4 L 0 4 Z" fill="#ff0000" '
    'fill-rule="evenodd"/></svg>'
)


def _report(svg_text: str = CANONICAL_SVG) -> PipelineCanonicalSvgReport:
    digest = sha256(svg_text.encode("utf-8")).hexdigest()
    document = ShadowSvgDocumentReport(
        svg_text=svg_text,
        width=4,
        height=4,
        face_count=1,
        document_sha256=digest,
        valid=True,
        errors=(),
    )
    promotion = ShadowSvgPromotionGateReport(
        ready=True,
        checked_runs=3,
        document_sha256=digest,
        face_count=1,
        errors=(),
    )
    candidate = SimpleNamespace(
        valid=True,
        document=document,
        promotion=promotion,
        palette_size=1,
        errors=(),
    )
    return PipelineCanonicalSvgReport(
        schema_version="canonical-pipeline-report-v1",
        enabled=True,
        attempted=True,
        status="ready",
        candidate=candidate,
        document_sha256=digest,
        face_count=1,
        palette_size=1,
        errors=(),
    )


def _legacy_exporter(legacy_text: str = LEGACY_SVG):
    def export_all(*, best_svg, job_dir, job_id, candidate_id=None, formats=("svg",), png_size=None):
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        svg_path = job_dir / f"{job_id}.svg"
        svg_path.write_text(legacy_text, encoding="utf-8")
        outputs["svg"] = str(svg_path)
        for fmt in formats:
            if fmt == "svg":
                continue
            dst = job_dir / f"{job_id}.{fmt}"
            dst.write_text(f"legacy-{fmt}", encoding="utf-8")
            outputs[fmt] = str(dst)
        return outputs, {}

    return export_all


def _register(job_dir: Path, report: PipelineCanonicalSvgReport | None = None) -> PipelineCanonicalSvgReport:
    value = report or _report()
    assert integration.register_pipeline_canonical_report(
        job_dir,
        {"best": {"name": "legacy", "svg_path": Path("legacy.svg")},
         "canonical_svg_candidate": value},
    ) is True
    return value


def test_registry_is_job_scoped_and_one_shot(tmp_path) -> None:
    report = _register(tmp_path)

    assert integration.pending_report_count() == 1
    assert integration.consume_pipeline_canonical_report(tmp_path) is report
    assert integration.consume_pipeline_canonical_report(tmp_path) is None
    assert integration.pending_report_count() == 0


def test_runtime_flag_off_preserves_legacy_artifacts(tmp_path) -> None:
    report = _register(tmp_path)
    outputs, errors = integration.export_all_with_canonical(
        _legacy_exporter(),
        best_svg=tmp_path / "best.svg",
        job_dir=tmp_path,
        job_id="job",
        formats=("svg", "pdf"),
        environ={"VEKTORYUM_CANONICAL_SVG_SHA256": report.document_sha256},
    )

    assert Path(outputs["svg"]).read_text(encoding="utf-8") == LEGACY_SVG
    assert Path(outputs["pdf"]).read_text(encoding="utf-8") == "legacy-pdf"
    assert errors == {}
    assert integration.pending_report_count() == 0


def test_digest_pinned_cutover_publishes_exact_svg_and_regenerates_derivatives(
    monkeypatch, tmp_path
) -> None:
    report = _register(tmp_path)
    expected_digest = report.document_sha256

    def derived(src, dst):
        assert Path(src).read_text(encoding="utf-8") == CANONICAL_SVG
        Path(dst).write_text(sha256(Path(src).read_bytes()).hexdigest(), encoding="utf-8")
        return Path(dst)

    def png(src, dst, width=None, height=None):
        assert (width, height) == (4, 4)
        return derived(src, dst)

    monkeypatch.setattr(integration, "export_pdf", derived)
    monkeypatch.setattr(integration, "export_eps", derived)
    monkeypatch.setattr(integration, "export_dxf", derived)
    monkeypatch.setattr(integration, "export_png", png)

    outputs, errors = integration.export_all_with_canonical(
        _legacy_exporter(),
        best_svg=tmp_path / "best.svg",
        job_dir=tmp_path,
        job_id="job",
        formats=("svg", "pdf", "eps", "dxf", "png"),
        png_size=(4, 4),
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "on",
            "VEKTORYUM_CANONICAL_SVG_SHA256": expected_digest,
        },
    )

    svg_path = Path(outputs["svg"])
    assert svg_path.read_bytes() == CANONICAL_SVG.encode("utf-8")
    assert sha256(svg_path.read_bytes()).hexdigest() == expected_digest
    for fmt in ("pdf", "eps", "dxf", "png"):
        assert Path(outputs[fmt]).read_text(encoding="utf-8") == expected_digest
    assert errors == {}


def test_wrong_approved_digest_fails_closed_to_legacy(tmp_path) -> None:
    _register(tmp_path)
    outputs, errors = integration.export_all_with_canonical(
        _legacy_exporter(),
        best_svg=tmp_path / "best.svg",
        job_dir=tmp_path,
        job_id="job",
        formats=("svg", "pdf"),
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "true",
            "VEKTORYUM_CANONICAL_SVG_SHA256": "b" * 64,
        },
    )

    assert Path(outputs["svg"]).read_text(encoding="utf-8") == LEGACY_SVG
    assert Path(outputs["pdf"]).read_text(encoding="utf-8") == "legacy-pdf"
    assert "approved digest does not match candidate" in errors["canonical_svg"]


def test_failed_canonical_derivative_removes_stale_legacy_file(monkeypatch, tmp_path) -> None:
    report = _register(tmp_path)

    def fail_pdf(src, dst):
        raise RuntimeError("pdf failed")

    monkeypatch.setattr(integration, "export_pdf", fail_pdf)
    outputs, errors = integration.export_all_with_canonical(
        _legacy_exporter(),
        best_svg=tmp_path / "best.svg",
        job_dir=tmp_path,
        job_id="job",
        formats=("svg", "pdf"),
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "yes",
            "VEKTORYUM_CANONICAL_SVG_SHA256": report.document_sha256,
        },
    )

    assert Path(outputs["svg"]).read_text(encoding="utf-8") == CANONICAL_SVG
    assert "pdf" not in outputs
    assert errors["pdf"] == "pdf failed"
    assert not (tmp_path / "job.pdf").exists()


def test_post_publish_digest_failure_atomically_restores_legacy(monkeypatch, tmp_path) -> None:
    report = _register(tmp_path)

    def corrupt_publish(*, legacy_svg, destination, cutover, environ=None):
        Path(destination).write_text("<svg>corrupt</svg>", encoding="utf-8")
        return SimpleNamespace(
            published=True,
            selection=SimpleNamespace(promoted=True, errors=()),
        )

    monkeypatch.setattr(integration, "publish_runtime_svg", corrupt_publish)
    outputs, errors = integration.export_all_with_canonical(
        _legacy_exporter(),
        best_svg=tmp_path / "best.svg",
        job_dir=tmp_path,
        job_id="job",
        formats=("svg",),
        environ={
            "VEKTORYUM_CANONICAL_SVG_ENABLED": "on",
            "VEKTORYUM_CANONICAL_SVG_SHA256": report.document_sha256,
        },
    )

    assert Path(outputs["svg"]).read_text(encoding="utf-8") == LEGACY_SVG
    assert "published canonical SVG digest mismatch" in errors["canonical_svg"]


def test_winnerless_or_invalid_report_is_never_registered(tmp_path) -> None:
    report = _report()
    assert integration.register_pipeline_canonical_report(
        tmp_path,
        {"best": None, "canonical_svg_candidate": report},
    ) is False
    assert integration.pending_report_count() == 0
