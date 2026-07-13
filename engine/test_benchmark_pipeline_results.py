import hashlib
import json
from pathlib import Path

from PIL import Image

from benchmark.manifest import BenchmarkCase
from benchmark.pipeline_results import extract_metrics, run_case, write_results


def _pipeline_result() -> dict:
    return {
        "final_svg_sha256": "d" * 64,
        "legacy_candidate_report": {"metrics": {"fidelity_score": 0.98}},
        "quality_report": {
            "metrics": {
                "B_appearance": {"ssim": 0.97, "edge_f1": 0.96},
                "C_color": {"de00_p95": 1.2},
                "G_gradient_alpha": {"alpha_iou": 0.95},
            }
        },
        "final_artifact": {"exact_metrics": {"path_count": 7, "svg_bytes": 1234}},
    }


def test_extracts_only_measured_pipeline_metrics() -> None:
    metrics = extract_metrics(_pipeline_result(), elapsed_ms=12.5, peak_rss_mb=44.0)
    assert metrics == {
        "fidelity": 0.98,
        "ssim": 0.97,
        "edge_f1": 0.96,
        "alpha_iou": 0.95,
        "delta_e00": 1.2,
        "path_count": 7,
        "svg_bytes": 1234,
        "render_ms": 12.5,
        "peak_rss_mb": 44.0,
    }


def test_missing_quality_values_remain_unmeasured() -> None:
    metrics = extract_metrics({"final_artifact": {"exact_metrics": {}}}, elapsed_ms=1.0, peak_rss_mb=None)
    assert metrics["ssim"] is None
    assert metrics["peak_rss_mb"] is None


def test_run_case_verifies_source_hash_and_preserves_winner(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "case.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(source)
    case = BenchmarkCase(
        case_id="case-1",
        category="logos",
        source_path="case.png",
        license_id="CC0-1.0",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    calls = []

    def fake_pipeline(image, original_path, trace_mode, job_dir):
        calls.append((image.size, original_path, trace_mode, job_dir))
        return _pipeline_result()

    result = run_case(
        case,
        corpus_root=corpus,
        work_root=tmp_path / "work",
        pipeline=fake_pipeline,
        engine_version="test-sha",
        peak_rss_mb=10.0,
    )
    assert result.artifact_sha256 == "d" * 64
    assert calls[0][0] == (8, 8)


def test_results_are_written_in_case_order(tmp_path: Path) -> None:
    from benchmark.manifest import BenchmarkResult

    metrics = {name: 1.0 for name in (
        "fidelity", "ssim", "edge_f1", "alpha_iou", "delta_e00",
        "path_count", "svg_bytes", "render_ms", "peak_rss_mb"
    )}
    path = tmp_path / "results.json"
    write_results(path, [
        BenchmarkResult("z", "v", metrics, "a" * 64),
        BenchmarkResult("a", "v", metrics, "b" * 64),
    ])
    payload = json.loads(path.read_text())
    assert [item["case_id"] for item in payload["results"]] == ["a", "z"]
