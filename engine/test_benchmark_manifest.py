from pathlib import Path

import pytest

from benchmark.manifest import (
    REQUIRED_METRICS,
    BenchmarkCase,
    BenchmarkResult,
    validate_manifest,
)


def _case(case_id: str = "logo-001", category: str = "logos") -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        category=category,
        source_path=f"fixtures/{case_id}.png",
        license_id="CC0-1.0",
        source_sha256="a" * 64,
        tags=("flat-color",),
    )


def test_manifest_is_deterministic_and_sorted() -> None:
    cases = validate_manifest([_case("z-case"), _case("a-case")])
    assert [case.case_id for case in cases] == ["a-case", "z-case"]


def test_manifest_rejects_duplicate_ids_and_unknown_category() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest([_case(), _case()])
    with pytest.raises(ValueError, match="unsupported"):
        _case(category="unknown").validate()


def test_manifest_rejects_path_escape() -> None:
    case = BenchmarkCase(
        case_id="escape",
        category="logos",
        source_path="../secret.png",
        license_id="CC0-1.0",
        source_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="escapes"):
        validate_manifest([case], root=Path("benchmark"))


def test_result_requires_complete_metric_schema() -> None:
    metrics = {name: 0.0 for name in REQUIRED_METRICS}
    result = BenchmarkResult(
        case_id="logo-001",
        engine_version="benchmark-v1",
        metrics=metrics,
        artifact_sha256="c" * 64,
    )
    payload = result.to_dict()
    assert set(payload["metrics"]) == REQUIRED_METRICS

    metrics.pop("ssim")
    with pytest.raises(ValueError, match="missing benchmark metrics"):
        BenchmarkResult(
            case_id="logo-001",
            engine_version="benchmark-v1",
            metrics=metrics,
        ).validate()
