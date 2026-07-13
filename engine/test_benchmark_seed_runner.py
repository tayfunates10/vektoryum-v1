import json
from pathlib import Path

from benchmark.seed_runner import CATEGORIES, generate_seed_corpus, write_reports


def test_seed_runner_generates_all_categories_and_stable_hashes(tmp_path: Path):
    first = generate_seed_corpus(tmp_path / "first")
    second = generate_seed_corpus(tmp_path / "second")

    assert {case.category for case in first} == set(CATEGORIES)
    assert [case.source_sha256 for case in first] == [case.source_sha256 for case in second]
    assert all(len(case.source_sha256) == 64 for case in first)


def test_seed_runner_writes_json_and_html_reports(tmp_path: Path):
    cases = generate_seed_corpus(tmp_path)
    write_reports(tmp_path, cases)

    payload = json.loads((tmp_path / "seed_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "benchmark-seed-v1"
    assert payload["case_count"] == len(CATEGORIES)
    assert payload["categories"] == sorted(CATEGORIES)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Benchmark v1 Seed Corpus" in html
    assert "CC0-1.0" in html
