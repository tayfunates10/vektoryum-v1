from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "benchmark-seed.yml"


def test_pull_request_quality_and_timing_use_same_runner_base() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'if [ "${GITHUB_EVENT_NAME}" = "pull_request" ] && [ -f same_runner_baseline/pipeline_results.json ]; then' in text
    assert 'args+=(--baseline same_runner_baseline/pipeline_results.json)' in text
    assert 'args+=(--timing-baseline same_runner_baseline/pipeline_results.json)' in text


def test_previous_successful_artifact_is_only_non_pr_quality_fallback() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    pr_branch = text.index('if [ "${GITHUB_EVENT_NAME}" = "pull_request" ]')
    fallback = text.index('elif [ -f baseline_artifact/pipeline/pipeline_results.json ]; then', pr_branch)
    command = text.index('python -m benchmark.release_gate "${args[@]}"', fallback)

    section = text[pr_branch:command]
    assert section.index('--baseline same_runner_baseline/pipeline_results.json') < section.index('elif [ -f baseline_artifact/pipeline/pipeline_results.json ]; then')
    assert '--baseline baseline_artifact/pipeline/pipeline_results.json' in section
