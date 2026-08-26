from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import pipeline_smoke


def test_isolated_timeout_emits_bounded_relative_work_root_evidence(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    work_root = tmp_path / "work"
    job_dir = work_root / "qualification-public-17"
    job_dir.mkdir(parents=True)
    (job_dir / "candidate.svg").write_text("<svg/>", encoding="utf-8")
    (job_dir / "candidate_refit.svg").write_text("<svg><path/></svg>", encoding="utf-8")

    class FakeProcess:
        exitcode = -15

        def __init__(self) -> None:
            self.alive = True

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

    class FakeQueue:
        def empty(self) -> bool:
            return True

    class FakeContext:
        def Queue(self, maxsize=1):
            return FakeQueue()

        def Process(self, target, args):
            return FakeProcess()

    monkeypatch.setattr(
        pipeline_smoke.multiprocessing, "get_context", lambda mode: FakeContext()
    )
    case = type(
        "Case",
        (),
        {"case_id": "qualification-public-17", "to_dict": lambda self: {}},
    )()

    with pytest.raises(
        TimeoutError,
        match="isolated benchmark repeat timed out: qualification-public-17",
    ):
        pipeline_smoke._run_case_isolated(
            case,
            corpus_root=tmp_path / "corpus",
            work_root=work_root,
            engine_version="test",
            timeout_seconds=1,
        )

    stderr = capsys.readouterr().err
    line = next(
        item
        for item in stderr.splitlines()
        if item.startswith("isolated_benchmark_timeout_evidence=")
    )
    payload = json.loads(line.split("=", 1)[1])
    assert payload["file_count"] == 2
    assert {entry["path"] for entry in payload["latest"]} == {
        "qualification-public-17/candidate.svg",
        "qualification-public-17/candidate_refit.svg",
    }
    assert all(entry["bytes"] > 0 for entry in payload["latest"])
    assert str(tmp_path) not in stderr
