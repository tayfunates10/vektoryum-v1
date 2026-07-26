"""RFV-3D3 — aday arama süresi: turnuva elemesi ve paylaşılan ölçüm önbelleği.

Her iki mekanizma da SEÇİM-EŞDEĞERDİR: hiçbir eşik, sıra veya kabul kuralı
değişmez, yalnız sonucu değiştiremeyecek iş elenir. Testler bunu ölçer.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app import transform_journal as journal_module
from app.alpha_candidate_painter import apply_candidate_painter_reconstruction
from app.transform_journal import TransformJournal

from test_alpha_candidate_painter import _covering_candidate, _soft_ring_source

_SOURCE_RGB = np.zeros((16, 16, 3), dtype=np.uint8)
_SVG = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
    'viewBox="0 0 16 16"><path d="M2 2H14V14H2Z" fill="#204080"/></svg>'
).encode("utf-8")


def _journal(tmp_path: Path, cache: dict | None, **kwargs) -> TransformJournal:
    baseline = tmp_path / "baseline.svg"
    if not baseline.exists():
        baseline.write_bytes(_SVG)
    return TransformJournal(
        baseline, _SOURCE_RGB, required_metrics=set(), measurement_cache=cache, **kwargs
    )


def test_shared_cache_measures_identical_bytes_once(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    original = journal_module._measure_svg_bytes

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(journal_module, "_measure_svg_bytes", counting)

    cache: dict = {}
    first = _journal(tmp_path, cache)
    second = _journal(tmp_path, cache)
    first_metrics = first._measure(_SVG)
    second_metrics = second._measure(_SVG)

    assert len(calls) == 1
    assert first_metrics is second_metrics
    # Bütçe muhasebesi korunur: önbellek isabetinde ilk ölçümün süresi yeniden
    # işlenir, böylece budget_exhausted davranışı değişmez.
    assert second.evaluation_seconds == pytest.approx(first.evaluation_seconds)
    assert second.evaluation_seconds > 0.0


def test_shared_cache_is_namespaced_by_measurement_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []
    original = journal_module._measure_svg_bytes

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(journal_module, "_measure_svg_bytes", counting)

    cache: dict = {}
    _journal(tmp_path, cache, max_side=256)._measure(_SVG)
    _journal(tmp_path, cache, max_side=512)._measure(_SVG)
    assert len(calls) == 2

    other_source = np.full((16, 16, 3), 255, dtype=np.uint8)
    baseline = tmp_path / "baseline.svg"
    TransformJournal(
        baseline, other_source, required_metrics=set(), measurement_cache=cache,
        max_side=256,
    )._measure(_SVG)
    assert len(calls) == 3


def test_default_journal_keeps_private_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    original = journal_module._measure_svg_bytes

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(journal_module, "_measure_svg_bytes", counting)
    _journal(tmp_path, None)._measure(_SVG)
    _journal(tmp_path, None)._measure(_SVG)
    assert len(calls) == 2


_ATTEMPT_MARKER = "source_alpha_candidate_painter_attempts="


def _painter_attempts(captured: str) -> list[dict]:
    """Attempt ledger'ı yapılandırılmış stderr satırından çözer."""
    for line in reversed(captured.splitlines()):
        if line.startswith(_ATTEMPT_MARKER):
            return json.loads(line[len(_ATTEMPT_MARKER):])
    raise AssertionError("attempt ledger stderr'de bulunamadı")


def test_tournament_prune_never_discards_a_possible_winner(capfd) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source.png"
        svg_path = root / "candidate.svg"
        _soft_ring_source(source_path)
        _covering_candidate(svg_path)

        report = apply_candidate_painter_reconstruction(svg_path, source_path, "logo_color")

    assert report["applied"]
    attempts = _painter_attempts(capfd.readouterr().err)
    assert len(attempts) == int(report["painter_attempt_count"])
    assert attempts, "attempt ledger boş olmamalı"

    accepted = [item for item in attempts if item["status"] == "accepted"]
    pruned = [item for item in attempts if item["status"] == "byte_dominated"]
    assert accepted, "en az bir aday kabul edilmeli"
    assert pruned, "bu vakada eleme devreye girmeli (aksi halde test anlamsızlaşır)"

    winner_bytes = min(int(item["actual_serialized_bytes"]) for item in accepted)
    for item in pruned:
        # Elenen her aday KESİN olarak daha büyüktür; turnuva anahtarı
        # (byte, path, node, sıra) sözlüksel olduğundan kazanması imkânsızdı.
        assert int(item["actual_serialized_bytes"]) > winner_bytes
        assert item["validation_stage"] == "tournament_prune"
        assert item["journal_gate_started"] is False
        assert item["validation_started"] is False


def test_painter_result_is_deterministic_under_pruning() -> None:
    digests: list[str] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            svg_path = root / "candidate.svg"
            _soft_ring_source(source_path)
            _covering_candidate(svg_path)
            report = apply_candidate_painter_reconstruction(
                svg_path, source_path, "logo_color"
            )
            assert report["applied"]
            digests.append(hashlib.sha256(svg_path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
