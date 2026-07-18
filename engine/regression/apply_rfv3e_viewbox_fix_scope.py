from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "engine/app/transform_journal.py"
TESTS = ROOT / "engine/test_transform_journal.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one binding, found {count}")
    return text.replace(old, new, 1)


journal = JOURNAL.read_text(encoding="utf-8")
journal = replace_once(
    journal,
    '''    required_metrics: set[str] | None = None,\n    _allow_topology_refinement: bool = True,\n''',
    '''    required_metrics: set[str] | None = None,\n    measure_alpha: bool = False,\n    _allow_topology_refinement: bool = True,\n''',
    "measurement signature",
)
journal = replace_once(
    journal,
    '''        if "alpha_fidelity" in set(required_metrics or ()):\n            render_rgba = _source_truth.render_svg_to_rgba(path, w, h)\n''',
    '''        if measure_alpha and "alpha_fidelity" in set(required_metrics or ()):\n            render_rgba = _source_truth.render_svg_to_rgba(path, w, h)\n''',
    "bounded rgba render",
)
journal = replace_once(
    journal,
    '''    if "alpha_fidelity" in set(required_metrics or ()):\n        if render_rgba is None:\n''',
    '''    if measure_alpha and "alpha_fidelity" in set(required_metrics or ()):\n        if render_rgba is None:\n''',
    "alpha metric publication",
)
journal = replace_once(
    journal,
    '''            required_metrics=required_metrics,\n            _allow_topology_refinement=False,\n''',
    '''            required_metrics=required_metrics,\n            measure_alpha=measure_alpha,\n            _allow_topology_refinement=False,\n''',
    "refinement alpha scope",
)
journal = replace_once(
    journal,
    '''        self.required_metrics = set(required_metrics or ())\n        self.max_side = min(512, max(256, int(max_side)))\n''',
    '''        self.required_metrics = set(required_metrics or ())\n        # Alpha measurement is deliberately stage-scoped. The proven defect only\n        # affects the mandatory coordinate-contract repair; opening alpha for every\n        # downstream mutator would change previously fail-closed production scope.\n        self._measurement_stage_id: str | None = None\n        self.max_side = min(512, max(256, int(max_side)))\n''',
    "journal stage context",
)
journal = replace_once(
    journal,
    '''    def _measure(self, data: bytes) -> dict[str, Any]:\n        sha = _sha(data)\n        if sha not in self._cache:\n            started = time.perf_counter()\n            try:\n                self._cache[sha] = _measure_svg_bytes(\n                    data, self.source_rgb, max_side=self.max_side,\n                    required_metrics=self.required_metrics,\n                )\n            finally:\n                self.evaluation_seconds += time.perf_counter() - started\n        return self._cache[sha]\n''',
    '''    def _measure(self, data: bytes) -> dict[str, Any]:\n        sha = _sha(data)\n        measure_alpha = self._measurement_stage_id == "restore_source_dimensions"\n        cache_key = f"{sha}:alpha={int(measure_alpha)}"\n        if cache_key not in self._cache:\n            started = time.perf_counter()\n            try:\n                self._cache[cache_key] = _measure_svg_bytes(\n                    data, self.source_rgb, max_side=self.max_side,\n                    required_metrics=self.required_metrics,\n                    measure_alpha=measure_alpha,\n                )\n            finally:\n                self.evaluation_seconds += time.perf_counter() - started\n        return self._cache[cache_key]\n''',
    "stage-scoped cache",
)
journal = replace_once(
    journal,
    '''        else:\n            before = self._measure(parent_data)\n            after = self._measure(candidate_data)\n            reasons = self._decide(before, after)\n            accepted = not reasons\n            status = "accepted" if accepted else "rolled_back"\n''',
    '''        else:\n            previous_stage = self._measurement_stage_id\n            self._measurement_stage_id = stage_id\n            try:\n                before = self._measure(parent_data)\n                after = self._measure(candidate_data)\n            finally:\n                self._measurement_stage_id = previous_stage\n            reasons = self._decide(before, after)\n            accepted = not reasons\n            status = "accepted" if accepted else "rolled_back"\n''',
    "record stage binding",
)
JOURNAL.write_text(journal, encoding="utf-8")

tests = TESTS.read_text(encoding="utf-8")
tests = replace_once(
    tests,
    '''    accepted, stage = journal.consider_candidate("alpha_loss", parent, candidate)\n''',
    '''    accepted, stage = journal.consider_candidate(\n        "restore_source_dimensions", parent, candidate,\n    )\n''',
    "alpha regression stage",
)
insert = '''\n\ndef test_alpha_measurement_is_scoped_to_source_dimension_restore(\n    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,\n) -> None:\n    import app.fidelity as fidelity\n    import app.source_truth as source_truth\n    from app.transform_journal import TransformJournal\n\n    source = _square_source()\n    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())\n\n    def stable_alpha(_path: Path, width: int, height: int) -> np.ndarray:\n        rgba = np.zeros((height, width, 4), dtype=np.uint8)\n        rgba[24:104, 24:104, :3] = (227, 0, 11)\n        rgba[24:104, 24:104, 3] = 255\n        return rgba\n\n    monkeypatch.setattr(source_truth, "render_svg_to_rgba", stable_alpha)\n    parent = tmp_path / "parent.svg"\n    candidate = tmp_path / "candidate.svg"\n    parent.write_bytes(_square_svg())\n    candidate.write_bytes(_square_svg("<metadata>downstream-change</metadata>"))\n    journal = TransformJournal(\n        parent, source, required_metrics={"alpha_fidelity"},\n    )\n    accepted, stage = journal.consider_candidate(\n        "boundary_refit", parent, candidate,\n    )\n\n    assert accepted == parent\n    assert stage["status"] == "rolled_back"\n    assert "required_metric_unmeasured" in stage["reason_codes"]\n    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]\n    assert stage["required_unmeasured"] == ["alpha_fidelity"]\n    assert stage["alpha_comparison"] is None\n\n'''
tests = replace_once(
    tests,
    '''\n\ndef test_assertions_are_real() -> None:\n''',
    insert + '''\ndef test_assertions_are_real() -> None:\n''',
    "stage scope regression test",
)
TESTS.write_text(tests, encoding="utf-8")
print("RFV-3E stage-scoped alpha patch applied")
