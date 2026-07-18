from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "engine/app/transform_journal.py"
TESTS = ROOT / "engine/test_transform_journal.py"
DIAGNOSTIC = ROOT / "engine/regression/rfv3e_viewbox_journal_diagnostics.py"
DIAGNOSTIC_TEST = ROOT / "engine/regression/test_rfv3e_viewbox_journal_diagnostics.py"
DIAGNOSTIC_DOC = ROOT / "docs/real_world_fidelity/rfv-3e-exact-metric-path-viewbox-diagnostics.md"


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
insert = '''\n\ndef test_alpha_measurement_is_scoped_to_source_dimension_restore(\n    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,\n) -> None:\n    import app.fidelity as fidelity\n    import app.source_truth as source_truth\n    from app.transform_journal import TransformJournal\n\n    source = _square_source()\n    monkeypatch.setattr(fidelity, "render_svg_to_rgb", lambda *_args, **_kwargs: source.copy())\n\n    def stable_alpha(_path: Path, width: int, height: int) -> np.ndarray:\n        rgba = np.zeros((height, width, 4), dtype=np.uint8)\n        rgba[24:104, 24:104, :3] = (227, 0, 11)\n        rgba[24:104, 24:104, 3] = 255\n        return rgba\n\n    monkeypatch.setattr(source_truth, "render_svg_to_rgba", stable_alpha)\n    parent = tmp_path / "parent.svg"\n    candidate = tmp_path / "candidate.svg"\n    parent.write_bytes(_square_svg())\n    candidate.write_bytes(_square_svg("<metadata>downstream-change</metadata>"))\n    journal = TransformJournal(parent, source, required_metrics={"alpha_fidelity"})\n    accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)\n\n    assert accepted == parent\n    assert stage["status"] == "rolled_back"\n    assert "required_metric_unmeasured" in stage["reason_codes"]\n    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]\n    assert stage["required_unmeasured"] == ["alpha_fidelity"]\n    assert stage["alpha_comparison"] is None\n\n'''
tests = replace_once(
    tests,
    '''\n\ndef test_assertions_are_real() -> None:\n''',
    insert + '''\ndef test_assertions_are_real() -> None:\n''',
    "stage scope regression test",
)
TESTS.write_text(tests, encoding="utf-8")

DIAGNOSTIC.write_text('''"""Immutable historical verifier for the RFV-3E viewBox rollback evidence.\n\nThe committed JSON records the pre-fix behavior proved by PR #104. It is not\nrecomputed against current production code after PR #105; current behavior is\ncovered by the dedicated viewBox/alpha production contract.\n"""\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport re\nfrom pathlib import Path\nfrom typing import Any\n\nSCHEMA = "vektoryum-rfv3e-viewbox-journal-diagnostics-v1"\nSOURCE_MAIN_SHA = "19e91d10926f8709112b0afd6c576b886a5dfeb5"\nSOURCE_PR = 103\nSOURCE_HEAD_SHA = "92fa263a938a39f44c288109c8f05a8a38c98f7e"\nSOURCE_RUN_ID = 29623130466\nSOURCE_ARTIFACT_ID = 8424383328\nSOURCE_ARTIFACT_DIGEST = "sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0"\nCASES = ["qualification-public-10", "qualification-public-14", "qualification-public-18"]\n_PATH_OR_SECRET = re.compile(\n    r"(?:/home/|/tmp/|[A-Za-z]:\\\\|authorization\\s*:|bearer\\s+|token=|traceback)",\n    re.IGNORECASE,\n)\n\n\ndef validate_evidence(payload: dict[str, Any]) -> None:\n    expected_source = {\n        "repository": "tayfunates10/vektoryum-v1",\n        "main_sha": SOURCE_MAIN_SHA,\n        "pull_request": SOURCE_PR,\n        "measurement_head_sha": SOURCE_HEAD_SHA,\n        "workflow_run_id": SOURCE_RUN_ID,\n        "aggregate_artifact_id": SOURCE_ARTIFACT_ID,\n        "aggregate_artifact_digest": SOURCE_ARTIFACT_DIGEST,\n    }\n    expected_scope = {\n        "case_ids": CASES,\n        "observed_hard_fail_code": "viewbox_missing",\n        "observed_reason_code": "exact_component_metrics_missing",\n    }\n    if payload.get("schema") != SCHEMA:\n        raise ValueError("historical schema drift")\n    if payload.get("source") != expected_source:\n        raise ValueError("historical source binding drift")\n    if payload.get("scope") != expected_scope:\n        raise ValueError("historical scope binding drift")\n    diagnosis = payload.get("diagnosis")\n    if not isinstance(diagnosis, dict):\n        raise ValueError("historical diagnosis missing")\n    if diagnosis.get("direct_restore_added_viewbox") is not True:\n        raise ValueError("historical restore proof drift")\n    if diagnosis.get("repaired_viewbox") != "0 0 48 32":\n        raise ValueError("historical repaired viewBox drift")\n    if diagnosis.get("stage_measurement_required_unmeasured") != ["alpha_fidelity"]:\n        raise ValueError("historical required-unmeasured signature drift")\n    if diagnosis.get("journal_with_alpha_requirement") != {\n        "accepted": False,\n        "status": "rolled_back",\n        "reason_codes": ["required_metric_unmeasured"],\n        "output_viewbox_present": False,\n    }:\n        raise ValueError("historical rollback signature drift")\n    if diagnosis.get("journal_without_alpha_requirement") != {\n        "accepted": True,\n        "status": "accepted",\n        "reason_codes": ["metrics_non_regressing"],\n        "output_viewbox_present": True,\n    }:\n        raise ValueError("historical control signature drift")\n    if diagnosis.get("root_cause_status") != "proven":\n        raise ValueError("historical root-cause status drift")\n    if diagnosis.get("root_cause_class") != "transform_journal_required_alpha_metric_deadlock":\n        raise ValueError("historical root-cause class drift")\n    if diagnosis.get("production_fix_authorized") is not False:\n        raise ValueError("historical diagnostics cannot authorize production")\n    if payload.get("release_decision") != "no_go" or payload.get("rfv4_allowed") is not False:\n        raise ValueError("release/RFV-4 decision drift")\n    if _PATH_OR_SECRET.search(json.dumps(payload, sort_keys=True, ensure_ascii=True)):\n        raise ValueError("path, secret or traceback leaked into historical evidence")\n\n\ndef _parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Verify immutable RFV-3E viewBox evidence")\n    parser.add_argument("verify", nargs="?")\n    parser.add_argument("--evidence", type=Path, required=True)\n    return parser.parse_args()\n\n\ndef main() -> int:\n    args = _parse_args()\n    try:\n        payload = json.loads(args.evidence.read_text(encoding="utf-8"))\n        validate_evidence(payload)\n    except (OSError, json.JSONDecodeError, ValueError) as exc:\n        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))\n        return 2\n    print(json.dumps({"status": "historical_verified", "root_cause": payload["diagnosis"]["root_cause_class"]}, sort_keys=True))\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n''', encoding="utf-8")

DIAGNOSTIC_TEST.write_text('''"""Tests for immutable historical RFV-3E viewBox evidence."""\nfrom __future__ import annotations\n\nimport copy\nimport json\nimport unittest\nfrom pathlib import Path\n\nfrom engine.regression.rfv3e_viewbox_journal_diagnostics import (\n    CASES, SCHEMA, validate_evidence,\n)\n\nROOT = Path(__file__).resolve().parents[2]\nEVIDENCE = ROOT / "docs/real_world_fidelity/evidence/rfv3e_viewbox_journal_diagnostics.json"\n\n\nclass RFV3EViewBoxJournalDiagnosticsTests(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls) -> None:\n        cls.payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))\n\n    def test_committed_historical_evidence_is_valid(self):\n        validate_evidence(self.payload)\n\n    def test_exact_historical_bindings(self):\n        self.assertEqual(self.payload["schema"], SCHEMA)\n        self.assertEqual(self.payload["scope"]["case_ids"], CASES)\n        self.assertEqual(self.payload["source"]["pull_request"], 103)\n        self.assertEqual(self.payload["source"]["workflow_run_id"], 29623130466)\n        self.assertEqual(self.payload["source"]["aggregate_artifact_id"], 8424383328)\n\n    def test_historical_root_cause_signature(self):\n        diagnosis = self.payload["diagnosis"]\n        self.assertEqual(diagnosis["stage_measurement_required_unmeasured"], ["alpha_fidelity"])\n        self.assertEqual(diagnosis["root_cause_class"], "transform_journal_required_alpha_metric_deadlock")\n        self.assertFalse(diagnosis["production_fix_authorized"])\n\n    def test_tampered_historical_signature_is_rejected(self):\n        tampered = copy.deepcopy(self.payload)\n        tampered["diagnosis"]["stage_measurement_required_unmeasured"] = []\n        with self.assertRaisesRegex(ValueError, "required-unmeasured"):\n            validate_evidence(tampered)\n\n    def test_release_or_rfv4_drift_is_rejected(self):\n        for field, value in (("release_decision", "go"), ("rfv4_allowed", True)):\n            with self.subTest(field=field):\n                tampered = copy.deepcopy(self.payload)\n                tampered[field] = value\n                with self.assertRaisesRegex(ValueError, "decision drift"):\n                    validate_evidence(tampered)\n\n    def test_path_or_secret_leakage_is_rejected(self):\n        tampered = copy.deepcopy(self.payload)\n        tampered["diagnosis"]["root_cause_summary"] = "/tmp/raw/winner.svg"\n        with self.assertRaisesRegex(ValueError, "leaked"):\n            validate_evidence(tampered)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''', encoding="utf-8")

DIAGNOSTIC_DOC.write_text('''# RFV-3E exact metric path viewBox diagnosis — historical evidence\n\n## Status\n\nThis document and its JSON evidence are an immutable historical record of the defect proved by PR #104. They are no longer recomputed against current production code.\n\nThe bound source remains:\n\n- PR #103;\n- main SHA `19e91d10926f8709112b0afd6c576b886a5dfeb5`;\n- RFV-3B run `29623130466`;\n- aggregate artifact `8424383328`;\n- digest `sha256:ff45ec277fe8162f3be117cff76ec3fb82e3cafc4d563941fcabd145ff1e8cb0`.\n\n## Historical finding\n\nFor `qualification-public-10`, `qualification-public-14`, and `qualification-public-18`, `_restore_source_dimensions` created a valid viewBox, but the pre-fix RGBA journal rolled the candidate back because `alpha_fidelity` remained unmeasured. The proven class is:\n\n```text\ntransform_journal_required_alpha_metric_deadlock\n```\n\n## Superseding production contract\n\nPR #105 introduces a separate production contract. Alpha preservation is measured only for the mandatory `restore_source_dimensions` stage. Downstream mutators retain the prior fail-closed scope. Current behavior is tested in `engine/test_transform_journal.py` and `.github/workflows/real-world-fidelity-rfv3e-viewbox-fix.yml`.\n\nThe historical JSON is intentionally unchanged; changing current code must not rewrite past evidence.\n\n## Canonical state\n\n- RFV-3: pending;\n- release decision: `no_go`;\n- `rfv4_allowed`: `false`;\n- RFV-4: pending.\n''', encoding="utf-8")

print("RFV-3E stage-scoped patch and historical diagnostics retirement applied")
