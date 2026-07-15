import json
import math
import re
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "fidelity99_roadmap.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPONENTS = ("geometric", "edge", "color", "typography", "print_readiness")


def validate_final_gate(records, minimum_cases=24, overall_threshold=0.990, component_threshold=0.980):
    if not isinstance(minimum_cases, int) or isinstance(minimum_cases, bool) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if not isinstance(records, list) or len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")
    ids = []
    overall_scores = []
    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        ids.append(case_id)
        for key in ("source_sha256", "artifact_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise AssertionError(f"invalid {key}")
        components = record.get("components")
        if not isinstance(components, dict) or set(components) != set(COMPONENTS):
            raise AssertionError("missing or unexpected component")
        for name in COMPONENTS:
            value = components[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AssertionError(f"non-finite {name}")
            if not 0 <= value <= 1:
                raise AssertionError(f"out-of-range {name}")
            if value < component_threshold:
                raise AssertionError("component fidelity regression")
        overall = record.get("overall_fidelity")
        if not isinstance(overall, (int, float)) or isinstance(overall, bool) or not math.isfinite(overall):
            raise AssertionError("non-finite overall fidelity")
        if not 0 <= overall <= 1:
            raise AssertionError("out-of-range overall fidelity")
        published = record.get("published")
        if not isinstance(published, bool):
            raise AssertionError("invalid publication state")
        if overall < overall_threshold:
            if published:
                raise AssertionError("rejected artifact published")
            raise AssertionError("overall fidelity regression")
        if not published:
            raise AssertionError("qualified artifact not published")
        overall_scores.append(float(overall))
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(overall_scores) < overall_threshold:
        raise AssertionError("mean overall fidelity regression")


class F998FinalFidelityGateTests(unittest.TestCase):
    def test_roadmap_is_complete_and_finite(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])
        self.assertEqual([p["status"] for p in phases[:7]], ["merged"] * 7)
        self.assertIn(phases[7]["status"], {"implemented", "merged"})
        self.assertTrue((ROOT / phases[7]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[7]["acceptance"]), 5)

    def test_accepts_qualified_artifacts(self):
        records = []
        for index in range(24):
            records.append({
                "case_id": f"case-{index}",
                "source_sha256": f"{index + 1:064x}",
                "artifact_sha256": f"{index + 101:064x}",
                "components": {name: 0.995 for name in COMPONENTS},
                "overall_fidelity": 0.995,
                "published": True,
            })
        validate_final_gate(records)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "source_sha256": "1" * 64,
            "artifact_sha256": "2" * 64,
            "components": {name: 0.995 for name in COMPONENTS},
            "overall_fidelity": 0.995,
            "published": True,
        }
        bad_sets = [
            [],
            [dict(base, overall_fidelity=0.989, published=True)],
            [dict(base, overall_fidelity=float("nan"))],
            [dict(base, source_sha256="bad")],
            [dict(base, components={"geometric": 0.995})],
            [dict(base, components={**base["components"], "color": 0.979})],
            [dict(base), dict(base)],
            [dict(base, published=False)],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_final_gate(records, minimum_cases=1)


if __name__ == "__main__":
    unittest.main()
