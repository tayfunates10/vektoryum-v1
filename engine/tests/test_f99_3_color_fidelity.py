import json
import math
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "fidelity99_roadmap.json"


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("empty evidence")
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def validate_color_evidence(records, minimum_cases):
    if not isinstance(minimum_cases, int) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")
    ids = []
    scores = []
    delta_es = []
    mismatch_total = 0
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        ids.append(case_id)
        for key in ("perceptual_color_score", "normalized_delta_e"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AssertionError(f"non-finite {key}")
            if not 0 <= value <= 1:
                raise AssertionError(f"out-of-range {key}")
        scores.append(float(record["perceptual_color_score"]))
        delta_es.append(float(record["normalized_delta_e"]))
        for key in ("palette_omissions", "palette_duplicates", "transparency_mismatches"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(f"invalid {key}")
            mismatch_total += value
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(scores) < 0.990:
        raise AssertionError("mean color score regression")
    if percentile(scores, 0.05) < 0.980:
        raise AssertionError("p05 color score regression")
    if percentile(delta_es, 0.95) > 0.020:
        raise AssertionError("p95 Delta E regression")
    if mismatch_total != 0:
        raise AssertionError("palette/transparency mismatch")


class F99ColorFidelityContractTests(unittest.TestCase):
    def test_roadmap_is_finite_and_ordered(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in roadmap["phases"]], [f"F99-{i}" for i in range(1, 9)])
        self.assertEqual(roadmap["phases"][0]["status"], "merged")
        self.assertEqual(roadmap["phases"][1]["status"], "merged")
        self.assertIn(roadmap["phases"][2]["status"], {"implemented", "merged"})

    def test_accepts_qualified_color_evidence(self):
        records = [{
            "case_id": f"case-{i}",
            "perceptual_color_score": 0.995,
            "normalized_delta_e": 0.010,
            "palette_omissions": 0,
            "palette_duplicates": 0,
            "transparency_mismatches": 0,
        } for i in range(20)]
        validate_color_evidence(records, 20)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "perceptual_color_score": 0.995,
            "normalized_delta_e": 0.010,
            "palette_omissions": 0,
            "palette_duplicates": 0,
            "transparency_mismatches": 0,
        }
        bad_sets = [
            [],
            [dict(base, perceptual_color_score=float("nan"))],
            [dict(base, normalized_delta_e=1.1)],
            [dict(base), dict(base)],
            [dict(base, palette_omissions=1)],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_color_evidence(records, 1)


if __name__ == "__main__":
    unittest.main()
