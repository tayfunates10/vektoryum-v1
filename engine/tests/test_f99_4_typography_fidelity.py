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


def validate_typography_evidence(records, minimum_cases):
    if not isinstance(minimum_cases, int) or isinstance(minimum_cases, bool) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if not isinstance(records, list) or len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")
    ids = []
    scores = []
    stroke_errors = []
    defect_total = 0
    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        ids.append(case_id)
        for key in ("glyph_fidelity_score", "normalized_stroke_width_error"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AssertionError(f"non-finite {key}")
            if not 0 <= value <= 1:
                raise AssertionError(f"out-of-range {key}")
        scores.append(float(record["glyph_fidelity_score"]))
        stroke_errors.append(float(record["normalized_stroke_width_error"]))
        for key in ("missing_glyphs", "merged_glyphs", "broken_counters", "lost_fine_details"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(f"invalid {key}")
            defect_total += value
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(scores) < 0.990:
        raise AssertionError("mean glyph fidelity regression")
    if percentile(scores, 0.05) < 0.980:
        raise AssertionError("p05 glyph fidelity regression")
    if percentile(stroke_errors, 0.95) > 0.020:
        raise AssertionError("p95 stroke-width regression")
    if defect_total != 0:
        raise AssertionError("typography/detail defect")


class F99TypographyFidelityContractTests(unittest.TestCase):
    def test_roadmap_is_finite_and_ordered(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])

        statuses = [p["status"] for p in phases]
        self.assertEqual(statuses[:4], ["merged"] * 4)
        seen_implemented = False
        seen_pending = False
        for status in statuses:
            self.assertIn(status, {"merged", "implemented", "pending"})
            if status == "merged":
                self.assertFalse(seen_implemented)
                self.assertFalse(seen_pending)
            elif status == "implemented":
                self.assertFalse(seen_implemented)
                self.assertFalse(seen_pending)
                seen_implemented = True
            else:
                seen_pending = True

        self.assertTrue((ROOT / phases[3]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[3]["acceptance"]), 5)

    def test_accepts_qualified_typography_evidence(self):
        records = [{
            "case_id": f"case-{i}",
            "glyph_fidelity_score": 0.995,
            "normalized_stroke_width_error": 0.010,
            "missing_glyphs": 0,
            "merged_glyphs": 0,
            "broken_counters": 0,
            "lost_fine_details": 0,
        } for i in range(20)]
        validate_typography_evidence(records, 20)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "glyph_fidelity_score": 0.995,
            "normalized_stroke_width_error": 0.010,
            "missing_glyphs": 0,
            "merged_glyphs": 0,
            "broken_counters": 0,
            "lost_fine_details": 0,
        }
        bad_sets = [
            [],
            [dict(base, glyph_fidelity_score=float("nan"))],
            [dict(base, normalized_stroke_width_error=1.1)],
            [dict(base), dict(base)],
            [dict(base, broken_counters=1)],
            [dict(base, lost_fine_details=-1)],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_typography_evidence(records, 1)


if __name__ == "__main__":
    unittest.main()
