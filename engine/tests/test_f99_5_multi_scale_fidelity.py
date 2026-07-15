import json
import math
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "fidelity99_roadmap.json"
EXPECTED_SCALES = ("0.25x", "0.5x", "1x", "2x", "4x")


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


def validate_multi_scale_evidence(records, minimum_cases):
    if not isinstance(minimum_cases, int) or isinstance(minimum_cases, bool) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if not isinstance(records, list) or len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")
    case_ids = []
    scores = []
    drifts = []
    defect_total = 0
    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        case_ids.append(case_id)
        levels = record.get("levels")
        if not isinstance(levels, list) or tuple(level.get("scale") for level in levels if isinstance(level, dict)) != EXPECTED_SCALES:
            raise AssertionError("invalid scale levels")
        for level in levels:
            if not isinstance(level, dict):
                raise AssertionError("invalid scale record")
            for key in ("fidelity_score", "normalized_scale_drift"):
                value = level.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise AssertionError(f"non-finite {key}")
                if not 0 <= value <= 1:
                    raise AssertionError(f"out-of-range {key}")
            scores.append(float(level["fidelity_score"]))
            drifts.append(float(level["normalized_scale_drift"]))
        for key in ("missing_levels", "scale_order_regressions", "topology_regressions", "detail_loss_regressions"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(f"invalid {key}")
            defect_total += value
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(scores) < 0.990:
        raise AssertionError("mean cross-scale fidelity regression")
    if percentile(scores, 0.05) < 0.980:
        raise AssertionError("p05 cross-scale fidelity regression")
    if percentile(drifts, 0.95) > 0.020:
        raise AssertionError("p95 scale drift regression")
    if defect_total != 0:
        raise AssertionError("multi-scale defect")


class F995MultiScaleFidelityContractTests(unittest.TestCase):
    def test_roadmap_is_finite_and_ordered(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])
        self.assertEqual([p["status"] for p in phases[:4]], ["merged"] * 4)
        self.assertIn(phases[4]["status"], {"implemented", "merged"})
        self.assertEqual([p["status"] for p in phases[5:]], ["pending"] * 3)
        self.assertTrue((ROOT / phases[4]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[4]["acceptance"]), 5)

    def test_accepts_qualified_multi_scale_evidence(self):
        records = []
        for index in range(24):
            records.append({
                "case_id": f"case-{index}",
                "levels": [{"scale": scale, "fidelity_score": 0.995, "normalized_scale_drift": 0.010} for scale in EXPECTED_SCALES],
                "missing_levels": 0,
                "scale_order_regressions": 0,
                "topology_regressions": 0,
                "detail_loss_regressions": 0,
            })
        validate_multi_scale_evidence(records, 24)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "levels": [{"scale": scale, "fidelity_score": 0.995, "normalized_scale_drift": 0.010} for scale in EXPECTED_SCALES],
            "missing_levels": 0,
            "scale_order_regressions": 0,
            "topology_regressions": 0,
            "detail_loss_regressions": 0,
        }
        bad_sets = [
            [],
            [dict(base, levels=base["levels"][:-1])],
            [dict(base, levels=[dict(base["levels"][0], fidelity_score=float("nan")), *base["levels"][1:]])],
            [dict(base), dict(base)],
            [dict(base, topology_regressions=1)],
            [dict(base, detail_loss_regressions=-1)],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_multi_scale_evidence(records, 1)


if __name__ == "__main__":
    unittest.main()
