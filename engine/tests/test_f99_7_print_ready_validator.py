import json
import math
import re
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "fidelity99_roadmap.json"
REQUIRED_FORMATS = ("svg", "pdf", "eps", "dxf")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_print_ready_evidence(records, minimum_cases=24):
    if not isinstance(minimum_cases, int) or isinstance(minimum_cases, bool) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if not isinstance(records, list) or len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")

    case_ids = []
    consistency_scores = []
    defect_total = 0

    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        case_ids.append(case_id)

        formats = record.get("formats")
        if not isinstance(formats, list) or tuple(item.get("format") for item in formats if isinstance(item, dict)) != REQUIRED_FORMATS:
            raise AssertionError("invalid formats")

        for item in formats:
            if not isinstance(item, dict):
                raise AssertionError("invalid format record")
            digest = item.get("artifact_sha256")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise AssertionError("invalid artifact digest")
            for key in ("parsed", "structure_valid"):
                if item.get(key) is not True:
                    raise AssertionError(f"failed {key}")
            score = item.get("metadata_consistency")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise AssertionError("non-finite metadata consistency")
            if not 0 <= score <= 1:
                raise AssertionError("out-of-range metadata consistency")
            consistency_scores.append(float(score))

        for key in (
            "open_paths",
            "self_intersections",
            "invalid_fill_rules",
            "missing_color_profiles",
            "unsupported_features",
        ):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(f"invalid {key}")
            defect_total += value

    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(consistency_scores) < 0.990:
        raise AssertionError("metadata consistency regression")
    if defect_total != 0:
        raise AssertionError("print-ready defect")


class F997PrintReadyValidatorContractTests(unittest.TestCase):
    def test_roadmap_is_finite_and_ordered(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])
        self.assertEqual([p["status"] for p in phases[:6]], ["merged"] * 6)
        self.assertIn(phases[6]["status"], {"implemented", "merged"})
        self.assertEqual(phases[7]["status"], "pending")
        self.assertTrue((ROOT / phases[6]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[6]["acceptance"]), 5)

    def test_accepts_qualified_print_ready_evidence(self):
        records = []
        for index in range(24):
            records.append({
                "case_id": f"case-{index}",
                "formats": [
                    {
                        "format": fmt,
                        "artifact_sha256": f"{index * 4 + offset + 1:064x}",
                        "parsed": True,
                        "structure_valid": True,
                        "metadata_consistency": 0.995,
                    }
                    for offset, fmt in enumerate(REQUIRED_FORMATS)
                ],
                "open_paths": 0,
                "self_intersections": 0,
                "invalid_fill_rules": 0,
                "missing_color_profiles": 0,
                "unsupported_features": 0,
            })
        validate_print_ready_evidence(records)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "formats": [
                {
                    "format": fmt,
                    "artifact_sha256": f"{offset + 1:064x}",
                    "parsed": True,
                    "structure_valid": True,
                    "metadata_consistency": 0.995,
                }
                for offset, fmt in enumerate(REQUIRED_FORMATS)
            ],
            "open_paths": 0,
            "self_intersections": 0,
            "invalid_fill_rules": 0,
            "missing_color_profiles": 0,
            "unsupported_features": 0,
        }
        bad_sets = [
            [],
            [dict(base, formats=base["formats"][:-1])],
            [dict(base, formats=[dict(base["formats"][0], artifact_sha256="bad"), *base["formats"][1:]])],
            [dict(base, formats=[dict(base["formats"][0], parsed=False), *base["formats"][1:]])],
            [dict(base, formats=[dict(base["formats"][0], metadata_consistency=float("nan")), *base["formats"][1:]])],
            [dict(base), dict(base)],
            [dict(base, self_intersections=1)],
            [dict(base, unsupported_features=-1)],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_print_ready_evidence(records, minimum_cases=1)


if __name__ == "__main__":
    unittest.main()
