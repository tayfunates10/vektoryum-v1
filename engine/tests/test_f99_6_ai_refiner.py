import json
import math
import re
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "fidelity99_roadmap.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EDIT_CLASSES = {"geometry", "color", "topology", "artifact_cleanup"}


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


def validate_refiner_evidence(records, minimum_cases=24, max_edits=64):
    if not isinstance(minimum_cases, int) or isinstance(minimum_cases, bool) or minimum_cases < 1:
        raise AssertionError("invalid minimum_cases")
    if not isinstance(max_edits, int) or isinstance(max_edits, bool) or max_edits < 0:
        raise AssertionError("invalid max_edits")
    if not isinstance(records, list) or len(records) < minimum_cases:
        raise AssertionError("shrinking corpus")

    ids = []
    gains = []
    post_scores = []
    defect_total = 0
    digest_by_identity = {}

    for record in records:
        if not isinstance(record, dict):
            raise AssertionError("invalid record")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AssertionError("missing case_id")
        ids.append(case_id)

        for key in ("input_sha256", "model_sha256", "output_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise AssertionError(f"invalid {key}")

        for key in ("pre_fidelity", "post_fidelity"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AssertionError(f"non-finite {key}")
            if not 0 <= value <= 1:
                raise AssertionError(f"out-of-range {key}")

        pre = float(record["pre_fidelity"])
        post = float(record["post_fidelity"])
        if post < pre:
            raise AssertionError("fidelity regression")
        gains.append(post - pre)
        post_scores.append(post)

        edit_count = record.get("edit_count")
        if not isinstance(edit_count, int) or isinstance(edit_count, bool) or not 0 <= edit_count <= max_edits:
            raise AssertionError("unbounded edit count")

        edit_classes = record.get("edit_classes")
        if not isinstance(edit_classes, list) or any(not isinstance(v, str) for v in edit_classes):
            raise AssertionError("invalid edit classes")
        if not set(edit_classes).issubset(ALLOWED_EDIT_CLASSES):
            raise AssertionError("unexpected edit class")

        for key in ("topology_damage", "provenance_gaps", "digest_drift", "source_mutations"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssertionError(f"invalid {key}")
            defect_total += value

        identity = (record["input_sha256"], record["model_sha256"])
        previous = digest_by_identity.setdefault(identity, record["output_sha256"])
        if previous != record["output_sha256"]:
            raise AssertionError("nondeterministic output digest")

    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate case_id")
    if statistics.fmean(gains) < 0:
        raise AssertionError("negative mean fidelity gain")
    if percentile(post_scores, 0.05) < 0.990:
        raise AssertionError("p05 post-refinement fidelity regression")
    if defect_total != 0:
        raise AssertionError("refiner defect")


class F996AIRefinerContractTests(unittest.TestCase):
    def test_roadmap_is_finite_and_ordered(self):
        roadmap = json.loads(ROADMAP.read_text(encoding="utf-8"))
        phases = roadmap["phases"]
        self.assertEqual(roadmap["phase_count"], 8)
        self.assertEqual([p["id"] for p in phases], [f"F99-{i}" for i in range(1, 9)])
        self.assertEqual([p["status"] for p in phases[:5]], ["merged"] * 5)
        self.assertIn(phases[5]["status"], {"implemented", "merged"})
        self.assertEqual([p["status"] for p in phases[6:]], ["pending"] * 2)
        self.assertTrue((ROOT / phases[5]["evidence"]).is_file())
        self.assertGreaterEqual(len(phases[5]["acceptance"]), 5)

    def test_accepts_qualified_refiner_evidence(self):
        records = []
        for index in range(24):
            records.append({
                "case_id": f"case-{index}",
                "input_sha256": f"{index + 1:064x}",
                "model_sha256": "a" * 64,
                "output_sha256": f"{index + 101:064x}",
                "pre_fidelity": 0.991,
                "post_fidelity": 0.995,
                "edit_count": 4,
                "edit_classes": ["geometry", "artifact_cleanup"],
                "topology_damage": 0,
                "provenance_gaps": 0,
                "digest_drift": 0,
                "source_mutations": 0,
            })
        validate_refiner_evidence(records)

    def test_fails_closed_on_invalid_evidence(self):
        base = {
            "case_id": "case-0",
            "input_sha256": "1" * 64,
            "model_sha256": "a" * 64,
            "output_sha256": "b" * 64,
            "pre_fidelity": 0.991,
            "post_fidelity": 0.995,
            "edit_count": 2,
            "edit_classes": ["geometry"],
            "topology_damage": 0,
            "provenance_gaps": 0,
            "digest_drift": 0,
            "source_mutations": 0,
        }
        bad_sets = [
            [],
            [dict(base, post_fidelity=0.990)],
            [dict(base, post_fidelity=float("nan"))],
            [dict(base, edit_count=65)],
            [dict(base, edit_classes=["rewrite_everything"])],
            [dict(base), dict(base)],
            [dict(base, topology_damage=1)],
            [dict(base, input_sha256="bad")],
        ]
        for records in bad_sets:
            with self.subTest(records=records):
                with self.assertRaises(AssertionError):
                    validate_refiner_evidence(records, minimum_cases=1)

    def test_same_input_and_model_require_same_output_digest(self):
        first = {
            "case_id": "a",
            "input_sha256": "1" * 64,
            "model_sha256": "2" * 64,
            "output_sha256": "3" * 64,
            "pre_fidelity": 0.991,
            "post_fidelity": 0.995,
            "edit_count": 1,
            "edit_classes": ["geometry"],
            "topology_damage": 0,
            "provenance_gaps": 0,
            "digest_drift": 0,
            "source_mutations": 0,
        }
        second = dict(first, case_id="b", output_sha256="4" * 64)
        with self.assertRaises(AssertionError):
            validate_refiner_evidence([first, second], minimum_cases=2)


if __name__ == "__main__":
    unittest.main()
