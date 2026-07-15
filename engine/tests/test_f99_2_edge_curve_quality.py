import math

MIN_CASES = 24


def validate(rows):
    assert isinstance(rows, list) and len(rows) >= MIN_CASES
    ids = [r.get("id") for r in rows]
    assert all(isinstance(x, str) and x for x in ids)
    assert len(ids) == len(set(ids))
    required = ("edge_fscore", "tangent_error", "curvature_error")
    for row in rows:
        for key in required:
            value = row.get(key)
            assert isinstance(value, (int, float)) and math.isfinite(value)
        assert 0.0 <= row["edge_fscore"] <= 1.0
        assert 0.0 <= row["tangent_error"] <= 1.0
        assert 0.0 <= row["curvature_error"] <= 1.0
        assert row.get("open_contours", 1) == 0
        assert row.get("self_intersections", 1) == 0
        assert row.get("cusp_regressions", 1) == 0
    fs = sorted(r["edge_fscore"] for r in rows)
    tang = sorted(r["tangent_error"] for r in rows)
    curv = sorted(r["curvature_error"] for r in rows)
    assert sum(fs) / len(fs) >= 0.990
    assert fs[max(0, math.ceil(0.05 * len(fs)) - 1)] >= 0.980
    assert tang[min(len(tang)-1, math.ceil(0.95 * len(tang)) - 1)] <= 0.020
    assert curv[min(len(curv)-1, math.ceil(0.95 * len(curv)) - 1)] <= 0.030


def good_rows():
    return [{"id": f"case-{i}", "edge_fscore": 0.995, "tangent_error": 0.01,
             "curvature_error": 0.02, "open_contours": 0,
             "self_intersections": 0, "cusp_regressions": 0}
            for i in range(MIN_CASES)]


def test_accepts_qualified_corpus():
    validate(good_rows())


def test_rejects_duplicate_or_non_finite_evidence():
    rows = good_rows(); rows[-1]["id"] = rows[0]["id"]
    try: validate(rows); assert False
    except AssertionError: pass
    rows = good_rows(); rows[0]["tangent_error"] = float("nan")
    try: validate(rows); assert False
    except AssertionError: pass


def test_rejects_threshold_and_topology_regression():
    rows = good_rows(); rows[0]["edge_fscore"] = 0.5
    try: validate(rows); assert False
    except AssertionError: pass
    rows = good_rows(); rows[0]["self_intersections"] = 1
    try: validate(rows); assert False
    except AssertionError: pass
