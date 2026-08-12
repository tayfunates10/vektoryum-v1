from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


# 1) Analyzer: the foreground-local fallback must describe the Issue-153 mixed
# paint model (continuous ramp + real hard semantic boundary), not every smooth
# field. This preserves the hard-stop target and prevents pure smooth benchmark
# gradients from being rerouted into the expensive specialized candidate graph.
p = Path("engine/app/analyzer.py")
s = p.read_text()
old = '''    pair_count = 0
    continuous_count = 0
    for first, second, mask in (
        (arr[:, :-1], arr[:, 1:], interior[:, :-1] & interior[:, 1:]),
        (arr[:-1, :], arr[1:, :], interior[:-1, :] & interior[1:, :]),
    ):
        distances = np.linalg.norm(first - second, axis=2)
        pair_count += int(np.count_nonzero(mask))
        continuous_count += int(np.count_nonzero(mask & (distances >= 1.0) & (distances <= 32.0)))
    transition_ratio = float(continuous_count) / float(max(1, pair_count))
    foreground_bins = np.unique((arr[interior].astype(np.uint8) // 8), axis=0).shape[0]
    return bool(
        foreground_bins >= 12
        and transition_ratio >= 0.08
        and smooth_area_ratio >= 0.90
    )
'''
new = '''    pair_count = 0
    continuous_count = 0
    hard_boundary_count = 0
    for first, second, mask in (
        (arr[:, :-1], arr[:, 1:], interior[:, :-1] & interior[:, 1:]),
        (arr[:-1, :], arr[1:, :], interior[:-1, :] & interior[1:, :]),
    ):
        distances = np.linalg.norm(first - second, axis=2)
        pair_count += int(np.count_nonzero(mask))
        continuous_count += int(np.count_nonzero(mask & (distances >= 1.0) & (distances <= 32.0)))
        hard_boundary_count += int(np.count_nonzero(mask & (distances >= 80.0)))
    transition_ratio = float(continuous_count) / float(max(1, pair_count))
    foreground_bins = np.unique((arr[interior].astype(np.uint8) // 8), axis=0).shape[0]
    hard_boundary_min = max(4, int(round(0.05 * float(np.sqrt(interior_count)))))
    return bool(
        foreground_bins >= 12
        and transition_ratio >= 0.08
        and hard_boundary_count >= hard_boundary_min
        and smooth_area_ratio >= 0.90
    )
'''
s = replace_once(s, old, new, "analyzer local fallback")
p.write_text(s)

# 2) Journal: preserve downstream gradient render evidence, but alpha-only
# mutations remain fail-closed outside the source-dimension measurement boundary.
p = Path("engine/app/transform_journal.py")
s = p.read_text()
old = '''        capture_render = (
            self._measurement_stage_id == "restore_source_dimensions"
            or "gradient_fidelity" in self.required_metrics
        )
        measure_alpha = "alpha_fidelity" in self.required_metrics
'''
new = '''        is_restore_stage = self._measurement_stage_id == "restore_source_dimensions"
        capture_render = is_restore_stage or "gradient_fidelity" in self.required_metrics
        measure_alpha = is_restore_stage and "alpha_fidelity" in self.required_metrics
'''
s = replace_once(s, old, new, "journal measurement scope")
p.write_text(s)

# 3) Exact Final alpha parent scope must be the proven required-base contract.
p = Path("engine/app/alpha_parent_selection.py")
s = p.read_text()
if "_MAX_ALPHA_LEVELS = 4" not in s:
    raise SystemExit("bounded alpha parent constant missing")
if "if len(levels) <= 1 or len(levels) > _MAX_ALPHA_LEVELS:" not in s:
    raise SystemExit("bounded alpha parent applicability missing")

# 4) Analyzer regression: target hard-stop true, palette steps/thin AA false,
# and a pure smooth ramp must not be promoted by the new local fallback.
p = Path("engine/test_analyzer_contracts.py")
s = p.read_text()
needle = '''    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True
    assert analyzer.detect_gradient_like_surface(Image.fromarray(stepped, "RGBA")) is False

    thin = np.full((256, 256, 4), 255, dtype=np.uint8)
'''
replacement = '''    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True
    assert analyzer.detect_gradient_like_surface(Image.fromarray(stepped, "RGBA")) is False

    smooth_only = np.full((256, 256, 4), 255, dtype=np.uint8)
    for x in range(24, 232):
        t = (x - 24) / 207.0
        smooth_only[48:208, x, :3] = np.rint(
            np.asarray((225, 40, 35)) * (1.0 - t) + np.asarray((35, 92, 220)) * t
        ).astype(np.uint8)
    assert analyzer.detect_gradient_like_surface(Image.fromarray(smooth_only, "RGBA")) is False

    thin = np.full((256, 256, 4), 255, dtype=np.uint8)
'''
s = replace_once(s, needle, replacement, "analyzer smooth-only regression")
p.write_text(s)

# 5) Journal regression tests: alpha-only downstream stages do not manufacture
# evidence; the existing gradient downstream tests stay positive.
p = Path("engine/test_transform_journal.py")
s = p.read_text()
s = replace_once(
    s,
    "def test_alpha_required_measurement_runs_on_downstream_stage(\n",
    "def test_alpha_only_measurement_remains_scoped_to_source_dimension_restore(\n",
    "alpha downstream test name",
)
old = '''    assert accepted == candidate
    assert stage["status"] == "accepted"
    assert "required_metric_unmeasured" not in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" not in stage["reason_codes"]
    assert stage["required_unmeasured"] == []
    assert stage["alpha_comparison"] is not None
    assert stage["alpha_comparison"]["alpha_iou"] == pytest.approx(1.0)
'''
new = '''    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]
    assert stage["required_unmeasured"] == ["alpha_fidelity"]
    assert stage["alpha_comparison"] is None
'''
s = replace_once(s, old, new, "alpha downstream assertions")
old = '''def test_required_alpha_metric_is_measured_not_faked_on_downstream_stage(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal
    parent = tmp_path / "alpha_parent.svg"
    candidate = tmp_path / "alpha_candidate.svg"
    alpha_svg = _svg('<rect x="24" y="24" width="80" height="80" fill="#e3000b" fill-opacity="0.5"/>')
    parent.write_bytes(alpha_svg)
    candidate.write_bytes(alpha_svg.replace(b'</svg>', b'<metadata>same-alpha</metadata></svg>'))
    journal = TransformJournal(parent, _square_source(), required_metrics={"alpha_fidelity"})
    accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)
    assert accepted == candidate
    assert stage["alpha_comparison"] is not None
    assert "alpha_stage_metrics_incomplete" not in stage["reason_codes"]
'''
new = '''def test_required_alpha_metric_is_not_faked_on_downstream_stage(tmp_path: Path) -> None:
    from app.transform_journal import TransformJournal
    parent = tmp_path / "alpha_parent.svg"
    candidate = tmp_path / "alpha_candidate.svg"
    alpha_svg = _svg('<rect x="24" y="24" width="80" height="80" fill="#e3000b" fill-opacity="0.5"/>')
    parent.write_bytes(alpha_svg)
    candidate.write_bytes(alpha_svg.replace(b'</svg>', b'<metadata>same-alpha</metadata></svg>'))
    journal = TransformJournal(parent, _square_source(), required_metrics={"alpha_fidelity"})
    accepted, stage = journal.consider_candidate("boundary_refit", parent, candidate)
    assert accepted == parent
    assert stage["alpha_comparison"] is None
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]
'''
s = replace_once(s, old, new, "extra alpha downstream regression")
p.write_text(s)
