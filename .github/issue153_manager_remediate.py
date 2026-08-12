from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


# Exact Final: restore the required-base bounded alpha-parent applicability.
p = Path("engine/app/alpha_parent_selection.py")
s = p.read_text()
if "_MAX_ALPHA_LEVELS = 4" not in s:
    s = replace_once(s, "_MAX_TRIALS = 4\n", "_MAX_ALPHA_LEVELS = 4\n_MAX_TRIALS = 4\n", "alpha max trials")
s, count = re.subn(
    r"(?m)^    if len\(levels\) <= 1:\s*$",
    "    if len(levels) <= 1 or len(levels) > _MAX_ALPHA_LEVELS:",
    s,
    count=1,
)
if count != 1:
    raise SystemExit(f"alpha applicability replacement count={count}")
p.write_text(s)

# Journal: keep downstream gradient render measurement, but restore base fail-closed
# scope for alpha-only downstream transforms.
p = Path("engine/app/transform_journal.py")
s = p.read_text()
s = replace_once(
    s,
    '        measure_alpha = "alpha_fidelity" in self.required_metrics\n',
    '        measure_alpha = capture_render and "alpha_fidelity" in self.required_metrics\n',
    "journal alpha measurement",
)
p.write_text(s)

# Analyzer: only the NEW foreground-local fallback is restricted to opaque inputs.
# The legacy global detector above remains authoritative for transparent gradients.
p = Path("engine/app/analyzer.py")
s = p.read_text()
fn = s.index("def detect_gradient_like_surface")
legacy = s.index("    if unique_ratio > 0.095 and smooth_area_ratio > 0.68:", fn)
insertion = s.index("    h, w = arr.shape[:2]", legacy)
guard = (
    "    # Transparent sources already have a dedicated alpha reconstruction path.\n"
    "    # Do not let the new local fallback alone expand their candidate graph.\n"
    "    if bool(np.any(alpha < 255)):\n"
    "        return False\n\n"
)
if guard not in s:
    s = s[:insertion] + guard + s[insertion:]
p.write_text(s)

# Journal regression contracts: alpha-only downstream mutation remains fail-closed;
# gradient-required downstream mutation still measures real render evidence.
p = Path("engine/test_transform_journal.py")
s = p.read_text()
s = replace_once(
    s,
    "def test_alpha_required_measurement_runs_on_downstream_stage(",
    "def test_alpha_only_measurement_remains_scoped_to_source_dimension_restore(",
    "alpha downstream test name",
)
pattern = re.compile(
    r"(def test_alpha_only_measurement_remains_scoped_to_source_dimension_restore\([\s\S]*?"
    r"accepted, stage = journal\.consider_candidate\(\"boundary_refit\", parent, candidate\)\n)"
    r"[\s\S]*?(?=\n\ndef test_alpha_restore_reuses_single_rgba_render_per_artifact)"
)
replacement = r'''\1
    assert accepted == parent
    assert stage["status"] == "rolled_back"
    assert "required_metric_unmeasured" in stage["reason_codes"]
    assert "alpha_stage_metrics_incomplete" in stage["reason_codes"]
    assert stage["required_unmeasured"] == ["alpha_fidelity"]
    assert stage["alpha_comparison"] is None
'''
s, count = pattern.subn(replacement, s, count=1)
if count != 1:
    raise SystemExit(f"alpha journal test replacement count={count}")

pattern = re.compile(
    r"def test_required_alpha_metric_is_measured_not_faked_on_downstream_stage\(tmp_path: Path\) -> None:"
    r"[\s\S]*?(?=\n\ndef test_winner_selection_filters_unsafe_when_safe_exists_and_preserves_no_safe_fallback)"
)
replacement = '''def test_required_alpha_metric_is_not_faked_on_downstream_stage(tmp_path: Path) -> None:
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
s, count = pattern.subn(replacement, s, count=1)
if count != 1:
    raise SystemExit(f"extra alpha regression replacement count={count}")
p.write_text(s)
