from pathlib import Path
import runpy

# Reuse the analyzer half of v1; it intentionally stops at the changed
# editability anchor after writing analyzer.py and before writing pipeline.py.
try:
    runpy.run_path('.github/issue153_selection_balance.py', run_name='__main__')
except SystemExit as exc:
    if 'editability guard' not in str(exc):
        raise


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f'missing v2 anchor: {label}')
    return text.replace(old, new, 1)

p = Path('engine/app/pipeline.py')
s = p.read_text(encoding='utf-8')
finite = '''def _finite(value: Any) -> float | None:\n    try:\n        number = float(value)\n    except (TypeError, ValueError):\n        return None\n    return number if math.isfinite(number) else None\n'''
extra = finite + '''\n\ndef _needs_component_pareto_guard(candidate: dict[str, Any]) -> bool:\n    if bool(candidate.get("hard_stop_gradient")):\n        return True\n    component = candidate.get("component_quality") or {}\n    if not component.get("applicable"):\n        return False\n    if not component.get("measured"):\n        return True\n    for name in ("source_cc_recall", "render_cc_precision", "min_true_cc_iou"):\n        measured = _finite(component.get(name))\n        if measured is None or measured < 1.0 - 1e-9:\n            return True\n    return False\n\n\ndef _annotate_selection_context(candidate: dict[str, Any] | None, analysis: dict[str, Any]) -> dict[str, Any] | None:\n    if candidate is not None:\n        candidate["hard_stop_gradient"] = bool(analysis.get("hard_stop_gradient"))\n    return candidate\n'''
if 'def _needs_component_pareto_guard(' not in s:
    s = replace_once(s, finite, extra, 'helper')
s = replace_once(
    s,
    '    if _is_selection_safe(legacy_chosen):\n        return legacy_chosen, legacy_raw, legacy_reason\n',
    '    if _is_selection_safe(legacy_chosen) and not _needs_component_pareto_guard(legacy_chosen):\n        return legacy_chosen, legacy_raw, legacy_reason\n',
    'select_best',
)
s = replace_once(
    s,
    '''def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:\n    """Picklable facade entrypoint so worker processes install the same policy."""\n    return _base_produce_and_score_job(args)\n''',
    '''def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:\n    """Picklable facade entrypoint with measured selection context attached."""\n    result, scored = _base_produce_and_score_job(args)\n    analysis = args[7] if len(args) > 7 and isinstance(args[7], dict) else {}\n    return result, _annotate_selection_context(scored, analysis)\n''',
    'produce_score',
)
old_edit = '''def _apply_editability_preference(\n    scored: list[dict[str, Any]], current_best: dict[str, Any]\n) -> tuple[dict[str, Any], str]:\n    pool = _selection_pool(scored)\n    if current_best not in pool:\n        current_best = max(pool, key=_core._fidelity_rank_key)\n    guarded_best, guarded_reason = _base_apply_editability_preference(pool, current_best)\n    if pool is scored:\n        return guarded_best, guarded_reason\n    return guarded_best, f"component_integrity_pareto_guard+{guarded_reason}"\n'''
new_edit = '''def _apply_editability_preference(\n    scored: list[dict[str, Any]], current_best: dict[str, Any]\n) -> tuple[dict[str, Any], str]:\n    legacy_best, legacy_reason = _base_apply_editability_preference(scored, current_best)\n    if _is_selection_safe(legacy_best) and not _needs_component_pareto_guard(legacy_best):\n        return legacy_best, legacy_reason\n    safe = [candidate for candidate in scored if _is_selection_safe(candidate)]\n    if not safe:\n        return legacy_best, legacy_reason\n    pool = _pareto_front(safe)\n    safe_seed = current_best if current_best in pool else max(pool, key=_core._fidelity_rank_key)\n    guarded_best, guarded_reason = _base_apply_editability_preference(pool, safe_seed)\n    return guarded_best, f"component_integrity_pareto_guard+{guarded_reason}"\n'''
s = replace_once(s, old_edit, new_edit, 'editability')
s = replace_once(
    s,
    '''    refined, info = _base_refine_best(\n        best, mode, analysis, original_path, preprocessed_path, job_dir, scored\n    )\n''',
    '''    refined, info = _base_refine_best(\n        best, mode, analysis, original_path, preprocessed_path, job_dir, scored\n    )\n    _annotate_selection_context(refined, analysis)\n''',
    'refine',
)
s = replace_once(
    s,
    '    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)\n    if (\n',
    '    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)\n    _annotate_selection_context(refined, analysis)\n    if (\n',
    'refit',
)
s = replace_once(
    s,
    '''    candidate, info = _base_apply_boundary_refit(\n        best, mode, analysis, original_path, job_dir, scored\n    )\n    if candidate is not best and _is_selection_safe(best):\n''',
    '''    candidate, info = _base_apply_boundary_refit(\n        best, mode, analysis, original_path, job_dir, scored\n    )\n    _annotate_selection_context(candidate, analysis)\n    if candidate is not best and _is_selection_safe(best):\n''',
    'boundary',
)
p.write_text(s, encoding='utf-8')

p = Path('engine/test_analyzer_contracts.py')
s = p.read_text(encoding='utf-8')
if 'detect_hard_stop_gradient_surface(Image.fromarray(ramp' not in s:
    s = replace_once(s, '    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True\n', '    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True\n    assert analyzer.detect_hard_stop_gradient_surface(Image.fromarray(ramp, "RGBA")) is True\n', 'analyzer positive')
if 'detect_hard_stop_gradient_surface(Image.fromarray(smooth_only' not in s:
    s = replace_once(s, '    assert analyzer.detect_gradient_like_surface(Image.fromarray(smooth_only, "RGBA")) is False\n', '    assert analyzer.detect_gradient_like_surface(Image.fromarray(smooth_only, "RGBA")) is False\n    assert analyzer.detect_hard_stop_gradient_surface(Image.fromarray(smooth_only, "RGBA")) is False\n', 'analyzer negative')
p.write_text(s, encoding='utf-8')

p = Path('engine/test_ai2_component_quality.py')
s = p.read_text(encoding='utf-8')
if 'test_safe_perfect_legacy_winner_is_preserved_from_pareto_churn' not in s:
    s += r'''


def _selection_candidate(name: str, *, total: float, fidelity: float, component: dict, hard_stop: bool = False) -> dict:
    return {"name": name, "total_score": total, "fidelity_score": fidelity, "rendered_ok": True,
            "selection_safe": True, "selection_disqualified": False, "hard_stop_gradient": hard_stop,
            "component_quality": component,
            "score_details": {"path_count": 10, "edge_f1": fidelity / 100.0,
                              "ssim": fidelity / 100.0, "mean_delta_e": 100.0 - fidelity,
                              "has_bitmap": False}}


def test_safe_perfect_legacy_winner_is_preserved_from_pareto_churn() -> None:
    perfect = {"applicable": True, "measured": True, "status": "pass", "source_cc_recall": 1.0, "render_cc_precision": 1.0, "min_true_cc_iou": 1.0}
    legacy = _selection_candidate("legacy", total=99.0, fidelity=98.0, component=perfect)
    alternative = _selection_candidate("alternative", total=98.0, fidelity=99.0, component=perfect)
    chosen, _raw, reason = select_best([alternative, legacy], "single_color")
    assert chosen["name"] == "legacy"
    assert "component_integrity_pareto_guard" not in reason


def test_safe_but_component_imperfect_legacy_allows_pareto_guard() -> None:
    imperfect = {"applicable": True, "measured": True, "status": "pass", "source_cc_recall": 1.0, "render_cc_precision": 1.0, "min_true_cc_iou": 0.98}
    perfect = {"applicable": True, "measured": True, "status": "pass", "source_cc_recall": 1.0, "render_cc_precision": 1.0, "min_true_cc_iou": 1.0}
    legacy = _selection_candidate("legacy", total=99.0, fidelity=98.0, component=imperfect)
    alternative = _selection_candidate("alternative", total=98.0, fidelity=99.0, component=perfect)
    chosen, _raw, reason = select_best([alternative, legacy], "single_color")
    assert chosen["name"] == "alternative"
    assert reason.startswith("component_integrity_pareto_guard+")


def test_hard_stop_gradient_context_allows_pareto_but_plain_gradient_stays_legacy() -> None:
    na = {"applicable": False, "measured": False, "status": "not_applicable", "reason": "gradient_input"}
    hard_legacy = _selection_candidate("hard_legacy", total=99.0, fidelity=98.0, component=na, hard_stop=True)
    hard_alt = _selection_candidate("hard_alt", total=98.0, fidelity=99.0, component=na, hard_stop=True)
    hard_chosen, _raw, hard_reason = select_best([hard_alt, hard_legacy], "logo_color")
    assert hard_chosen["name"] == "hard_alt"
    assert hard_reason.startswith("component_integrity_pareto_guard+")
    smooth_legacy = _selection_candidate("smooth_legacy", total=99.0, fidelity=98.0, component=na)
    smooth_alt = _selection_candidate("smooth_alt", total=98.0, fidelity=99.0, component=na)
    smooth_chosen, _raw, smooth_reason = select_best([smooth_alt, smooth_legacy], "logo_color")
    assert smooth_chosen["name"] == "smooth_legacy"
    assert "component_integrity_pareto_guard" not in smooth_reason
'''
p.write_text(s, encoding='utf-8')
