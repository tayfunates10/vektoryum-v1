from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


# Analyzer: expose the specific hard-stop mixed-paint signal. A plain smooth
# gradient remains a gradient, but must not opt into the Issue-153 topology
# winner override.
p = Path("engine/app/analyzer.py")
s = p.read_text(encoding="utf-8")
helper = r'''

def detect_hard_stop_gradient_surface(image: Image.Image) -> bool:
    """Opaque continuous foreground paint with a measured hard semantic stop."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    if bool(np.any(alpha < 255)):
        return False
    arr = rgba[:, :, :3].astype(np.float32)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    edge_mask = cv2.Canny(gray, 70, 160) > 0
    smooth_area_ratio = float(np.mean(~edge_mask))
    h, w = arr.shape[:2]
    ph, pw = max(2, round(h * 0.04)), max(2, round(w * 0.04))
    corners = np.concatenate((
        arr[:ph, :pw].reshape(-1, 3), arr[:ph, -pw:].reshape(-1, 3),
        arr[-ph:, :pw].reshape(-1, 3), arr[-ph:, -pw:].reshape(-1, 3),
    ))
    background = np.median(corners, axis=0)
    foreground = (np.linalg.norm(arr - background[None, None, :], axis=2) > 40.0) & (alpha > 40)
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 20:
        return False
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 5)
    interior = foreground & (distance >= 3.5)
    interior_count = int(np.count_nonzero(interior))
    if interior_count < 20 or interior_count < 0.10 * foreground_count:
        return False
    pair_count = 0
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
if "def detect_hard_stop_gradient_surface(" not in s:
    s = replace_once(s, "\n\ndef score_image_quality(\n", helper + "\n\ndef score_image_quality(\n", "analyzer helper")
if "hard_stop_gradient = detect_hard_stop_gradient_surface(image)" not in s:
    s = replace_once(
        s,
        '    has_gradient = detect_gradient_like_surface(image)\n\n    estimated_color_count = color_data["estimated_color_count"]\n',
        '    has_gradient = detect_gradient_like_surface(image)\n    hard_stop_gradient = detect_hard_stop_gradient_surface(image)\n\n    estimated_color_count = color_data["estimated_color_count"]\n',
        "analyzer assignment",
    )
if '        "hard_stop_gradient": hard_stop_gradient,\n' not in s:
    s = replace_once(
        s,
        '        "has_gradient": has_gradient,\n        "tonal_gradient_foreground": tonal_gradient_foreground,\n',
        '        "has_gradient": has_gradient,\n        "hard_stop_gradient": hard_stop_gradient,\n        "tonal_gradient_foreground": tonal_gradient_foreground,\n',
        "analyzer output",
    )
s = s.replace(
    '    if bwr_low_color_signature:\n        has_gradient = False\n        tonal_gradient_foreground = False\n',
    '    if bwr_low_color_signature:\n        has_gradient = False\n        hard_stop_gradient = False\n        tonal_gradient_foreground = False\n',
    1,
)
s = s.replace(
    '    if detected_type == "geometric_logo":\n        has_gradient = False\n        tonal_gradient_foreground = False\n        likely_color_logo = False\n',
    '    if detected_type == "geometric_logo":\n        has_gradient = False\n        hard_stop_gradient = False\n        tonal_gradient_foreground = False\n        likely_color_logo = False\n',
    1,
)
p.write_text(s, encoding="utf-8")

# Pipeline: preserve a safe legacy winner only when its measured component
# topology is non-applicable or exact. A measured component imperfection or the
# explicit hard-stop-gradient context activates the existing safe Pareto guard.
p = Path("engine/app/pipeline.py")
s = p.read_text(encoding="utf-8")
finite = '''def _finite(value: Any) -> float | None:\n    try:\n        number = float(value)\n    except (TypeError, ValueError):\n        return None\n    return number if math.isfinite(number) else None\n'''
extra = finite + '''\n\ndef _needs_component_pareto_guard(candidate: dict[str, Any]) -> bool:\n    if bool(candidate.get("hard_stop_gradient")):\n        return True\n    component = candidate.get("component_quality") or {}\n    if not component.get("applicable"):\n        return False\n    if not component.get("measured"):\n        return True\n    for name in ("source_cc_recall", "render_cc_precision", "min_true_cc_iou"):\n        measured = _finite(component.get(name))\n        if measured is None or measured < 1.0 - 1e-9:\n            return True\n    return False\n\n\ndef _annotate_selection_context(candidate: dict[str, Any] | None, analysis: dict[str, Any]) -> dict[str, Any] | None:\n    if candidate is not None:\n        candidate["hard_stop_gradient"] = bool(analysis.get("hard_stop_gradient"))\n    return candidate\n'''
if "def _needs_component_pareto_guard(" not in s:
    s = replace_once(s, finite, extra, "pipeline helper")
s = replace_once(
    s,
    '    if _is_selection_safe(legacy_chosen):\n        return legacy_chosen, legacy_raw, legacy_reason\n',
    '    if _is_selection_safe(legacy_chosen) and not _needs_component_pareto_guard(legacy_chosen):\n        return legacy_chosen, legacy_raw, legacy_reason\n',
    "select guard",
)
s = replace_once(
    s,
    '''def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:\n    """Picklable facade entrypoint so worker processes install the same policy."""\n    return _base_produce_and_score_job(args)\n''',
    '''def _produce_and_score_job(args: tuple) -> tuple[dict[str, Any], dict[str, Any] | None]:\n    """Picklable facade entrypoint with measured selection context attached."""\n    result, scored = _base_produce_and_score_job(args)\n    analysis = args[7] if len(args) > 7 and isinstance(args[7], dict) else {}\n    return result, _annotate_selection_context(scored, analysis)\n''',
    "job annotation",
)
s = replace_once(
    s,
    '    if _is_selection_safe(legacy_best):\n        return legacy_best, legacy_reason\n',
    '    if _is_selection_safe(legacy_best) and not _needs_component_pareto_guard(legacy_best):\n        return legacy_best, legacy_reason\n',
    "editability guard",
)
s = replace_once(
    s,
    '''    refined, info = _base_refine_best(\n        best, mode, analysis, original_path, preprocessed_path, job_dir, scored\n    )\n''',
    '''    refined, info = _base_refine_best(\n        best, mode, analysis, original_path, preprocessed_path, job_dir, scored\n    )\n    _annotate_selection_context(refined, analysis)\n''',
    "refine annotation",
)
s = replace_once(
    s,
    '    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)\n    if (\n',
    '    refined = _base_refit_one(cand, mode, analysis, original_path, job_dir)\n    _annotate_selection_context(refined, analysis)\n    if (\n',
    "refit annotation",
)
s = replace_once(
    s,
    '''    candidate, info = _base_apply_boundary_refit(\n        best, mode, analysis, original_path, job_dir, scored\n    )\n    if candidate is not best and _is_selection_safe(best):\n''',
    '''    candidate, info = _base_apply_boundary_refit(\n        best, mode, analysis, original_path, job_dir, scored\n    )\n    _annotate_selection_context(candidate, analysis)\n    if candidate is not best and _is_selection_safe(best):\n''',
    "boundary annotation",
)
p.write_text(s, encoding="utf-8")

# Contracts.
p = Path("engine/test_analyzer_contracts.py")
s = p.read_text(encoding="utf-8")
if "detect_hard_stop_gradient_surface(Image.fromarray(ramp" not in s:
    s = replace_once(
        s,
        '    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True\n',
        '    assert analyzer.detect_gradient_like_surface(Image.fromarray(ramp, "RGBA")) is True\n    assert analyzer.detect_hard_stop_gradient_surface(Image.fromarray(ramp, "RGBA")) is True\n',
        "hard stop positive",
    )
if "detect_hard_stop_gradient_surface(Image.fromarray(smooth_only" not in s:
    s = replace_once(
        s,
        '    assert analyzer.detect_gradient_like_surface(Image.fromarray(smooth_only, "RGBA")) is False\n',
        '    assert analyzer.detect_gradient_like_surface(Image.fromarray(smooth_only, "RGBA")) is False\n    assert analyzer.detect_hard_stop_gradient_surface(Image.fromarray(smooth_only, "RGBA")) is False\n',
        "hard stop negative",
    )
p.write_text(s, encoding="utf-8")

p = Path("engine/test_ai2_component_quality.py")
s = p.read_text(encoding="utf-8")
if "test_safe_perfect_legacy_winner_is_preserved_from_pareto_churn" not in s:
    s += r'''


def _selection_candidate(name: str, *, total: float, fidelity: float, component: dict, hard_stop: bool = False) -> dict:
    return {
        "name": name, "total_score": total, "fidelity_score": fidelity,
        "rendered_ok": True, "selection_safe": True,
        "selection_disqualified": False, "hard_stop_gradient": hard_stop,
        "component_quality": component,
        "score_details": {"path_count": 10, "edge_f1": fidelity / 100.0,
                          "ssim": fidelity / 100.0, "mean_delta_e": 100.0 - fidelity,
                          "has_bitmap": False},
    }


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
p.write_text(s, encoding="utf-8")
