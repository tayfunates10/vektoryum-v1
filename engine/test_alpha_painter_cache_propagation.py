from __future__ import annotations
import ast
from pathlib import Path

def _function(path: str, name: str) -> ast.FunctionDef:
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)

def _kw(call: ast.Call, name: str) -> ast.keyword:
    values = [kw for kw in call.keywords if kw.arg == name]
    assert len(values) == 1
    return values[0]

def test_painter_journal_reuses_outer_request_cache() -> None:
    apply_fn = _function("app/alpha_candidate_painter.py", "apply_candidate_painter_reconstruction")
    assert "measurement_cache" in [arg.arg for arg in apply_fn.args.args]
    calls = [node for node in ast.walk(apply_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_run_painter_geometry_journal"]
    assert len(calls) == 2
    for call in calls:
        value = _kw(call, "measurement_cache").value
        assert isinstance(value, ast.Name) and value.id == "measurement_cache"
    journal_fn = _function("app/alpha_candidate_painter.py", "_run_painter_geometry_journal")
    assert "measurement_cache" in [arg.arg for arg in journal_fn.args.args]
    constructors = [node for node in ast.walk(journal_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TransformJournal"]
    assert len(constructors) == 1
    value = _kw(constructors[0], "measurement_cache").value
    assert isinstance(value, ast.Name) and value.id == "measurement_cache"

def test_alpha_wrapper_passes_request_local_cache_to_painter() -> None:
    wrapper = _function("app/alpha_svg_mask.py", "alpha_mask_finalized_pipeline")
    calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "apply_candidate_painter_reconstruction"]
    assert len(calls) == 1
    value = _kw(calls[0], "measurement_cache").value
    assert isinstance(value, ast.Name) and value.id == "alpha_measurement_cache"
