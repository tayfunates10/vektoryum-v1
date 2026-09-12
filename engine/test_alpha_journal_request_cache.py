from __future__ import annotations

import ast
from pathlib import Path


def _function(path: str, name: str) -> ast.FunctionDef:
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_alpha_retry_shares_only_request_local_measurement_cache() -> None:
    wrapper = _function("app/alpha_svg_mask.py", "alpha_mask_finalized_pipeline")
    assignments = [
        node for node in ast.walk(wrapper)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "alpha_measurement_cache"
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Dict)

    journal_calls = [
        node for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TransformJournal"
    ]
    assert len(journal_calls) == 2
    for call in journal_calls:
        cache_args = [kw for kw in call.keywords if kw.arg == "measurement_cache"]
        assert len(cache_args) == 1
        assert isinstance(cache_args[0].value, ast.Name)
        assert cache_args[0].value.id == "alpha_measurement_cache"


def test_transform_journal_accepts_injected_instance_cache() -> None:
    constructor = _function("app/transform_journal.py", "__init__")
    keyword_names = [arg.arg for arg in constructor.args.kwonlyargs]
    assert "measurement_cache" in keyword_names

    cache_assignments = [
        node for node in ast.walk(constructor)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
        and node.target.attr == "_cache"
    ]
    assert len(cache_assignments) == 1
    assert any(
        isinstance(node, ast.Name) and node.id == "measurement_cache"
        for node in ast.walk(cache_assignments[0].value)
    )
