from __future__ import annotations

import ast
from pathlib import Path


def _run_pipeline_node() -> ast.FunctionDef:
    source = Path("app/pipeline.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline":
            return node
    raise AssertionError("run_pipeline not found")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def test_edge_cleanup_journal_precedes_rescore_and_gates_it() -> None:
    run_pipeline = _run_pipeline_node()
    calls = [node for node in ast.walk(run_pipeline) if isinstance(node, ast.Call)]

    journal_calls = [
        node
        for node in calls
        if _call_name(node) == "consider_candidate"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "edge_cleanup"
            for arg in node.args
        )
    ]
    assert len(journal_calls) == 1
    journal_call = journal_calls[0]

    edge_score_calls = [
        node
        for node in calls
        if _call_name(node) == "score_candidate"
        and any(
            isinstance(arg, ast.Name) and arg.id == "cleaned"
            for arg in node.args
        )
    ]
    assert len(edge_score_calls) == 1
    score_call = edge_score_calls[0]
    assert journal_call.lineno < score_call.lineno

    accepted_if = next(
        (
            node
            for node in ast.walk(run_pipeline)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "accepted_path"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
            and node.test.comparators[0].id == "ec_dst"
        ),
        None,
    )
    assert accepted_if is not None
    assert any(node is score_call for node in ast.walk(accepted_if))


def test_edge_cleanup_rejection_branch_contains_no_rescore() -> None:
    run_pipeline = _run_pipeline_node()
    accepted_if = next(
        node
        for node in ast.walk(run_pipeline)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "accepted_path"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Name)
        and node.test.comparators[0].id == "ec_dst"
    )
    rejection_calls = [
        node
        for stmt in accepted_if.orelse
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and _call_name(node) == "score_candidate"
    ]
    assert rejection_calls == []
