from __future__ import annotations

from pathlib import Path
from typing import Any

from app.centerline_contracts import _attach_score_contract


def test_centerline_score_contract_forwards_optional_render_fn() -> None:
    received: dict[str, Any] = {}

    def scorer(
        *,
        original_path: Path,
        svg_path: Path,
        analysis_report: dict[str, Any],
        mode: str,
        geometry_report: dict[str, Any] | None = None,
        render_fn: Any = None,
    ) -> dict[str, Any]:
        received["render_fn"] = render_fn
        return {"score_details": {}}

    def render_fn(path: Path, width: int, height: int) -> object:
        return object()

    wrapped = _attach_score_contract(scorer)
    result = wrapped(
        original_path=Path("source.png"),
        svg_path=Path("candidate.svg"),
        analysis_report={},
        mode="auto",
        render_fn=render_fn,
    )

    assert result == {"score_details": {}}
    assert received["render_fn"] is render_fn
