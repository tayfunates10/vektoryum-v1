from __future__ import annotations

import json
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
ROADMAP_PATH = ENGINE_DIR / "production_platform_roadmap.json"


def _roadmap() -> dict:
    return json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))


def test_schema() -> None:
    assert _roadmap()["schema_version"] == "production-platform-closure-v1"
