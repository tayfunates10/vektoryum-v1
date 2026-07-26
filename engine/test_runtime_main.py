from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def test_runtime_main_reuses_existing_fastapi_app(monkeypatch):
    """Exercise the production entrypoint in a fresh interpreter.

    The ASGI entrypoint is imported before the application starts in production.
    Running this contract in the shared pytest interpreter made its result depend
    on whether an earlier TestClient had already built FastAPI's middleware stack.
    """
    monkeypatch.setenv("VEKTORYUM_SHADOW_SELECTOR", "off")
    script = """
import importlib

runtime_main = importlib.import_module("app.runtime_main")
main = importlib.import_module("app.main")
pipeline_entry = importlib.import_module("app.pipeline_entry")
exporters = importlib.import_module("app.exporters")

assert runtime_main.app is main.app
assert main.run_pipeline is pipeline_entry.run_pipeline
assert main.export_all is runtime_main._runtime_export_all
assert runtime_main._legacy_export_all is exporters.export_all
assert runtime_main.app.state.request_compat_installed is True
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_entry_keeps_shadow_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VEKTORYUM_SHADOW_SELECTOR", raising=False)

    runtime = importlib.import_module("app.shadow_runtime")

    assert runtime.shadow_selector_enabled() is False
