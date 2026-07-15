from __future__ import annotations

import importlib


def test_runtime_main_reuses_existing_fastapi_app(monkeypatch):
    monkeypatch.setenv("VEKTORYUM_SHADOW_SELECTOR", "off")

    runtime_main = importlib.import_module("app.runtime_main")
    main = importlib.import_module("app.main")
    pipeline_entry = importlib.import_module("app.pipeline_entry")
    exporters = importlib.import_module("app.exporters")

    assert runtime_main.app is main.app
    assert main.run_pipeline is pipeline_entry.run_pipeline
    assert main.export_all is runtime_main._runtime_export_all
    assert runtime_main._legacy_export_all is exporters.export_all
    assert runtime_main.app.state.request_compat_installed is True


def test_runtime_entry_keeps_shadow_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VEKTORYUM_SHADOW_SELECTOR", raising=False)

    runtime = importlib.import_module("app.shadow_runtime")

    assert runtime.shadow_selector_enabled() is False
