"""Shadow sistem production izolasyonu + feature flag regresyonları (SHADOW).

Kanıt: shadow modülleri production yolundan (pipeline/exporters/main) import
EDİLMEZ; flag'ler varsayılan KAPALI; güvenli sarmalayıcı flag kapalıyken None
döner ve hata yutar. Böylece production SVG shadow'dan bağımsız (byte-değişmez).

Çalıştırma::  .venv/bin/python test_shadow_isolation.py   (~3 sn)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []
SHADOW_MODULES = {
    "app.half_edge_graph", "app.region_consolidation", "app.graph_source",
    "app.canonical_curve", "app.graph_serializer", "app.shadow_pipeline",
}
PROD_MODULES = ["app.pipeline", "app.exporters", "app.main"]


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def test_no_static_import_in_production() -> None:
    print("== Production kaynak dosyaları shadow modülü import etmez ==")
    shadow_names = ["half_edge_graph", "region_consolidation", "graph_source",
                    "canonical_curve", "graph_serializer", "shadow_pipeline"]
    for mod in ("pipeline", "exporters", "main"):
        src = (ENGINE_DIR / "app" / f"{mod}.py").read_text()
        hit = [s for s in shadow_names if f"import {s}" in src or f"from app.{s}" in src]
        check(not hit, f"{mod}.py shadow import yok ({hit})")


def test_production_import_does_not_load_shadow() -> None:
    print("== Production modülü import edilince shadow modülü yüklenmez ==")
    for m in list(sys.modules):
        if m in SHADOW_MODULES:
            del sys.modules[m]
    import importlib
    for pm in PROD_MODULES:
        importlib.import_module(pm)
    loaded = SHADOW_MODULES & set(sys.modules)
    check(not loaded, f"production import sonrası shadow modül yüklenmedi ({loaded})")


def test_flags_default_off() -> None:
    print("== Feature flag'ler varsayılan KAPALI ==")
    from app import shadow_pipeline as sp
    for fn in (sp.half_edge_shadow_enabled, sp.consolidation_shadow_enabled,
               sp.canonical_curve_shadow_enabled, sp.graph_serializer_shadow_enabled):
        # ilgili env değişkenini temizle
        name = {
            sp.half_edge_shadow_enabled: "VEKTORYUM_HALF_EDGE_SHADOW",
            sp.consolidation_shadow_enabled: "VEKTORYUM_REGION_CONSOLIDATION_SHADOW",
            sp.canonical_curve_shadow_enabled: "VEKTORYUM_CANONICAL_CURVE_SHADOW",
            sp.graph_serializer_shadow_enabled: "VEKTORYUM_GRAPH_SERIALIZER_SHADOW",
        }[fn]
        os.environ.pop(name, None)
        check(fn() is False, f"{name} varsayılan kapalı")


def test_safe_wrapper_returns_none_when_disabled() -> None:
    print("== Güvenli sarmalayıcı: flag kapalıyken None (production'ı çalıştırmaz) ==")
    from app.shadow_pipeline import build_shadow_graph_safe
    os.environ.pop("VEKTORYUM_HALF_EDGE_SHADOW", None)
    img = np.full((20, 20, 3), 255, np.uint8)
    check(build_shadow_graph_safe(img) is None, "flag kapalı → None")


def test_safe_wrapper_swallows_errors() -> None:
    print("== Güvenli sarmalayıcı: shadow hatası yutulur (None), production düşmez ==")
    from app.shadow_pipeline import build_shadow_graph_safe
    os.environ["VEKTORYUM_HALF_EDGE_SHADOW"] = "1"
    try:
        # bozuk girdi (yanlış shape) → içeride hata; sarmalayıcı None dönmeli
        bad = np.zeros((5,), np.uint8)
        check(build_shadow_graph_safe(bad) is None, "hata yutuldu → None")
    finally:
        os.environ.pop("VEKTORYUM_HALF_EDGE_SHADOW", None)


def main() -> int:
    test_no_static_import_in_production()
    test_production_import_does_not_load_shadow()
    test_flags_default_off()
    test_safe_wrapper_returns_none_when_disabled()
    test_safe_wrapper_swallows_errors()
    print("=" * 60)
    if FAILS:
        print(f"SONUC: {len(FAILS)} KONTROL BASARISIZ")
        for m in FAILS:
            print(" -", m)
        return 1
    print("SONUC: tum kontroller gecti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
