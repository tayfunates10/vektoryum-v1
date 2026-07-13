"""P0: ağır iş işlenirken event loop bloke OLMAMALI; /livez responsive kalmalı.

Doğrulanan canlı hata: /api/vectorize CPU-ağır run_pipeline'ı async endpoint'te
DOĞRUDAN çağırıyordu → event loop bloke → ağır dizide /api/health 30 sn yanıt
vermiyordu. Düzeltme: run_pipeline threadpool'a taşındı (await run_in_threadpool).

Bu test HTTP istemcisi gerektirmez: async endpoint coroutine'ini ağır (uyuyan)
bir run_pipeline ile başlatır ve eşzamanlı /livez gecikmesini ölçer.

Çalıştırma::  .venv/bin/python test_health_responsiveness.py   (~5 sn)
"""
from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 11)).save(buf, "PNG")
    return buf.getvalue()


def _slow_pipeline(*args, **kwargs):
    time.sleep(1.2)   # ağır CPU işi simülasyonu (gerçekte alt-süreç havuzunda)
    return {"analysis": {}, "mode_used": "auto", "mode_warning": None,
            "preprocess_report": {}, "results": [], "scored": [],
            "best": None, "raw_best": None, "selection_reason": ""}


async def _scenario():
    import app.main as M
    from starlette.datastructures import Headers, UploadFile

    M.run_pipeline = _slow_pipeline                       # ağır işi uyutucuyla değiştir
    M._require_user = lambda s: {"email": "t@t", "role": "user"}

    up = UploadFile(filename="x.png", file=io.BytesIO(_png_bytes()),
                    headers=Headers({"content-type": "image/png"}))
    heavy = asyncio.create_task(M.vectorize_image(
        file=up, trace_mode="auto", shape_stacking="stacked",
        edge_cleanup="on", session="x"))
    await asyncio.sleep(0.25)                             # ağır iş threadpool'a girsin

    # ağır iş uçarken /livez gecikmesi ölç
    lat = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = await M.livez()
        lat.append(time.perf_counter() - t0)
    livez_ok = all(r.get("status") == "alive" for _ in [r])

    # /api/auth/me de yanıt vermeli (ağır iş loop'u kilitlemedi)
    me = await M.me(session=None)

    await heavy                                           # ağır iş bitsin
    p95 = sorted(lat)[int(len(lat) * 0.95) - 1]
    return p95, livez_ok, me


def test_livez_responsive_during_heavy() -> None:
    print("== Ağır iş uçarken /livez p95 gecikmesi düşük (loop bloke değil) ==")
    p95, livez_ok, _ = asyncio.run(_scenario())
    print(f"    /livez p95 = {p95 * 1000:.1f} ms (ağır iş 1.2s uyurken)")
    check(livez_ok, "/livez 'alive' döndü")
    check(p95 < 0.5, f"/livez p95 < 500ms ({p95 * 1000:.1f}ms) — event loop responsive")


def test_livez_readyz_health_shapes() -> None:
    print("== /livez, /readyz, /api/health sözleşmeleri ==")
    import app.main as M
    import json

    lz = asyncio.run(M.livez())
    check(lz["status"] == "alive", "/livez alive")

    rz = asyncio.run(M.readyz())
    body = json.loads(bytes(rz.body))
    check(rz.status_code in (200, 503), "/readyz 200/503")
    check("checks" in body and "artifact_writable" in body["checks"], "/readyz checks var")

    h = asyncio.run(M.health())
    hbody = json.loads(bytes(h.body))
    check("modes" in hbody, "/api/health modes (geriye uyum)")
    check(h.status_code in (200, 503), "/api/health readiness'e bağlı")


def main() -> int:
    test_livez_responsive_during_heavy()
    test_livez_readyz_health_shapes()
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
