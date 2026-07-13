"""P0: paralel havuz hatası KONTROLLÜ olmalı — inline sıralı yeniden çalıştırma YOK.

Doğrulanan canlı hata: _run_jobs timeout/BrokenProcessPool sonrası aynı ağır
işleri sıralı tekrar çalıştırıp çift CPU-ağır iş + event-loop kilidi + /api/health
zaman aşımı doğuruyordu. Artık: havuz sıfırlanır (çocuklar öldürülür) ve
WorkerFailure üretilir; işler İNLINE YENİDEN ÇALIŞTIRILMAZ.

Çalıştırma::  .venv/bin/python test_worker_failure.py   (~2 sn)
"""
from __future__ import annotations

import sys
from concurrent.futures import TimeoutError as FTimeout
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


class _FakePool:
    def __init__(self, exc):
        self.exc = exc
        self._processes = {}
        self.shutdown_called = False

    def map(self, fn, jobs, timeout=None):
        raise self.exc

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_called = True


def test_timeout_no_inline_rerun() -> None:
    print("== Havuz timeout → WorkerFailure, inline sıralı rerun YOK ==")
    from app import pipeline as P

    calls = {"n": 0}

    def _spy(job):
        calls["n"] += 1
        return ({"name": "x"}, None)

    fake = _FakePool(FTimeout("map timed out"))
    orig_pool, orig_size, orig_job = P._get_pool, P._pool_size, P._produce_and_score_job
    P._get_pool = lambda: fake
    P._pool_size = lambda: 4
    P._produce_and_score_job = _spy
    P._POOL = fake
    try:
        raised = False
        try:
            P._run_jobs([("a",), ("b",), ("c",)])
        except P.WorkerFailure:
            raised = True
        check(raised, "WorkerFailure üretildi (timeout)")
        check(calls["n"] == 0, f"iş inline yeniden çalıştırılmadı ({calls['n']} çağrı)")
        check(fake.shutdown_called, "kırık havuz shutdown edildi")
        check(P._POOL is None, "havuz tekilliği sıfırlandı (sonraki istek taze kurar)")
    finally:
        P._get_pool, P._pool_size, P._produce_and_score_job = orig_pool, orig_size, orig_job
        P._POOL = None


def test_broken_pool_no_inline_rerun() -> None:
    print("== BrokenProcessPool → WorkerFailure, rerun YOK ==")
    from app import pipeline as P
    try:
        from concurrent.futures.process import BrokenProcessPool
    except Exception:  # noqa: BLE001
        BrokenProcessPool = RuntimeError

    calls = {"n": 0}
    fake = _FakePool(BrokenProcessPool("pool broke"))
    orig = (P._get_pool, P._pool_size, P._produce_and_score_job)
    P._get_pool = lambda: fake
    P._pool_size = lambda: 4
    P._produce_and_score_job = lambda job: calls.__setitem__("n", calls["n"] + 1)
    P._POOL = fake
    try:
        raised = False
        try:
            P._run_jobs([("a",), ("b",)])
        except P.WorkerFailure:
            raised = True
        check(raised, "WorkerFailure üretildi (broken pool)")
        check(calls["n"] == 0, "inline rerun yok")
    finally:
        P._get_pool, P._pool_size, P._produce_and_score_job = orig
        P._POOL = None


def test_single_job_sequential_ok() -> None:
    print("== Tek iş / havuz yok → sıralı (çift-iş değil, hata değil) ==")
    from app import pipeline as P
    calls = {"n": 0}
    orig = (P._pool_size, P._produce_and_score_job)
    P._pool_size = lambda: 4
    P._produce_and_score_job = lambda job: (calls.__setitem__("n", calls["n"] + 1) or ({"name": "x"}, None))
    try:
        out = P._run_jobs([("only",)])   # tek iş → doğrudan sıralı
        check(calls["n"] == 1, "tek iş bir kez çalıştı")
        check(len(out) == 1, "sonuç döndü")
    finally:
        P._pool_size, P._produce_and_score_job = orig


def test_job_timeout_env() -> None:
    print("== VEKTORYUM_JOB_TIMEOUT env ==")
    import os
    from app import pipeline as P
    os.environ["VEKTORYUM_JOB_TIMEOUT"] = "120"
    check(P._job_timeout() == 120.0, "env timeout okundu")
    os.environ["VEKTORYUM_JOB_TIMEOUT"] = "5"
    check(P._job_timeout() == 30.0, "alt sınır 30s uygulandı")
    os.environ.pop("VEKTORYUM_JOB_TIMEOUT", None)
    check(P._job_timeout() == 600.0, "varsayılan 600s")


def main() -> int:
    test_timeout_no_inline_rerun()
    test_broken_pool_no_inline_rerun()
    test_single_job_sequential_ok()
    test_job_timeout_env()
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
