"""FAZ 1 — Exact final SVG TEK gerçeklik entegrasyonu (gerçek pipeline).

Doğrulanan hata: main.py export sonrası basic_svg_quality_check kullanıyor;
FinalArtifactEvaluator production'a bağlı değildi → API metriği (path/byte) stale,
indirilen dosyayla eşleşmiyor (T5). Artık quality_report exact exported SVG'den
türer; response final_svg_sha256 indirilen baytların SHA-256'sıdır; response
path_count exact XML <path> sayısıdır.

Özel zorunlu assertion'lar (şartname):
1. Response final_svg_sha256 == indirilen byte SHA-256.
2. Response path_count == exact XML <path> count.
5. Ölçülemeyen zorunlu metrik varsa production_ready değil.

Çalıştırma::  .venv/bin/python test_exact_final_artifact_integration.py  (~60 sn)
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ENGINE_DIR / "regression"))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _png_bytes(rgb: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "PNG")
    return b.getvalue()


def _run_endpoint(png: bytes, filename="t1.png"):
    import app.main as M
    from starlette.datastructures import Headers, UploadFile
    M._require_user = lambda s: {"email": "it@t", "role": "user"}
    up = UploadFile(filename=filename, file=io.BytesIO(png),
                    headers=Headers({"content-type": "image/png"}))
    resp = asyncio.run(M.vectorize_image(
        file=up, trace_mode="auto", shape_stacking="stacked",
        edge_cleanup="on", session="x"))
    body = json.loads(bytes(resp.body))
    return M, resp, body


def _download_bytes(M, job_id: str, fmt: str) -> bytes | None:
    try:
        fr = asyncio.run(M.download_file(job_id, fmt))
    except Exception:  # noqa: BLE001
        return None
    path = getattr(fr, "path", None)
    return Path(path).read_bytes() if path and Path(path).exists() else None


def test_sha_and_pathcount_match_exact() -> None:
    print("== Exact final: sha256 zinciri + path_count == XML (stale değil) ==")
    from exact_corpus import t1_topology
    fx = t1_topology(256)
    M, resp, body = _run_endpoint(_png_bytes(fx.rgb))
    check(getattr(resp, "status_code", 200) == 200, f"200 döndü ({getattr(resp,'status_code',200)})")
    if getattr(resp, "status_code", 200) != 200:
        return
    job_id = body["job_id"]
    fa = body.get("final_artifact", {})
    qr = body.get("quality_report", {})
    check(qr.get("source") == "final_artifact_evaluator", "quality_report evaluator'dan türedi")
    resp_sha = body.get("final_svg_sha256")
    check(bool(resp_sha), f"final_svg_sha256 response'ta var ({(resp_sha or '')[:8]})")

    # 1) response sha == indirilen byte sha
    dl = _download_bytes(M, job_id, "svg")
    check(dl is not None, "svg indirilebildi")
    if dl is not None:
        dl_sha = hashlib.sha256(dl).hexdigest()
        check(dl_sha == resp_sha, "response sha256 == indirilen SVG sha256 (KESİN dosya)")

        # 2) response path_count == exact XML <path> count
        xml_paths = len(re.findall(r"<path\b", dl.decode("utf-8", "ignore")))
        rep_paths = (fa.get("exact_metrics") or {}).get("path_count")
        check(rep_paths == xml_paths, f"path_count == XML ({rep_paths} vs {xml_paths})")
        # XML gerçekten parse edilebilir + bitmap yok
        try:
            ET.fromstring(dl)
            parse_ok = True
        except Exception:  # noqa: BLE001
            parse_ok = False
        check(parse_ok, "exported SVG XML parse edilebilir")
        check(b"<image" not in dl and b"data:image" not in dl, "gömülü bitmap yok")

    # download link yalnız geçerli format için
    dls = set(body.get("download_links", {}).keys())
    valid = set(fa.get("valid_formats", []))
    check(dls == valid, f"download_links == valid_formats ({sorted(dls)})")


def test_verdict_is_honest() -> None:
    print("== Verdict dürüst: production_ready|needs_review|failed; unmeasured→değil ==")
    from exact_corpus import t1_topology
    _M, resp, body = _run_endpoint(_png_bytes(t1_topology(256).rgb))
    if getattr(resp, "status_code", 200) != 200:
        check(False, "endpoint 200 değil")
        return
    fa = body["final_artifact"]
    v = fa["verdict"]
    check(v in ("production_ready", "needs_review", "failed"), f"geçerli verdict ({v})")
    # ölçülemeyen zorunlu metrik varsa production_ready OLAMAZ
    if fa.get("unmeasured_required"):
        check(v != "production_ready", "unmeasured varsa production_ready değil")
    else:
        check(True, "unmeasured yok")
    # legacy rapor ayrı tutuluyor, final kararı etkilemiyor
    check("legacy_candidate_report" in body, "legacy_candidate_report ayrı")


def main() -> int:
    test_sha_and_pathcount_match_exact()
    test_verdict_is_honest()
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
