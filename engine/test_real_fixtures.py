"""Gerçek logo görselleriyle (Class Reklam / ARCAATES) uçtan uca regresyon.

Kullanım
--------
1) Görselleri kaydedip varsayılan yollara koyun:

    engine\\regression\\fixtures\\class_reklam.png
    engine\\regression\\fixtures\\arcaates.png

   sonra:

    .\\.venv\\Scripts\\python.exe test_real_fixtures.py

2) Veya görselleri herhangi bir yere kaydedip yollarını verin:

    .\\.venv\\Scripts\\python.exe test_real_fixtures.py "C:\\...\\cr.png" "C:\\...\\arca.png"

Çıktılar engine\\regression\\output\\ altına yazılır (svg/pdf/eps/dxf + report.json).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

# Windows konsolu cp1254 olabilir; Unicode çıktısı için UTF-8'e geç.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from starlette.datastructures import Headers, UploadFile  # noqa: E402

from app.main import _job_dir, vectorize_image  # noqa: E402

FIXTURES = ENGINE_DIR / "regression" / "fixtures"
OUTPUT = ENGINE_DIR / "regression" / "output"


def _find_fixture(*names: str) -> Path | None:
    for name in names:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = FIXTURES / f"{name}{ext}"
            if p.exists():
                return p
    return None


def _upload(path: Path) -> UploadFile:
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".") or "png"
    ctype = {"jpg": "jpeg"}.get(ext, ext)
    return UploadFile(file=io.BytesIO(data), filename=path.name,
                      headers=Headers({"content-type": f"image/{ctype}"}))


async def _run(path: Path) -> dict:
    resp = await vectorize_image(file=_upload(path), trace_mode="auto")
    data = json.loads(resp.body.decode())
    data["_http_status"] = getattr(resp, "status_code", 200)
    return data


def _evaluate(label: str, data: dict, expected_mode: str) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    if "error" in data:
        checks.append((f"{label}: pipeline başarılı", False, data["error"]))
        return checks

    an = data["analysis"]
    cr = data["candidate_report"]
    qr = data["quality_report"]
    det = (next((c for c in cr["candidates"] if c["name"] == cr["best_candidate"]), {}) or {}).get("details", {}) or {}

    checks.append((f"{label}: mode_used == {expected_mode}", data["mode_used"] == expected_mode, f"={data['mode_used']}"))
    checks.append((f"{label}: recommended_mode == {expected_mode}", an["recommended_mode"] == expected_mode, f"={an['recommended_mode']}"))
    checks.append((f"{label}: aday sayısı >= 4", len(cr["candidates"]) >= 4, f"={len(cr['candidates'])}"))
    checks.append((f"{label}: export hatası yok (output_errors boş)", not data["output_errors"], f"={list(data['output_errors'].keys())}"))

    if expected_mode == "geometric_logo":
        checks.append((f"{label}: likely_geometric_logo == True", an["likely_geometric_logo"] is True, f"={an['likely_geometric_logo']}"))
        # _refit (renk) / _bnd (sınır) sonekleri kazananın orijinale yeniden
        # oturtulmuş türevleridir; temel aday adına göre denetlenir
        base_name = cr["best_candidate"]
        while base_name.endswith(("_refit", "_bnd")):
            base_name = base_name.rsplit("_", 1)[0]
        checks.append((f"{label}: best in geo_standard/clean/contour/mixed",
                       base_name in {"geo_standard", "geo_clean", "geo_contour", "geo_mixed"},
                       f"={cr['best_candidate']} (reason={cr['selection_reason']})"))
        checks.append((f"{label}: quality production_ready/needs_review",
                       qr["status"] in {"production_ready", "needs_review"}, f"={qr['status']} warnings={qr['warnings']}"))
    else:  # logo_color
        checks.append((f"{label}: likely_geometric_logo == False", an["likely_geometric_logo"] is False, f"={an['likely_geometric_logo']}"))
        # bilgilendirici (hedef aralık) — başarısızlık saymaz, sadece raporlanır
        pc = det.get("path_count")
        uc = det.get("unique_colors")
        fid = (next((c for c in cr["candidates"] if c["name"] == cr["best_candidate"]), {}) or {}).get("fidelity_score")
        print(f"   [bilgi] {label} path_count={pc}, unique_colors={uc}, fidelity={fid} "
              f"(seçim sadakat-öncelikli; az path = daha düzenlenebilir)")

    return checks


async def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    if len(args) >= 2:
        geo_src, color_src = Path(args[0]), Path(args[1])
        # kanonik isimlerle fixtures'a kopyala
        FIXTURES.mkdir(parents=True, exist_ok=True)
        if geo_src.exists():
            shutil.copyfile(geo_src, FIXTURES / f"class_reklam{geo_src.suffix.lower()}")
        if color_src.exists():
            shutil.copyfile(color_src, FIXTURES / f"arcaates{color_src.suffix.lower()}")

    geo = _find_fixture("class_reklam", "cr", "classreklam")
    color = _find_fixture("arcaates", "arcaates_logo", "arca")

    if not geo and not color:
        print("HİÇ FIXTURE BULUNAMADI.\n")
        print("Görselleri şu yollara kaydedip tekrar çalıştırın:")
        print(f"  {FIXTURES / 'class_reklam.png'}")
        print(f"  {FIXTURES / 'arcaates.png'}")
        print("\nveya: python test_real_fixtures.py <class_reklam_yolu> <arcaates_yolu>")
        return 2

    all_checks: list[tuple[str, bool, str]] = []
    report: dict = {}

    for label, src, expected in (("class_reklam", geo, "geometric_logo"), ("arcaates", color, "logo_color")):
        print("=" * 64)
        if not src:
            print(f"[ATLA] {label}: fixture bulunamadı")
            continue
        print(f"[ÇALIŞ] {label}  <-  {src.name}")
        data = await _run(src)
        report[label] = {k: data[k] for k in ("mode_used", "candidate_report", "quality_report", "output_errors", "outputs") if k in data}

        if "error" not in data:
            an = data["analysis"]
            cr = data["candidate_report"]
            print(f"  detected={an['detected_type']} recommended={an['recommended_mode']} mode_used={data['mode_used']}")
            print(f"  colors={an['estimated_color_count']} edge={an['edge_density']} likely_geo={an['likely_geometric_logo']}")
            print(f"  best={cr['best_candidate']} (raw={cr['raw_best_candidate']}, reason={cr['selection_reason']})")
            for c in cr["candidates"]:
                d = c.get("details") or {}
                print(f"    {c['name']:<20} ok={c.get('success')!s:<5} total={c.get('total_score')} "
                      f"se={c.get('straight_edge_score')} cc={c.get('corner_cleanliness_score')} "
                      f"ax={c.get('axis_alignment_score')} paths={d.get('path_count')} colors={d.get('unique_colors')}")
            print(f"  quality={data['quality_report']['status']} warnings={data['quality_report']['warnings']}")
            print(f"  outputs={list(data['outputs'].keys())} output_errors={list(data['output_errors'].keys())}")

            # çıktıları regression/output'a kopyala
            jd = _job_dir(data["job_id"])
            for fmt in ("svg", "pdf", "eps", "dxf"):
                f = jd / f"{data['job_id']}.{fmt}"
                if f.exists():
                    shutil.copyfile(f, OUTPUT / f"{label}.{fmt}")
            print(f"  -> kaydedildi: {OUTPUT / (label + '.svg')} (+pdf/eps/dxf)")

        checks = _evaluate(label, data, expected)
        all_checks.extend(checks)
        for name, ok, detail in checks:
            print(f"   [{'PASS' if ok else 'FAIL'}] {name} :: {detail}")

    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(1 for _, ok, _ in all_checks if ok)
    print("\n" + "=" * 64)
    print(f"SONUC: {passed}/{len(all_checks)} kabul kriteri gecti")
    print(f"Ciktilar: {OUTPUT}")
    print("=" * 64)
    return 0 if passed == len(all_checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
