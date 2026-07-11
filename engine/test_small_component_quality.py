"""Küçük bileşen kalite regresyonu (LEGO ® vakası).

Genel benzerlik yüksekken küçük ama anlamlı bir bileşenin (ör. ® simgesi)
bozulmasını yakalar. Kök vaka: boundary_refit'in yay uçlarını kaydırması
diametral kirişli daireleri şişiriyordu (halka 186px -> 221px); global skor
%99+ kaldığı için seçim hatayı görmüyordu.

Kullanım::

    .venv/bin/python test_small_component_quality.py
    .venv/bin/python test_small_component_quality.py --keep   # debug çıktıları sakla

Süre: ~3-4 dk (tam pipeline koşar). Sabit palet YOK: renk sınıfları kaynaktan
k-means ile örneklenir; bölge koordinatları görüntüden ölçülür (hardcode yok).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

# izleme tavanı: üretimle aynı (engine/Dockerfile) — küçük öğe hassasiyeti
os.environ.setdefault("VEKTORYUM_TRACE_CAP", "2200")

FIXTURE = ENGINE_DIR / "regression" / "fixtures" / "lego_rmark.png"

# Eşikler (ölçülmüş baseline'a göre; gevşetme = hatayı geçirme, yapma):
MIN_PALETTE_AGREE = 0.995      # genel palet sınıf uyumu
MIN_CLASS_IOU = 0.965          # her palet sınıfı için IoU tabanı
MIN_SMALL_COMPONENT_IOU = 0.90 # küçük anlamlı bileşenlerin en kötüsü (® dahil)
MAX_SVG_COMMANDS = 900         # karmaşıklık sınırsız büyümesin


def _fail(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="debug çıktılarının dizinini yazdır ve silme")
    args = ap.parse_args()

    if not FIXTURE.exists():
        print(f"FIXTURE YOK: {FIXTURE}")
        return 1

    from app.exporters import export_svg
    from app.fidelity import render_svg_to_rgb
    from app.pipeline import run_pipeline

    im = Image.open(FIXTURE).convert("RGB")
    w, h = im.size
    job = Path(tempfile.mkdtemp(prefix="smallcomp_"))
    res = run_pipeline(im, FIXTURE, "auto", job, edge_cleanup=True)
    best = res.get("best") or {}
    errors: list[str] = []
    _fail(errors, bool(best.get("svg_path")), "pipeline kazanan üretmedi")
    if errors:
        print("FAIL:", errors)
        return 1
    # ÜRETİM YOLU: kullanıcıya giden dosya export katmanından geçer
    # (koordinat normalizasyonu pipeline içinde, fill-rule export'ta).
    svg_path = export_svg(Path(best["svg_path"]), job / "final.svg",
                          f"{res.get('mode_used')}:{best.get('name')}")
    svg_txt = svg_path.read_text()

    # --- A. SVG yapısal denetim -------------------------------------------
    root = ET.fromstring(svg_txt)
    _fail(errors, bool(root.get("viewBox")), "viewBox yok")
    _fail(errors, root.get("width") == str(w) and root.get("height") == str(h),
          f"width/height kaynakla uyumsuz: {root.get('width')}x{root.get('height')} != {w}x{h}")
    # KOORDİNAT SÖZLEŞMESİ: viewBox = kaynak uzay (2200/3840 ayrışması yok),
    # transform düzleşmiş, koordinatlar sınır içinde (3841.001 tarzı taşma yok)
    vb = (root.get("viewBox") or "").replace(",", " ").split()
    _fail(errors, len(vb) == 4 and float(vb[2]) == w and float(vb[3]) == h,
          f"viewBox kaynak uzayda değil: {root.get('viewBox')} != 0 0 {w} {h}")
    _fail(errors, 'transform=' not in svg_txt, "düzleştirilmemiş transform kaldı")
    d_all = " ".join(re.findall(r'd="([^"]+)"', svg_txt))
    _fail(errors, "NaN" not in d_all and "Infinity" not in d_all, "path verisinde NaN/Infinity")
    coords = [float(v) for v in re.findall(r"-?\d+\.?\d*", d_all)]
    _fail(errors, min(coords) >= -1.5 and max(coords) <= max(w, h) + 1.5,
          f"koordinatlar viewBox dışında: min={min(coords)} max={max(coords)}")
    cmds = len(re.findall(r"[MLCQAZHVSTmlcqazhvst]", d_all))
    _fail(errors, cmds <= MAX_SVG_COMMANDS, f"komut sayısı {cmds} > {MAX_SVG_COMMANDS}")
    for m in re.finditer(r"<path[^>]*>", svg_txt):
        tag = m.group(0)
        dm = re.search(r'd="([^"]*)"', tag)
        if dm and len(re.findall(r"(?<![0-9a-zA-Z.,-])[Mm]", " " + dm.group(1))) >= 2:
            _fail(errors, "fill-rule" in tag, "bileşik path'te açık fill-rule yok")

    # --- RENK SÖZLEŞMESİ (grayscale-red regresyonu) -----------------------
    # Kaynak kromatik + opak: çıktı siyah + fill-opacity katmanlarıyla renk
    # TAKLİT EDEMEZ; gerçek dolgular ve opak zemin zorunludur.
    _fail(errors, not re.search(r'fill-opacity="0?\.\d+"', svg_txt),
          "SVG fill-opacity katmanlarıyla renk taklit ediyor (grayscale çıktı)")
    fills = sorted({f.lower() for f in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg_txt)})
    _fail(errors, len(fills) >= 4, f"en az 4 gerçek dolgu rengi bekleniyordu: {fills}")

    def _near(hex_c: str, rgb: tuple[int, int, int], tol: int = 60) -> bool:
        v = int(hex_c[1:], 16)
        return abs((v >> 16) - rgb[0]) + abs(((v >> 8) & 255) - rgb[1]) + abs((v & 255) - rgb[2]) <= tol

    # kaynağın baskın renkleri fixture'dan otomatik örneklenir (hardcode yok):
    # geniş iç bölgelerden en sık 4 renk
    small_img = np.asarray(im.resize((256, 256)))
    px = small_img.reshape(-1, 3)
    uniq, counts = np.unique(px // 24 * 24, axis=0, return_counts=True)
    dominants = [tuple(int(c) for c in uniq[i]) for i in np.argsort(-counts)[:4]]
    for drgb in dominants:
        _fail(errors, any(_near(f, drgb, 90) for f in fills),
              f"kaynak baskın rengi {drgb} için dolgu karşılığı yok: {fills}")
    grays_only = all(
        abs((int(f[1:], 16) >> 16) - ((int(f[1:], 16) >> 8) & 255)) < 12
        and abs(((int(f[1:], 16) >> 8) & 255) - (int(f[1:], 16) & 255)) < 12
        for f in fills
    ) if fills else True
    _fail(errors, not grays_only, f"tüm dolgular gri tonlu — kromatik kaynak için kabul edilemez: {fills}")

    # --- B. Palet + C. bileşen testleri (render karşılaştırması) ----------
    rnd = render_svg_to_rgb(svg_path, w, h)
    _fail(errors, rnd is not None, "SVG render edilemedi")
    if rnd is not None:
        src = np.asarray(im)
        lab_o = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_r = cv2.cvtColor(rnd, cv2.COLOR_RGB2LAB).astype(np.float32)
        samples = lab_o.reshape(-1, 3)
        sub = samples[:: max(1, samples.shape[0] // 60000)]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.4)
        # seed: kmeans deterministik değil; sınıf SAYISI görüntü karakterine
        # göre 4 (bilinen düz-renk fixture). Merkezlere göre eşleme yapılır.
        cv2.setRNGSeed(7)
        _c, _l, centers = cv2.kmeans(sub, 4, None, crit, 4, cv2.KMEANS_PP_CENTERS)

        def classify(lab: np.ndarray) -> np.ndarray:
            dist = np.linalg.norm(lab[:, :, None, :] - centers[None, None, :, :], axis=3)
            return np.argmin(dist, axis=2)

        co, cr = classify(lab_o), classify(lab_r)
        agree = float((co == cr).mean())
        _fail(errors, agree >= MIN_PALETTE_AGREE,
              f"palet uyumu {agree:.4f} < {MIN_PALETTE_AGREE}")
        min_area = max(60, int(0.00005 * w * h))
        worst_small = 1.0
        worst_info = None
        for ci in range(centers.shape[0]):
            mo = (co == ci).astype(np.uint8)
            mr = cr == ci
            inter, uni = int(((mo > 0) & mr).sum()), int(((mo > 0) | mr).sum())
            iou_c = inter / uni if uni else 1.0
            _fail(errors, iou_c >= MIN_CLASS_IOU, f"sınıf {ci} IoU {iou_c:.4f} < {MIN_CLASS_IOU}")
            n, lmap, stats, _ = cv2.connectedComponentsWithStats(mo, 8)
            for i in range(1, n):
                x, y, ww, hh, area = stats[i]
                if area < min_area or max(ww, hh) > 0.2 * max(w, h):
                    continue  # yalnız KÜÇÜK anlamlı bileşenler
                mm = lmap[y:y + hh, x:x + ww] == i
                rr = mr[y:y + hh, x:x + ww]
                near = cv2.dilate(mm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                rr = rr & near
                u2 = int((mm | rr).sum())
                iou = float((mm & rr).sum()) / u2 if u2 else 1.0
                if iou < worst_small:
                    worst_small = iou
                    worst_info = {"bbox": [int(x), int(y), int(ww), int(hh)], "iou": round(iou, 4)}
        _fail(errors, worst_small >= MIN_SMALL_COMPONENT_IOU,
              f"en kötü küçük bileşen IoU {worst_small:.4f} < {MIN_SMALL_COMPONENT_IOU} ({worst_info})")

    summary = {
        "best": best.get("name"), "fidelity": best.get("fidelity_score"),
        "cmds": cmds, "palette_agree": round(agree, 5) if rnd is not None else None,
        "worst_small_component": worst_info if rnd is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.keep:
        print("debug:", job)
    else:
        import shutil
        shutil.rmtree(job, ignore_errors=True)
    if errors:
        print("FAIL:")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
