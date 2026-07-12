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

# Eşikler (ölçülmüş baseline'a göre; gevşetme = hatayı geçirme, yapma).
# Değerler renderer (resvg) anti-alias davranışına bağlıdır: sert sınıf
# ataması iki renk ortasındaki AA piksellerinde 1 birimlik render farkıyla
# oynayabilir; eşikler bu yüzden ölçülen değerin hemen altına konur, üstüne
# değil. Ölçülen güncel değerler (counter_merge + local_refine sonrası):
# palet 0.99865, en kötü sınıf 0.99437, halka 0.9942, en kötü küçük 0.9701.
MIN_PALETTE_AGREE = 0.9975     # genel palet sınıf uyumu (ölçülen 0.99876)
MIN_CLASS_IOU = 0.99           # her palet sınıfı IoU tabanı (ölçülen min 0.99436)
MIN_SMALL_COMPONENT_IOU = 0.975  # en kötü küçük bileşen (ölçülen R 0.9802 — 4x SS)
MIN_RING_IOU = 0.985           # halka-benzeri küçük bileşen (ölçülen 0.9942)
MIN_FIDELITY = 97.9            # kazanan aday sadakat tabanı
MAX_SVG_COMMANDS = 900         # karmaşıklık sınırsız büyümesin (poligon patlaması)
# G bölgesi hedefleri (3840 kaynak uzayında; kullanıcı 1536 ölçeğinde
# raporlar: 2.5x böl): p95 kenar sapması ve yerel palet uyumu
G_REGION = (1625, 1125, 2625, 2650)  # fixture ölçüm bölgesi (yalnız test)
MAX_G_P95_DEV = 1.5            # px @3840 (=0.6 px @1536; ölçülen 1.0)
MAX_G_P99_DEV = 2.5            # px @3840 — robust maks kilidi (ölçülen 2.0);
                               # dağılımın büyük gövdesi 2px altında
MAX_G_MAX_DEV = 9.0            # px @3840 — bilinen sınır: tek keskin kama
                               # tepesi (52px blob; tepe kaynağa 0.2px, keskin
                               # V-çentiğin duvarı stacked boyamada bir pikselde
                               # 7.8px kalıyor). Ham eşik gevşetilMEZ (r36=9.0);
                               # p99≤2.5 gerçek iyileşmeyi ayrıca kilitler
MIN_G_AGREE = 0.9955           # ölçülen 0.99648
MAX_FRAME_THICKNESS_DIFF = 0.5  # px, coverage-ağırlıklı alt-piksel ölçüm


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
    _fail(errors, float(best.get("fidelity_score") or 0.0) >= MIN_FIDELITY,
          f"sadakat {best.get('fidelity_score')} < {MIN_FIDELITY}")
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

    # --- D. Sayaç (counter) sözleşmesi: gerçek evenodd delik + arka plan
    # bağımsızlığı. Halka benzeri küçük bileşen (®) kaynaktan otomatik bulunur
    # (koordinat hardcode YOK); içindeki sayaç ayrı zemin-renkli örtme path'i
    # olamaz — ebeveyne evenodd alt-yolu olarak gömülmeli ve altındaki katman
    # yeniden boyanınca delikte eski renkte sabit leke kalmamalı.
    src_arr = np.asarray(im)
    lab_src = cv2.cvtColor(src_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    dark_mask = (lab_src[:, :, 0] < 40).astype(np.uint8)
    n_d, _lmap_d, stats_d, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    ring_bbox = None
    min_area_d = max(60, int(0.00005 * w * h))
    for i in range(1, n_d):
        x, y, ww, hh, area = stats_d[i]
        if area < min_area_d or max(ww, hh) > 0.2 * max(w, h):
            continue
        if 0.8 < ww / max(1, hh) < 1.25 and area < 0.45 * ww * hh and ww > 60:
            ring_bbox = (int(x), int(y), int(ww), int(hh))
            break
    if ring_bbox:
        from svgpathtools import parse_path as _pp

        rx, ry, rw2, rh2 = ring_bbox

        def _is_dark_fill(f: str) -> bool:
            v = int(f[1:], 16)
            return (v >> 16) + ((v >> 8) & 255) + (v & 255) < 120

        inner_paths = []
        for m in re.finditer(r"<path[^>]*>", svg_txt):
            tag = m.group(0)
            dm = re.search(r'd="([^"]*)"', tag)
            fm = re.search(r'fill="(#[0-9a-fA-F]{6})"', tag)
            if not dm or not fm:
                continue
            try:
                bx0, bx1, by0, by1 = _pp(dm.group(1)).bbox()
            except Exception:  # noqa: BLE001
                continue
            if bx0 >= rx - 2 and bx1 <= rx + rw2 + 2 and by0 >= ry - 2 and by1 <= ry + rh2 + 2:
                subs = len(re.findall(r"(?<![0-9a-zA-Z.,-])[Mm]", " " + dm.group(1)))
                inner_paths.append({"tag": tag, "fill": fm.group(1).lower(), "subs": subs})
        light = [p for p in inner_paths if not _is_dark_fill(p["fill"])]
        _fail(errors, len(light) <= 1,
              f"® içinde {len(light)} zemin-renkli path: sayaç hâlâ örtme (overlay), delik değil")
        holes = [p for p in inner_paths
                 if _is_dark_fill(p["fill"]) and p["subs"] >= 2 and "evenodd" in p["tag"]]
        _fail(errors, len(holes) >= 1,
              "® içinde evenodd delikli koyu path yok: R sayacı gerçek delik değil")
        if len(light) == 1:
            old_fill = light[0]["fill"]
            txt2 = svg_txt.replace(
                light[0]["tag"],
                re.sub(r'fill="#[0-9a-fA-F]{6}"', 'fill="#00a651"', light[0]["tag"], count=1), 1)
            bg_svg = job / "bg_independence.svg"
            bg_svg.write_text(txt2)
            rnd2 = render_svg_to_rgb(bg_svg, w, h)
            # leke denetimi yeniden boyanan path'in GERÇEK dolgu maskesi
            # içinde yapılır (solo render): bbox köşeleri meşru zemin rengi
            # içerir (stacked modelde "disk" halkanın altına dek uzanır),
            # bbox'a bakmak sahte pozitif üretir (ölçüldü: ~6.4k köşe pikseli)
            ld = re.search(r'd="([^"]*)"', light[0]["tag"]).group(1)
            solo = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                    f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#ffffff"/>'
                    f'<path d="{ld}" fill="#000000"/></svg>')
            solo_svg = job / "bg_solo_mask.svg"
            solo_svg.write_text(solo)
            rnd_solo = render_svg_to_rgb(solo_svg, w, h)
            if rnd2 is not None and rnd_solo is not None:
                mask = rnd_solo.astype(np.int32).sum(axis=2) < 200
                v = int(old_fill[1:], 16)
                stain_rgb = np.array([v >> 16, (v >> 8) & 255, v & 255])
                is_old = np.abs(rnd2.astype(np.int32) - stain_rgb).sum(axis=2) < 90
                stain = int((is_old & mask).sum())
                _fail(errors, stain < 20,
                      f"® deliğinde {stain}px eski renkte leke: sayaç arka plana bağımlı (gerçek delik değil)")

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
                # halka-benzeri bileşen (® halkası) için daha yüksek taban:
                # analitik daire refit'i sonrası ölçülen 0.9942
                if 0.8 < ww / max(1, hh) < 1.25 and area < 0.45 * ww * hh and ww > 60:
                    _fail(errors, iou >= MIN_RING_IOU,
                          f"halka bileşeni IoU {iou:.4f} < {MIN_RING_IOU} ({[int(x), int(y)]})")
        _fail(errors, worst_small >= MIN_SMALL_COMPONENT_IOU,
              f"en kötü küçük bileşen IoU {worst_small:.4f} < {MIN_SMALL_COMPONENT_IOU} ({worst_info})")

        # --- E. G bölgesi yerel doğruluk (yüksek eğrilikli iç ayrıntılar) --
        # Bölge koordinatı yalnız TESTTE kullanılır (üretimde harf tanıma yok).
        gx0, gy0, gx1, gy1 = G_REGION
        g_agree = float((co[gy0:gy1, gx0:gx1] == cr[gy0:gy1, gx0:gx1]).mean())
        _fail(errors, g_agree >= MIN_G_AGREE,
              f"G bölgesi palet uyumu {g_agree:.5f} < {MIN_G_AGREE}")
        black_ci = int(np.argmin(centers[:, 0]))  # LAB L* en düşük merkez
        g_src = (co[gy0:gy1, gx0:gx1] == black_ci).astype(np.uint8)
        g_rnd = (cr[gy0:gy1, gx0:gx1] == black_ci).astype(np.uint8)
        se = cv2.Canny(g_src * 255, 50, 150) > 0
        re2 = cv2.Canny(g_rnd * 255, 50, 150) > 0
        if se.any() and re2.any():
            dt = cv2.distanceTransform((~re2).astype(np.uint8), cv2.DIST_L2, 5)
            dt2 = cv2.distanceTransform((~se).astype(np.uint8), cv2.DIST_L2, 5)
            devs = np.concatenate([dt[se], dt2[re2]])
            g_p95 = float(np.percentile(devs, 95))
            g_p99 = float(np.percentile(devs, 99))
            g_max = float(devs.max())
            _fail(errors, g_p95 <= MAX_G_P95_DEV,
                  f"G p95 kenar sapması {g_p95:.2f}px > {MAX_G_P95_DEV}px")
            # robust maks (p99): dağılım kalitesini kilitler. Ham maks tekil
            # keskin kama-tepesi pikselidir (52px blob, tepe kaynağa 0.2px);
            # ham eşik gevşetilMEZ (r36'daki 9.0 korunur) — p99 ayrı kilit.
            _fail(errors, g_p99 <= MAX_G_P99_DEV,
                  f"G p99 kenar sapması {g_p99:.2f}px > {MAX_G_P99_DEV}px")
            _fail(errors, g_max <= MAX_G_MAX_DEV,
                  f"G maks kenar sapması {g_max:.2f}px > {MAX_G_MAX_DEV}px")

        # --- F. Çerçeve alt-piksel kalınlığı (varsa) -----------------------
        # 4 kenara da dokunan koyu bileşen = çerçeve. Kalınlık coverage
        # ağırlıklı ölçülür (ikili maske yarım pikseli göremez).
        has_frame = any(
            stats_d[i][0] == 0 and stats_d[i][1] == 0
            and stats_d[i][0] + stats_d[i][2] == w and stats_d[i][1] + stats_d[i][3] == h
            for i in range(1, n_d)
        )
        if has_frame:
            def _cov_thickness(img: np.ndarray) -> dict[str, float]:
                f32 = img.astype(np.float32)
                # ikili koşudan bant uzunluğu tahmini (üst kenar, orta %50)
                colsel = np.arange(w // 4, 3 * w // 4)
                dark_run = int(np.mean([
                    np.argmax(f32[:, c].sum(axis=1) > 240) or 100 for c in colsel[:64]
                ]))
                band = max(20, min(200, dark_run + 10))
                probe = band + 10

                def _edge(lines: np.ndarray) -> float:
                    neigh = lines[:, band:probe].reshape(-1, 3).mean(axis=0)
                    dv = neigh  # siyah(0,0,0) -> komşu renk ekseni
                    tproj = np.clip((lines @ dv) / float(dv @ dv), 0, 1)
                    return float((1.0 - tproj[:, :band]).sum(axis=1).mean())

                rowsel = np.arange(h // 4, 3 * h // 4)
                return {
                    "top": _edge(f32[:probe, colsel].transpose(1, 0, 2)),
                    "bottom": _edge(f32[::-1][:probe, colsel].transpose(1, 0, 2)),
                    "left": _edge(f32[rowsel, :probe]),
                    "right": _edge(f32[rowsel][:, ::-1][:, :probe]),
                }
            th_s, th_r = _cov_thickness(src), _cov_thickness(rnd)
            for edge in ("top", "bottom", "left", "right"):
                dthick = abs(th_s[edge] - th_r[edge])
                _fail(errors, dthick <= MAX_FRAME_THICKNESS_DIFF,
                      f"çerçeve {edge} kalınlık farkı {dthick:.3f}px > {MAX_FRAME_THICKNESS_DIFF}px "
                      f"(kaynak {th_s[edge]:.2f}, render {th_r[edge]:.2f})")

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
