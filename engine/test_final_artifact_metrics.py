"""FinalArtifactEvaluator birim regresyonları — kesin final SVG yargıcı.

Kritik: görsel olarak iyi ama HARD kusurlu SVG (gömülü raster / script /
non-finite / topoloji uyuşmazlığı / ağır seam) production_ready OLMAMALI
(candidate=100 / geometry=0 regresyonu). Ölçülemeyen zorunlu metrik ≠ 100.
Deterministik sha256. CIEDE2000 doğruluğu.

Çalıştırma::  .venv/bin/python test_final_artifact_metrics.py   (~20 sn)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

PAL2 = np.array([[255, 255, 255], [227, 0, 11]], np.uint8)


def check(cond: bool, msg: str) -> None:
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise AssertionError(msg)


def _svg(body: str, w=200, h=200) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{body}</svg>')


def _write(svg: str) -> Path:
    p = Path(tempfile.mktemp(suffix=".svg"))
    p.write_text(svg)
    return p


def _square_src(n=200):
    src = np.full((n, n, 3), 255, np.uint8)
    src[40:160, 40:160] = (227, 0, 11)
    return src


def _donut_src(n=200):
    """Kare halka (annulus): dış kırmızı kare − iç beyaz kare. cv2 ve SVG
    dikdörtgenleri BİREBİR aynı rasterleşir → doğru-topoloji testi kesin eşleşir."""
    src = np.full((n, n, 3), 255, np.uint8)
    src[40:160, 40:160] = (227, 0, 11)
    src[80:120, 80:120] = (255, 255, 255)   # gerçek delik (counter)
    return src


def _eval(svg, src, **kw):
    from app.final_artifact_evaluator import evaluate_final_svg
    return evaluate_final_svg(_write(svg), src, palette_rgb=kw.pop("pal", PAL2), **kw)


def test_ciede2000_correctness() -> None:
    print("== CIEDE2000: aynı renk 0, farklı renk büyük ==")
    from app.final_artifact_evaluator import ciede2000, _lab
    z = np.zeros((2, 2, 3), np.float32)
    check(float(ciede2000(z, z).max()) < 1e-6, "aynı LAB → ΔE00 0")
    black = _lab(np.zeros((1, 1, 3), np.uint8))
    white = _lab(np.full((1, 1, 3), 255, np.uint8))
    check(float(ciede2000(black, white)[0, 0]) > 90, "siyah-beyaz ΔE00 büyük")


def test_clean_svg_production_ready() -> None:
    print("== Temiz eşleşen SVG → production_ready ==")
    src = _square_src()
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<rect x="40" y="40" width="120" height="120" fill="#e3000b"/>'), src)
    check(rep.verdict == "production_ready", f"verdict={rep.verdict} ({rep.hard_fails}{rep.soft_warnings})")
    check(rep.metrics["C_color"]["de00_p95"] < 1.0, "ΔE00 p95 çok düşük")
    check(rep.metrics["E_topology"]["hole_delta"] == 0, "topoloji delik eşleşti")


def test_embedded_raster_veto() -> None:
    print("== Gömülü raster (candidate=100/geometry=0): HARD veto ==")
    src = _square_src()
    # görsel olarak MÜKEMMEL kopya ama gömülü bitmap → asla production_ready
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<rect x="40" y="40" width="120" height="120" fill="#e3000b"/>'
                     '<image href="data:image/png;base64,iVBORw0KGgo=" width="1" height="1"/>'), src)
    check(rep.verdict == "failed", f"gömülü raster failed ({rep.verdict})")
    check(any("raster" in f for f in rep.hard_fails), "raster hard-fail gerekçesi")


def test_script_veto() -> None:
    print("== Script/olay işleyici → HARD veto ==")
    src = _square_src()
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<rect x="40" y="40" width="120" height="120" fill="#e3000b"/>'
                     '<script>alert(1)</script>'), src)
    check(rep.verdict == "failed", f"script failed ({rep.verdict})")


def test_nonfinite_veto() -> None:
    print("== Non-finite geometri → HARD veto ==")
    src = _square_src()
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<path d="M40 40 L NaN 40 L160 160 Z" fill="#e3000b"/>'), src)
    check(rep.verdict == "failed", f"NaN failed ({rep.verdict})")
    check(any("finite" in f.lower() or "nan" in f.lower() for f in rep.hard_fails),
          "non-finite gerekçesi")


def test_topology_mismatch_veto() -> None:
    print("== Topoloji uyuşmazlığı (delik kayıp) → HARD veto ==")
    src = _donut_src()   # kare halka, gerçek delik (counter)
    # DOLU kare (delik yok) — renk/SSIM yüksekçe ama topoloji yanlış
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<rect x="40" y="40" width="120" height="120" fill="#e3000b"/>'), src,
                image_class="clean_logo")
    check(rep.verdict == "failed", f"delik kaybı failed ({rep.verdict})")
    check(rep.metrics["E_topology"]["hole_delta"] >= 1, "delik farkı tespit edildi")
    check(any("delik" in f or "topoloji" in f for f in rep.hard_fails), "topoloji gerekçesi")


def test_donut_correct_production_ready() -> None:
    print("== Doğru delikli kare-halka SVG (evenodd) → production_ready ==")
    src = _donut_src()
    # dış kare + iç kare evenodd hole (rect'ler cv2 ile birebir eşleşir)
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<path fill="#e3000b" fill-rule="evenodd" '
                     'd="M40 40 H160 V160 H40 Z M80 80 H120 V120 H80 Z"/>'), src,
                image_class="clean_logo")
    check(rep.metrics["E_topology"]["hole_delta"] == 0, "kare-halka delik eşleşti")
    check(not rep.hard_fails, f"hard-fail yok ({rep.hard_fails})")
    check(rep.verdict == "production_ready", f"production_ready ({rep.verdict}: {rep.soft_warnings})")


def test_unmeasured_not_100() -> None:
    print("== Ölçülemeyen zorunlu metrik → needs_review (100 sayılmaz) ==")
    from app.final_artifact_evaluator import evaluate_final_svg
    # tek renk (ön-plan yok) → bileşen IoU ölçülemez gibi sınır; en azından
    # unmeasured listesi doldukça verdict production_ready olmamalı
    src = np.full((80, 80, 3), 255, np.uint8)
    rep = evaluate_final_svg(_write(_svg('<rect width="80" height="80" fill="#ffffff"/>', 80, 80)),
                             src, palette_rgb=PAL2, image_class="clean_logo")
    # ön-plan yok → seam 0, ama bileşen IoU tek sınıf → ölçüm sınırlı
    check(rep.verdict != "production_ready" or not rep.unmeasured_required,
          "unmeasured varsa production_ready değil")


def test_deterministic_hash() -> None:
    print("== Deterministik sha256: aynı SVG aynı hash ==")
    from app.final_artifact_evaluator import evaluate_final_svg
    svg = _svg('<rect width="200" height="200" fill="#ffffff"/>'
               '<rect x="40" y="40" width="120" height="120" fill="#e3000b"/>')
    src = _square_src()
    p = _write(svg)
    r1 = evaluate_final_svg(p, src, palette_rgb=PAL2)
    r2 = evaluate_final_svg(p, src, palette_rgb=PAL2)
    check(r1.sha256 == r2.sha256, "iki değerlendirme aynı sha256")
    check(r1.byte_read_stable, "tek artifact okuması byte-kararlı")
    check(r1.deterministic is None, "iki bağımsız pipeline olmadan deterministic iddiası yok")
    import hashlib
    check(r1.sha256 == hashlib.sha256(p.read_bytes()).hexdigest(),
          "sha256 KESİN dosya baytlarının (stale değil)")


def test_highres_semantic_topology_no_false_veto() -> None:
    print("== Yüksek çöz. AA gürültüsü SAHTE topoloji veto'su yapmamalı ==")
    # 2048² kaynak: 4 ayrı kare + 2 kare-halka; eşleşen SVG. Ham piksel topolojisi
    # AA ile binlerce speck üretir; normalize+min_area SEMANTİK sayıyı korur.
    n = 2048
    src = np.full((n, n, 3), 255, np.uint8)
    body = ['<rect width="2048" height="2048" fill="#ffffff"/>']
    boxes = [(200, 200), (1200, 200), (200, 1200), (1200, 1200)]
    for (x, y) in boxes[:2]:
        src[y:y + 500, x:x + 500] = (227, 0, 11)
        body.append(f'<rect x="{x}" y="{y}" width="500" height="500" fill="#e3000b"/>')
    for (x, y) in boxes[2:]:  # kare-halka (delik)
        src[y:y + 500, x:x + 500] = (0, 0, 0)
        src[y + 180:y + 320, x + 180:x + 320] = (255, 255, 255)
        body.append(f'<path fill="#000000" fill-rule="evenodd" '
                    f'd="M{x} {y} h500 v500 h-500 Z '
                    f'M{x + 180} {y + 180} h140 v140 h-140 Z"/>')
    pal = np.array([[255, 255, 255], [227, 0, 11], [0, 0, 0]], np.uint8)
    rep = _eval(_svg("".join(body), n, n), src, pal=pal, image_class="clean_logo")
    ts = rep.metrics["E_topology"]
    check(ts["render"]["components"] <= 8,
          f"semantik bileşen makul (gürültü değil): {ts['render']['components']}")
    check(ts["component_delta"] == 0, f"bileşen delta 0 ({ts['component_delta']})")
    check(ts["hole_delta"] == 0, f"delik delta 0 ({ts['hole_delta']})")
    check(not rep.hard_fails, f"sahte veto yok ({rep.hard_fails})")
    check(rep.verdict == "production_ready", f"production_ready ({rep.verdict})")


def test_seam_gap_veto() -> None:
    print("== Ağır seam/gap (iç boşluk beyaz sızıntı) → veto ==")
    src = _square_src()
    # kırmızı kareyi İKİYE böl, arada 6px beyaz boşluk bırak (seam)
    rep = _eval(_svg('<rect width="200" height="200" fill="#ffffff"/>'
                     '<rect x="40" y="40" width="57" height="120" fill="#e3000b"/>'
                     '<rect x="103" y="40" width="57" height="120" fill="#e3000b"/>'), src)
    sr = rep.metrics["G_gradient_alpha"]["seam_ratio"]
    check(sr > 0.002, f"seam oranı tespit edildi ({sr:.4f})")
    check(rep.verdict == "failed", f"ağır seam failed ({rep.verdict})")


def main() -> int:
    test_ciede2000_correctness()
    test_clean_svg_production_ready()
    test_embedded_raster_veto()
    test_script_veto()
    test_nonfinite_veto()
    test_topology_mismatch_veto()
    test_donut_correct_production_ready()
    test_unmeasured_not_100()
    test_deterministic_hash()
    test_highres_semantic_topology_no_false_veto()
    test_seam_gap_veto()
    print("=" * 60)
    print("SONUC: tum kontroller gecti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
