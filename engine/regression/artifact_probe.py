"""Artefakt stres-testi: kırık çizgi / çizik (seam) / kenarlık / renk hataları.

Vektör çıktının "net" olmasını doğrudan tehdit eden artefakt sınıflarını
hedefleyen sentetik vakalar üretir, pipeline'ı uçtan uca çalıştırır, seçilen
SVG'yi render edip artefakta ÖZGÜ metriklerle ölçer:

* **ink_recall**     — orijinaldeki mürekkep (çizgi/şekil) piksellerinin
                       render'da karşılanma oranı. Düşükse çizgiler KIRIK/kopuk.
* **ink_precision**  — render'daki mürekkebin orijinalde karşılığı. Düşükse
                       hayalet çizik/leke var demektir.
* **component_delta**— bağlı bileşen sayısı farkı (render - orijinal). Pozitifse
                       şekiller PARÇALANMIŞ, negatifse birbirine yapışmış.
* **seam_ratio**     — bitişik renk bölgeleri arasında zemin renginin sızdığı
                       (hairline gap/çizik) piksellerin iç bölgeye oranı.
* **halo_ratio**     — beklenen palet dışına düşen (anti-alias halo bandı gibi)
                       piksellerin oranı. Yüksekse renk kirliliği var.
* **fidelity**       — genel algısal sadakat (SSIM + ΔE + kenar F1).

Kullanım::

    .venv/bin/python regression/artifact_probe.py             # tüm vakalar
    .venv/bin/python regression/artifact_probe.py --case thin_lines
    .venv/bin/python regression/artifact_probe.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR))

from app.fidelity import compute_fidelity, render_svg_to_rgb  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402

WHITE = (255, 255, 255, 255)


# ---------------------------------------------------------------------------
# Vaka üreteçleri — her biri (image, meta) döndürür
# ---------------------------------------------------------------------------
def make_thin_lines(path: Path) -> dict[str, Any]:
    """1-3 px çizgiler: kırılma/kopma en çok burada görülür."""
    scale = 3
    img = Image.new("RGBA", (640 * scale, 400 * scale), WHITE)
    d = ImageDraw.Draw(img)
    s = scale
    # yatay/dikey/diyagonal ince çizgiler (1..3 px nihai kalınlık)
    for i, w in enumerate((1, 2, 3)):
        y = (60 + i * 40) * s
        d.line((40 * s, y, 600 * s, y), fill=(0, 0, 0, 255), width=w * s)
    for i, w in enumerate((1, 2, 3)):
        x = (80 + i * 50) * s
        d.line((x, 200 * s, x, 370 * s), fill=(0, 0, 0, 255), width=w * s)
    d.line((260 * s, 370 * s, 560 * s, 210 * s), fill=(0, 0, 0, 255), width=2 * s)
    # renkli ince çizgiler (anti-alias'ta pembeleşme/kaybolma testi)
    d.line((40 * s, 385 * s, 600 * s, 385 * s), fill=(255, 0, 0, 255), width=2 * s)
    d.line((300 * s, 200 * s, 300 * s, 370 * s), fill=(0, 80, 200, 255), width=2 * s)
    img = img.resize((640, 400), Image.Resampling.LANCZOS)
    img.save(path)
    return {"trace_mode": "auto", "expect_palette": [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 80, 200)]}


def make_border_frames(path: Path) -> dict[str, Any]:
    """İç içe ince kenarlıklar: kenarlık kopması/incelme testi."""
    scale = 3
    img = Image.new("RGBA", (600 * scale, 440 * scale), WHITE)
    d = ImageDraw.Draw(img)
    s = scale
    d.rectangle((16 * s, 16 * s, 584 * s, 424 * s), outline=(255, 0, 0, 255), width=6 * s)
    d.rectangle((44 * s, 44 * s, 556 * s, 396 * s), outline=(0, 0, 0, 255), width=3 * s)
    d.rectangle((68 * s, 68 * s, 532 * s, 372 * s), outline=(0, 0, 0, 255), width=2 * s)
    d.rounded_rectangle((110 * s, 110 * s, 490 * s, 330 * s), radius=36 * s,
                        outline=(0, 0, 0, 255), width=4 * s)
    img = img.resize((600, 440), Image.Resampling.LANCZOS)
    img.save(path)
    return {"trace_mode": "auto", "expect_palette": [(0, 0, 0), (255, 255, 255), (255, 0, 0)]}


def make_adjacent_colors(path: Path) -> dict[str, Any]:
    """Bitişik düz renk bölgeleri: hairline seam (beyaz sızma/çizik) testi."""
    scale = 3
    img = Image.new("RGBA", (600 * scale, 400 * scale), WHITE)
    d = ImageDraw.Draw(img)
    s = scale
    # birbirine tam bitişik dikey şeritler
    stripes = [(214, 40, 40), (255, 152, 0), (76, 140, 74), (33, 118, 174), (103, 58, 143)]
    x0, y0, y1 = 60, 50, 200
    wst = 96
    for i, c in enumerate(stripes):
        d.rectangle(((x0 + i * wst) * s, y0 * s, (x0 + (i + 1) * wst) * s, y1 * s),
                    fill=(*c, 255))
    # bitişik pasta dilimleri
    cx, cy, r = 300, 305, 85
    seg = [(214, 40, 40), (255, 152, 0), (76, 140, 74), (33, 118, 174)]
    a = 0
    for i, c in enumerate(seg):
        d.pieslice(((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s),
                   a, a + 90, fill=(*c, 255))
        a += 90
    img = img.resize((600, 400), Image.Resampling.LANCZOS)
    img.save(path)
    return {
        "trace_mode": "auto",
        "expect_palette": [(255, 255, 255)] + stripes,
        "seam_bg": (255, 255, 255),
    }


def make_curves_smooth(path: Path) -> dict[str, Any]:
    """Daire/halka/oval: merdivenlenme ve eğri pürüzü testi."""
    scale = 3
    img = Image.new("RGBA", (600 * scale, 400 * scale), WHITE)
    d = ImageDraw.Draw(img)
    s = scale
    d.ellipse((50 * s, 60 * s, 270 * s, 280 * s), fill=(0, 0, 0, 255))
    d.ellipse((100 * s, 110 * s, 220 * s, 230 * s), fill=WHITE)
    d.ellipse((320 * s, 60 * s, 560 * s, 220 * s), outline=(214, 40, 40, 255), width=8 * s)
    d.ellipse((330 * s, 260 * s, 430 * s, 360 * s), fill=(33, 118, 174, 255))
    img = img.resize((600, 400), Image.Resampling.LANCZOS)
    img.save(path)
    return {"trace_mode": "auto",
            "expect_palette": [(0, 0, 0), (255, 255, 255), (214, 40, 40), (33, 118, 174)]}


def make_small_glyphs(path: Path) -> dict[str, Any]:
    """Küçük glif benzeri şekiller: detay kaybı/parçalanma testi."""
    scale = 3
    img = Image.new("RGBA", (600 * scale, 300 * scale), WHITE)
    d = ImageDraw.Draw(img)
    s = scale
    x = 40
    for size in (14, 18, 24, 32):
        # 'E' benzeri
        d.rectangle((x * s, 60 * s, (x + 4) * s, (60 + size) * s), fill=(0, 0, 0, 255))
        for dy in (0, size // 2 - 2, size - 4):
            d.rectangle((x * s, (60 + dy) * s, (x + size - 2) * s, (60 + dy + 4) * s),
                        fill=(0, 0, 0, 255))
        # '+' işareti
        cxx = x + size // 2
        d.rectangle(((cxx - 2) * s, 160 * s, (cxx + 2) * s, (160 + size) * s), fill=(0, 0, 0, 255))
        d.rectangle(((x) * s, (160 + size // 2 - 2) * s, (x + size) * s, (160 + size // 2 + 2) * s),
                    fill=(0, 0, 0, 255))
        x += size + 40
    img = img.resize((600, 300), Image.Resampling.LANCZOS)
    img.save(path)
    return {"trace_mode": "auto", "expect_palette": [(0, 0, 0), (255, 255, 255)]}


CASES: dict[str, Callable[[Path], dict[str, Any]]] = {
    "thin_lines": make_thin_lines,
    "border_frames": make_border_frames,
    "adjacent_colors": make_adjacent_colors,
    "curves_smooth": make_curves_smooth,
    "small_glyphs": make_small_glyphs,
}


# ---------------------------------------------------------------------------
# Artefakt metrikleri
# ---------------------------------------------------------------------------
def _ink_mask(rgb: np.ndarray, bg: tuple[int, int, int] = (255, 255, 255), tol: int = 60) -> np.ndarray:
    """Zeminden belirgin ayrışan (mürekkep) pikselleri işaretler."""
    diff = np.abs(rgb.astype(np.int32) - np.array(bg, dtype=np.int32))
    return (diff.max(axis=2) > tol).astype(np.uint8)


def ink_metrics(original: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    """Kırık çizgi / hayalet çizik metrikleri (1px tolerans)."""
    mo = _ink_mask(original)
    mr = _ink_mask(rendered)
    k = np.ones((3, 3), np.uint8)
    mo_d = cv2.dilate(mo, k)
    mr_d = cv2.dilate(mr, k)
    recall = float((mo & mr_d).sum()) / max(1.0, float(mo.sum()))
    precision = float((mr & mo_d).sum()) / max(1.0, float(mr.sum()))
    n_o, _ = cv2.connectedComponents(mo, connectivity=8)
    n_r, _ = cv2.connectedComponents(mr, connectivity=8)
    return {
        "ink_recall": round(recall, 4),
        "ink_precision": round(precision, 4),
        "components_original": int(n_o - 1),
        "components_rendered": int(n_r - 1),
        "component_delta": int(n_r - n_o),
    }


def seam_ratio(original: np.ndarray, rendered: np.ndarray,
               bg: tuple[int, int, int] = (255, 255, 255), tol: int = 40) -> float:
    """Orijinalde mürekkep OLAN yerde render'ın zemine döndüğü piksel oranı.

    Bitişik renk bölgeleri arasındaki hairline beyaz sızmaları (çizik) yakalar.
    Kenar anti-alias'ından etkilenmemek için orijinal mürekkep maskesi 1px
    erozyona uğratılır (yalnız bölge içleri sayılır).
    """
    mo = _ink_mask(original, bg=bg, tol=tol)
    mo_in = cv2.erode(mo, np.ones((3, 3), np.uint8))
    if mo_in.sum() == 0:
        return 0.0
    diff = np.abs(rendered.astype(np.int32) - np.array(bg, dtype=np.int32))
    bg_like = (diff.max(axis=2) <= 12).astype(np.uint8)
    leak = (mo_in & bg_like).sum()
    return round(float(leak) / float(mo_in.sum()), 5)


def halo_ratio(rendered: np.ndarray, palette: list[tuple[int, int, int]], tol: float = 40.0) -> float:
    """Beklenen palete uzak kalan İÇ BÖLGE piksellerinin oranı (renk kirliliği).

    Sınır anti-aliasing'i her render'da (orijinal rasterde bile) palet dışı
    piksel üretir ve kusur değildir; ölçüm kenar komşuluğu (5x5 dilate Canny)
    DIŞINDA yapılır. Böylece yalnız gerçek iç renk bantları/halo lekeleri
    sayılır; daha detaylı (daha uzun sınırlı) çıktılar cezalandırılmaz.
    """
    flat = rendered.reshape(-1, 3).astype(np.float32)
    pal = np.array(palette, dtype=np.float32)
    d2 = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    mind = np.sqrt(d2.min(axis=1)).reshape(rendered.shape[:2])
    off = mind > tol
    gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    near_edge = cv2.dilate(
        (cv2.Canny(gray, 60, 150) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    interior_off = off & ~near_edge
    return round(float(interior_off.mean()), 5)


# ---------------------------------------------------------------------------
# Vaka çalıştırma
# ---------------------------------------------------------------------------
def run_case(name: str, out_dir: Path, keep: bool = False) -> dict[str, Any]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / f"{name}.png"
    meta = CASES[name](input_path)

    image = Image.open(input_path).convert("RGBA")
    result = run_pipeline(image=image, original_path=input_path,
                          trace_mode=meta.get("trace_mode", "auto"), job_dir=case_dir)
    best = result.get("best")
    if not best:
        return {"case": name, "ok": False, "error": "no candidate"}

    original = np.asarray(Image.open(input_path).convert("RGB"))
    h, w = original.shape[:2]
    rendered = render_svg_to_rgb(best["svg_path"], w, h)
    if rendered is None:
        return {"case": name, "ok": False, "error": "render failed"}

    fid = compute_fidelity(original, rendered)
    ink = ink_metrics(original, rendered)
    seam = seam_ratio(original, rendered, bg=meta.get("seam_bg", (255, 255, 255)))
    halo = halo_ratio(rendered, meta["expect_palette"]) if meta.get("expect_palette") else None

    if keep:
        Image.fromarray(rendered).save(case_dir / f"{name}_render.png")

    return {
        "case": name,
        "ok": True,
        "mode_used": result["mode_used"],
        "best": best["name"],
        "selection_reason": result["selection_reason"],
        "fidelity": fid["fidelity_score"],
        "ssim": fid["ssim"],
        "delta_e": fid["mean_delta_e"],
        "edge_f1": fid["edge_f1"],
        **ink,
        "seam_ratio": seam,
        "halo_ratio": halo,
        "svg": str(best["svg_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Artefakt stres-testi (kırık çizgi/çizik/kenarlık/renk)")
    parser.add_argument("--case", action="append", help="Yalnız seçilen vaka(lar)")
    parser.add_argument("--json", help="Sonuçları JSON olarak yaz")
    parser.add_argument("--out-dir", help="Çıktı klasörü (varsayılan: geçici)")
    parser.add_argument("--keep", action="store_true", help="Render PNG'lerini sakla")
    args = parser.parse_args()

    names = args.case or list(CASES)
    tmp_ctx = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(tmp_ctx.name)

    rows = []
    for name in names:
        if name not in CASES:
            print(f"bilinmeyen vaka: {name}", file=sys.stderr)
            return 2
        r = run_case(name, out_dir, keep=args.keep)
        rows.append(r)
        if not r.get("ok"):
            print(f"[FAIL] {name}: {r.get('error')}")
            continue
        print(f"=== {name}  (mode={r['mode_used']}, best={r['best']}, {r['selection_reason']})")
        print(f"    fidelity={r['fidelity']:.1f}  ssim={r['ssim']:.3f}  dE={r['delta_e']:.2f}  edgeF1={r['edge_f1']:.3f}")
        print(f"    ink_recall={r['ink_recall']:.4f}  ink_precision={r['ink_precision']:.4f}"
              f"  comps {r['components_original']}->{r['components_rendered']} (d={r['component_delta']:+d})")
        halo = "-" if r["halo_ratio"] is None else f"{r['halo_ratio']:.5f}"
        print(f"    seam_ratio={r['seam_ratio']:.5f}  halo_ratio={halo}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if tmp_ctx:
        tmp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
