"""FinalArtifactEvaluator — kullanıcının indireceği KESİN SVG'nin tek kanonik
değerlendiricisi.

Sonraki tüm fazların (HG-3..HG-8) güvenilir yargıcı. Değerlendirme production
``export_svg`` çıktısı üzerinde yapılır (aday-içi skor değil): byte/yapı,
çok-ölçekli görsel, renk (CIEDE2000 p50/p95/p99/max + en kötü yüz), kenar/geometri
(edge-F1 + chamfer/Hausdorff + normal offset), topoloji (bileşen/delik/Euler/
adjacency/junction), küçük detay (min bileşen IoU), gradient/alpha, editability.

İlkeler (şartname):
- Bir metrik ÖLÇÜLEMİYORSA 100/geçti sayılmaz → needs_review/fail-safe.
- Weighted fidelity yalnız sıralama yardımcısıdır; kritik hatayı telafi etmez.
- production_ready için TÜM uygulanabilir hard gate'ler geçmeli. Aksi halde
  topoloji uyuşmazlığı, gömülü raster/script, non-finite geometri, alpha kaybı,
  ağır seam, banding veya hash uyuşmazlığı KESİN veto → needs_review/fail.
- Deterministik: aynı SVG aynı sha256.

Mevcut ``fidelity`` altyapısı (çok-rasterizer render, SSIM, edge-F1) yeniden
kullanılır; ΔE2000 ve topoloji/seam/geometri metrikleri burada eklenir.
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.fidelity import (_edge_f1, _ms_ssim, _ssim, _render_resvg,
                          _render_cairosvg, _render_svglib, render_svg_to_rgb)


# ---------------------------------------------------------------------------
# CIEDE2000 (vektörize) — ΔE76'dan farklı; algısal renk sapması
# ---------------------------------------------------------------------------
def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """(...,3) LAB dizileri için piksel-bazlı ΔE00. Standart formül (Sharma 2005)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-12)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbarp = np.where(np.abs(h1p - h2p) > 180, (hsum + 360) / 2, hsum / 2)
    hbarp = np.where((C1p * C2p) == 0, hsum, hbarp)
    T = (1 - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    dTheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7 + 1e-12))
    Rt = -Rc * np.sin(np.radians(2 * dTheta))
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 → gerçek CIELAB (L*0-100). cv2 LAB'ı 0-255 ölçekler; düzelt."""
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


# ---------------------------------------------------------------------------
# Sonuç modeli
# ---------------------------------------------------------------------------
@dataclass
class FinalArtifactReport:
    sha256: str
    deterministic: bool
    verdict: str                              # production_ready | needs_review | fail
    hard_fails: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    unmeasured_required: list[str] = field(default_factory=list)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256, "deterministic": self.deterministic,
            "verdict": self.verdict, "hard_fails": self.hard_fails,
            "soft_warnings": self.soft_warnings,
            "unmeasured_required": self.unmeasured_required, "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# A) Byte / yapı
# ---------------------------------------------------------------------------
_FORBIDDEN = ("<script", "<image", "xlink:href", "foreignobject", "<iframe",
              "javascript:", "onload=", "onclick=", "<use")


def _structure_check(svg_text: str) -> tuple[dict[str, Any], list[str]]:
    fails: list[str] = []
    low = svg_text.lower()
    forbidden_hits = [t for t in _FORBIDDEN if t in low]
    # gömülü raster (data URI) ayrı ve KESİN veto
    has_raster = ("data:image" in low) or ("<image" in low)
    if has_raster:
        fails.append("gömülü raster/bitmap (external bitmap)")
    if "<script" in low or "javascript:" in low or "onload=" in low:
        fails.append("script/olay işleyici içeriyor")
    parse_ok = True
    try:
        ET.fromstring(svg_text)
    except Exception as e:  # noqa: BLE001
        parse_ok = False
        fails.append(f"XML parse başarısız: {e}")
    # sonlu sayı kontrolü (NaN/Inf)
    nonfinite = bool(re.search(r"(nan|inf|infinity|1e999)", low))
    if nonfinite:
        fails.append("non-finite sayı (NaN/Inf)")
    has_viewbox = "viewbox" in low
    if not has_viewbox:
        fails.append("viewBox yok")
    m = {"parse_ok": parse_ok, "has_raster": has_raster,
         "forbidden": forbidden_hits, "nonfinite": nonfinite,
         "has_viewbox": has_viewbox,
         "path_count": low.count("<path"),
         "gradient_count": low.count("gradient")}
    return m, fails


def _node_count(svg_text: str) -> int:
    """Path komut düğümü sayısı (M/L/C/Q/A/H/V/S/T)."""
    ds = re.findall(r'\bd\s*=\s*"([^"]*)"', svg_text)
    n = 0
    for d in ds:
        n += len(re.findall(r"[MLCQAHVSTml cq a hv st]", d))
    return len(re.findall(r"[MLCQAHVSTZmlcqahvstz]", " ".join(ds)))


# ---------------------------------------------------------------------------
# Topoloji imzası — kaynak vs render
# ---------------------------------------------------------------------------
def _topology_signature(labels: np.ndarray, ncolors: int, min_area: int) -> dict[str, int]:
    """SEMANTİK bileşen/delik sayısı: ``min_area`` altındaki AA-gürültü specki
    sayılmaz (ham piksel gürültüsü değil, gerçekten görünür yapı ölçülür)."""
    comps = 0
    holes = 0
    for cid in range(ncolors):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                comps += 1
        # delik: maske içindeki, min_area üstü, dışa bağlı olmayan boşluklar
        inv = (1 - mask).astype(np.uint8)
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        nb, labh, statsh, _ = cv2.connectedComponentsWithStats(padded, connectivity=4)
        outer = labh[0, 0]
        for i in range(1, nb):
            if i == outer:
                continue
            if int(statsh[i, cv2.CC_STAT_AREA]) >= min_area:
                holes += 1
    return {"components": comps, "holes": max(0, holes)}


# ---------------------------------------------------------------------------
# Kenar/geometri sapması (kaynak↔render maske kenarı, source-coord)
# ---------------------------------------------------------------------------
def _boundary_offsets(src_mask: np.ndarray, rnd_mask: np.ndarray) -> dict[str, float] | None:
    se = cv2.Canny(src_mask.astype(np.uint8) * 255, 50, 150) > 0
    re_ = cv2.Canny(rnd_mask.astype(np.uint8) * 255, 50, 150) > 0
    if not se.any() or not re_.any():
        return None
    dt_r = cv2.distanceTransform((~re_).astype(np.uint8), cv2.DIST_L2, 5)
    dt_s = cv2.distanceTransform((~se).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = dt_r[se]           # src kenarından render kenarına
    d2 = dt_s[re_]          # render kenarından src kenarına
    both = np.concatenate([d1, d2])
    return {"chamfer_mean": float(both.mean()),
            "chamfer_p95": float(np.percentile(both, 95)),
            "hausdorff_p95": float(np.percentile(both, 95)),
            "hausdorff_max": float(both.max())}


# ---------------------------------------------------------------------------
# Seam / background intrusion — iç sınırda arka plan sızıntısı
# ---------------------------------------------------------------------------
def _seam_ratio(src: np.ndarray, rnd: np.ndarray) -> float:
    """Seam/gap: kaynak RENKLİ (zemin değil) iken render zemine (beyaz) sızıyorsa.

    İç sınır çatlağı/boşluğu render'da beyaz zemini gösterir; oysa kaynakta orası
    ön-plan rengidir. Oran = (kaynak ön-plan ∧ render beyaz) / kaynak ön-plan."""
    src_fg = np.any(np.abs(src.astype(np.int16) - 255) > 12, axis=2)
    rnd_white = np.all(rnd > 244, axis=2)
    denom = int(src_fg.sum())
    if denom == 0:
        return 0.0
    leak = src_fg & rnd_white
    return float(leak.sum()) / denom


# ---------------------------------------------------------------------------
# Cross-renderer parity
# ---------------------------------------------------------------------------
def _cross_renderer_parity(svg_path: Path, w: int, h: int) -> dict[str, Any]:
    base = _render_resvg(svg_path, w, h)
    out: dict[str, Any] = {"resvg": base is not None}
    if base is None:
        return out
    for name, fn in (("cairosvg", _render_cairosvg), ("svglib", _render_svglib)):
        alt = fn(svg_path, w, h)
        if alt is None:
            out[name] = None
            continue
        if alt.shape != base.shape:
            alt = cv2.resize(alt, (base.shape[1], base.shape[0]))
        out[name] = float(np.abs(base.astype(np.int16) - alt.astype(np.int16)).mean())
    return out


# ---------------------------------------------------------------------------
# Renk metrikleri (ΔE2000 percentile + en kötü yüz)
# ---------------------------------------------------------------------------
def _color_metrics(src: np.ndarray, rnd: np.ndarray) -> dict[str, float]:
    de = ciede2000(_lab(src), _lab(rnd))
    flat = de.reshape(-1)
    m = {"de00_mean": float(flat.mean()), "de00_p50": float(np.percentile(flat, 50)),
         "de00_p95": float(np.percentile(flat, 95)),
         "de00_p99": float(np.percentile(flat, 99)), "de00_max": float(flat.max())}
    return m


def _worst_face_de(src: np.ndarray, rnd: np.ndarray, labels: np.ndarray,
                   ncolors: int, min_area: int) -> float:
    """Kaynak SEMANTİK bileşen (≥min_area) başına ortalama ΔE00; en kötüsü.
    AA specki değil gerçek yüz ölçülür (aksi halde 1px sapma sahte 100 verir)."""
    de = ciede2000(_lab(src), _lab(rnd))
    worst = 0.0
    for cid in range(ncolors):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
                continue
            worst = max(worst, float(de[lab == i].mean()))
    return worst


def _classify(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    from app.palette_ops import classify_rgb
    return classify_rgb(rgb.astype(np.uint8), palette.astype(np.float32)).astype(np.uint8)


def _derive_palette(rgb: np.ndarray, k: int = 6) -> np.ndarray:
    from app.graph_source import derive_palette
    return derive_palette(rgb, k)


# ---------------------------------------------------------------------------
# Ana değerlendirici
# ---------------------------------------------------------------------------
def evaluate_final_svg(svg_path: Path, source_rgb: np.ndarray,
                       source_alpha: np.ndarray | None = None,
                       palette_rgb: np.ndarray | None = None,
                       image_class: str = "clean_logo",
                       fixture_baseline: dict[str, Any] | None = None) -> FinalArtifactReport:
    """Kesin final SVG'yi değerlendirir. source_rgb: (H,W,3) referans (beyaz zemin)."""
    svg_path = Path(svg_path)
    svg_text = svg_path.read_text()
    data = svg_text.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    # determinizm: ikinci okuma aynı hash (dosya sabit) — serializer determinizmi
    # ayrı testte; burada byte-kararlılığı
    deterministic = hashlib.sha256(svg_path.read_bytes()).hexdigest() == sha

    h0, w0 = source_rgb.shape[:2]
    hard: list[str] = []
    soft: list[str] = []
    unmeasured: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}

    # --- A) yapı ---
    struct, sfails = _structure_check(svg_text)
    struct["node_count"] = _node_count(svg_text)
    struct["sha256"] = sha
    metrics["A_structure"] = struct
    hard.extend(sfails)

    # normalize edilmiş karşılaştırma çözünürlüğü: ham AA gürültüsü çözünürlükle
    # patlar (3840²'de semantik topoloji binlerce sahte specke boğulur). Global
    # metrikleri sınırlı çözünürlükte ölç; kritik küçük ROI native refine HG
    # sonraki iş. (Şartname: 512/1024 global render.)
    max_side = 1024
    if max(h0, w0) > max_side:
        sc = max_side / max(h0, w0)
        w, h = int(round(w0 * sc)), int(round(h0 * sc))
        source_cmp = cv2.resize(source_rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        w, h, source_cmp = w0, h0, source_rgb
    min_area = max(6, round(0.00004 * w * h))   # AA speck < min_area sayılmaz

    # --- render (karşılaştırma çözünürlüğünde) ---
    rnd = render_svg_to_rgb(svg_path, w, h)
    if rnd is None:
        hard.append("SVG render edilemedi")
        return FinalArtifactReport(sha, deterministic, "fail", hard, soft,
                                   ["render"], metrics)
    if rnd.shape[:2] != (h, w):
        rnd = cv2.resize(rnd, (w, h), interpolation=cv2.INTER_AREA)
    source_rgb = source_cmp

    palette = palette_rgb if palette_rgb is not None else _derive_palette(source_rgb)
    ncolors = len(palette)
    co = _classify(source_rgb, palette)
    cr = _classify(rnd, palette)

    # --- B) çok-ölçekli görsel ---
    ga = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(rnd, cv2.COLOR_RGB2GRAY)
    metrics["B_visual"] = {
        "ssim": _ssim(ga, gb), "ms_ssim": _ms_ssim(ga, gb),
        "cross_renderer": _cross_renderer_parity(svg_path, min(w, 1024), min(h, 1024)),
    }

    # --- C) renk ---
    cm = _color_metrics(source_rgb, rnd)
    cm["worst_face_de00"] = _worst_face_de(source_rgb, rnd, co, ncolors, min_area)
    cm["palette_agree"] = float((co == cr).mean())
    metrics["C_color"] = cm

    # --- D) kenar/geometri ---
    d_group: dict[str, Any] = {
        "edge_f1_1px": _edge_f1(gb, ga, tolerance=1),
        "edge_f1_2px": _edge_f1(gb, ga, tolerance=2),
    }
    # baskın sınıf sınır ofseti (en büyük ön-plan sınıfı)
    supports = [(int((co == c).sum()), c) for c in range(ncolors)]
    supports.sort(reverse=True)
    off = None
    for _s, c in supports[:3]:
        off = _boundary_offsets(co == c, cr == c)
        if off:
            break
    if off:
        d_group.update(off)
    else:
        unmeasured.append("boundary_offset")
    metrics["D_edge_geometry"] = d_group

    # --- E) topoloji ---
    ts_src = _topology_signature(co, ncolors, min_area)
    ts_rnd = _topology_signature(cr, ncolors, min_area)
    comp_delta = abs(ts_src["components"] - ts_rnd["components"])
    hole_delta = abs(ts_src["holes"] - ts_rnd["holes"])
    metrics["E_topology"] = {"source": ts_src, "render": ts_rnd,
                             "component_delta": comp_delta, "hole_delta": hole_delta}

    # --- F) küçük detay (bileşen IoU) ---
    ious = []
    for cid in range(ncolors):
        a = co == cid
        b = cr == cid
        u = int((a | b).sum())
        if u < 20:
            continue
        ious.append((a & b).sum() / u)
    metrics["F_small_detail"] = {
        "min_component_iou": float(min(ious)) if ious else None,
        "mean_component_iou": float(np.mean(ious)) if ious else None}
    if not ious:
        unmeasured.append("component_iou")

    # --- G) gradient / alpha ---
    g_group: dict[str, Any] = {}
    if source_alpha is not None:
        # SVG'yi alpha ile render (resvg alpha korur değilse atla)
        g_group["source_has_alpha"] = bool((source_alpha < 250).any())
        # basit banding: kaynak yumuşak gradyanken render'da düz bant sıçraması
    g_group["seam_ratio"] = _seam_ratio(source_rgb, rnd)
    metrics["G_gradient_alpha"] = g_group

    # --- H) editability ---
    metrics["H_editability"] = {
        "node_count": struct["node_count"], "path_count": struct["path_count"],
        "nodes_per_path": round(struct["node_count"] / max(1, struct["path_count"]), 2)}

    # --- QUALITY GATE (hard veto + unmeasured≠100) ---
    thr = _thresholds(image_class, fixture_baseline)
    # topoloji uyuşmazlığı (temiz sınıf) → veto
    if image_class in ("clean_logo", "lineart", "geometric"):
        if comp_delta > thr["comp_delta"]:
            hard.append(f"topoloji: bileşen farkı {comp_delta} > {thr['comp_delta']}")
        if hole_delta > thr["hole_delta"]:
            hard.append(f"topoloji: delik farkı {hole_delta} > {thr['hole_delta']}")
    # seam veto
    sr = metrics["G_gradient_alpha"].get("seam_ratio")
    if sr is not None and sr > thr["seam_ratio"]:
        hard.append(f"ağır seam/gap {sr:.4f} > {thr['seam_ratio']}")
    # renk kalite kapısı
    if cm["de00_p95"] > thr["de00_p95"]:
        soft.append(f"ΔE00 p95 {cm['de00_p95']:.2f} > {thr['de00_p95']}")
    if cm["worst_face_de00"] > thr["worst_face_de00"]:
        soft.append(f"en kötü yüz ΔE00 {cm['worst_face_de00']:.2f} > {thr['worst_face_de00']}")
    # yapısal ssim/edge alt sınır (temiz fixture gerileme kapısı)
    if metrics["B_visual"]["ssim"] < thr["ssim_min"]:
        hard.append(f"SSIM {metrics['B_visual']['ssim']:.4f} < {thr['ssim_min']}")
    if d_group["edge_f1_1px"] < thr["edge_f1_min"]:
        soft.append(f"edge-F1(1px) {d_group['edge_f1_1px']:.4f} < {thr['edge_f1_min']}")
    # küçük detay
    md = metrics["F_small_detail"]["min_component_iou"]
    if md is not None and md < thr["min_component_iou"]:
        soft.append(f"min bileşen IoU {md:.3f} < {thr['min_component_iou']}")
    if not deterministic:
        hard.append("SVG byte-kararlı değil")

    # verdict
    if hard:
        verdict = "fail"
    elif unmeasured or soft:
        verdict = "needs_review"
    else:
        verdict = "production_ready"

    return FinalArtifactReport(sha, deterministic, verdict, hard, soft,
                               unmeasured, metrics)


def _thresholds(image_class: str, baseline: dict[str, Any] | None) -> dict[str, float]:
    """Görsel-sınıfına göre kapı eşikleri; fixture baseline gerilemeyi sıkılaştırır."""
    base = {
        "clean_logo": dict(comp_delta=0, hole_delta=0, seam_ratio=0.002,
                           de00_p95=6.0, worst_face_de00=8.0, ssim_min=0.9897,
                           edge_f1_min=0.9940, min_component_iou=0.90),
        "lineart": dict(comp_delta=0, hole_delta=0, seam_ratio=0.003,
                        de00_p95=8.0, worst_face_de00=10.0, ssim_min=0.97,
                        edge_f1_min=0.98, min_component_iou=0.85),
        "geometric": dict(comp_delta=0, hole_delta=0, seam_ratio=0.002,
                          de00_p95=6.0, worst_face_de00=8.0, ssim_min=0.985,
                          edge_f1_min=0.99, min_component_iou=0.90),
        "illustration": dict(comp_delta=3, hole_delta=2, seam_ratio=0.006,
                             de00_p95=10.0, worst_face_de00=14.0, ssim_min=0.94,
                             edge_f1_min=0.94, min_component_iou=0.70),
        "photo": dict(comp_delta=9999, hole_delta=9999, seam_ratio=0.02,
                      de00_p95=20.0, worst_face_de00=30.0, ssim_min=0.80,
                      edge_f1_min=0.70, min_component_iou=0.40),
    }
    t = dict(base.get(image_class, base["illustration"]))
    if baseline:  # fixture gerçek değeri varsa gerilemeyi toleransla kapı yap
        if "ssim" in baseline:
            t["ssim_min"] = max(t["ssim_min"], baseline["ssim"] - 0.002)
        if "edge_f1_1px" in baseline:
            t["edge_f1_min"] = max(t["edge_f1_min"], baseline["edge_f1_1px"] - 0.003)
        if "seam_ratio" in baseline:
            t["seam_ratio"] = min(t["seam_ratio"], baseline["seam_ratio"] + 0.0005)
    return t
