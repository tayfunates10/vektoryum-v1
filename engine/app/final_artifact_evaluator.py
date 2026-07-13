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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from defusedxml import ElementTree as SafeET

from app.fidelity import (_edge_f1, _ms_ssim, _ssim, _render_resvg,
                          _render_cairosvg, _render_resvg_py, _render_svglib,
                          render_svg_to_rgb)


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
    byte_read_stable: bool
    verdict: str                              # production_ready | needs_review | failed
    hard_fails: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    unmeasured_required: list[str] = field(default_factory=list)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    hard_fail_codes: list[str] = field(default_factory=list)
    soft_warning_codes: list[str] = field(default_factory=list)
    # Serializer determinizmi ancak iki bağımsız pipeline koşusuyla kanıtlanır.
    # Tek artifact değerlendirmesinde bu alan bilinçli olarak None kalır.
    deterministic: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256, "byte_read_stable": self.byte_read_stable,
            "deterministic": self.deterministic,
            "verdict": self.verdict, "hard_fails": self.hard_fails,
            "soft_warnings": self.soft_warnings,
            "unmeasured_required": self.unmeasured_required, "metrics": self.metrics,
            "hard_fail_codes": self.hard_fail_codes,
            "soft_warning_codes": self.soft_warning_codes,
        }


# ---------------------------------------------------------------------------
# A) Byte / yapı
# ---------------------------------------------------------------------------
_DANGEROUS_TAGS = {
    "script", "image", "feimage", "foreignobject", "iframe", "object", "embed",
    "animate", "animatetransform", "animatemotion", "set", "mpath",
}
_NUMERIC_ATTRS = {
    "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "width", "height", "stroke-width", "stroke-dashoffset", "opacity",
    "fill-opacity", "stroke-opacity", "offset",
}
# Yalnız geometri/numeric attribute'larında uygulanır; SVG path komutu ile sayı
# arasında boşluk zorunlu olmadığından ``LNaN`` de kesin yakalanmalıdır.
_NONFINITE_TOKEN = re.compile(r"(?:nan|[-+]?inf(?:inity)?)", re.I)
_PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_URL_REF = re.compile(r"url\(\s*#([^)\s]+)\s*\)", re.I)
_URL_ANY = re.compile(r"url\(\s*([^)]*?)\s*\)", re.I)


def _local_name(name: str) -> str:
    """XML QName/namespace attribute'ünden local-name döndürür."""
    return name.rsplit("}", 1)[-1] if "}" in name else name.split(":")[-1]


def _add_failure(fails: list[str], codes: list[str], code: str, message: str) -> None:
    if code not in codes:
        codes.append(code)
        fails.append(message)


def _structure_check(svg_bytes: bytes) -> tuple[dict[str, Any], list[str], list[str], Any | None]:
    """Ham SVG baytlarını güvenli parse eder ve exact XML metriklerini çıkarır.

    Sayaçlar string-substring üzerinden değil parse edilmiş element ağacından
    gelir. DTD/entity çözümleme defusedxml tarafından fail-closed engellenir.
    """
    fails: list[str] = []
    codes: list[str] = []
    root = None
    try:
        root = SafeET.fromstring(svg_bytes)
    except Exception as e:  # noqa: BLE001
        _add_failure(fails, codes, "xml_parse_failed", f"XML parse başarısız: {e}")

    metrics: dict[str, Any] = {
        "parse_ok": root is not None,
        "root_is_svg": False,
        "has_raster": False,
        "forbidden": [],
        "nonfinite": False,
        "has_viewbox": False,
        "viewbox_valid": False,
        "path_count": 0,
        "node_count": 0,
        "linear_gradient_count": 0,
        "radial_gradient_count": 0,
        "mesh_gradient_count": 0,
        "gradient_definition_count": 0,
        "gradient_reference_count": 0,
    }
    if root is None:
        metrics["gradient_count"] = 0
        return metrics, fails, codes, None

    root_name = _local_name(str(root.tag)).lower()
    metrics["root_is_svg"] = root_name == "svg"
    if root_name != "svg":
        _add_failure(fails, codes, "root_not_svg", "Kök element <svg> değil")

    vb = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    metrics["has_viewbox"] = vb is not None
    if vb is None:
        _add_failure(fails, codes, "viewbox_missing", "viewBox yok")
    else:
        try:
            parts = [float(x) for x in re.split(r"[\s,]+", vb.strip()) if x]
            valid = (len(parts) == 4 and all(math.isfinite(x) for x in parts)
                     and parts[2] > 0 and parts[3] > 0)
        except (TypeError, ValueError):
            valid = False
        metrics["viewbox_valid"] = valid
        if not valid:
            _add_failure(fails, codes, "viewbox_invalid", "viewBox geçersiz veya sonlu değil")

    paths = []
    gradient_ids: set[str] = set()
    for element in root.iter():
        tag = _local_name(str(element.tag)).lower()
        if tag in _DANGEROUS_TAGS:
            metrics["forbidden"].append(tag)
            if tag == "image":
                metrics["has_raster"] = True
                _add_failure(fails, codes, "embedded_raster", "gömülü/dış raster bitmap içeriyor")
            else:
                _add_failure(fails, codes, f"forbidden_{tag}", f"yasak SVG elementi: {tag}")
        if tag == "path":
            paths.append(element)
        elif tag == "lineargradient":
            metrics["linear_gradient_count"] += 1
            if element.attrib.get("id"):
                gradient_ids.add(element.attrib["id"])
        elif tag == "radialgradient":
            metrics["radial_gradient_count"] += 1
            if element.attrib.get("id"):
                gradient_ids.add(element.attrib["id"])
        elif tag == "meshgradient":
            metrics["mesh_gradient_count"] += 1
            if element.attrib.get("id"):
                gradient_ids.add(element.attrib["id"])

        for raw_name, raw_value in element.attrib.items():
            name = _local_name(raw_name).lower()
            value = str(raw_value).strip()
            low_value = value.lower()
            if name.startswith("on"):
                _add_failure(fails, codes, "event_handler", "SVG olay işleyici attribute içeriyor")
            if name in {"href", "src"}:
                # Yalnız aynı belge içi fragment referansı güvenlidir.
                if value and not value.startswith("#"):
                    code = "embedded_raster" if low_value.startswith("data:image") else "external_reference"
                    message = ("gömülü raster bitmap içeriyor" if code == "embedded_raster"
                               else "dış/tehlikeli kaynak referansı içeriyor")
                    if code == "embedded_raster":
                        metrics["has_raster"] = True
                    _add_failure(fails, codes, code, message)
            if "javascript:" in low_value or low_value.startswith("data:"):
                _add_failure(fails, codes, "unsafe_uri", "tehlikeli URI içeriyor")
            for url_target in _URL_ANY.findall(value):
                target = url_target.strip().strip("\"'")
                if target and not target.startswith("#"):
                    _add_failure(
                        fails, codes, "external_reference",
                        "CSS/SVG url() dış veya tehlikeli kaynak içeriyor",
                    )
            if name in _NUMERIC_ATTRS and _NONFINITE_TOKEN.search(value):
                metrics["nonfinite"] = True
                _add_failure(fails, codes, "nonfinite_geometry", "non-finite sayı (NaN/Inf)")
            if name in {"d", "points", "transform", "viewbox"} and _NONFINITE_TOKEN.search(value):
                metrics["nonfinite"] = True
                _add_failure(fails, codes, "nonfinite_geometry", "non-finite sayı (NaN/Inf)")

        # Inline CSS/text de renderer'ın ağ veya aktif içerik çözmesine neden
        # olamaz. Yalnız aynı belge içi url(#id) referanslarına izin verilir.
        text_value = str(element.text or "")
        low_text = text_value.lower()
        if "javascript:" in low_text or "@import" in low_text or "data:" in low_text:
            _add_failure(fails, codes, "unsafe_css", "tehlikeli inline CSS/metin içeriyor")
        for url_target in _URL_ANY.findall(text_value):
            target = url_target.strip().strip("\"'")
            if target and not target.startswith("#"):
                _add_failure(
                    fails, codes, "external_reference",
                    "Inline CSS dış veya tehlikeli url() içeriyor",
                )

    # Referanslar tanımlardan önce gelebilir; ikinci geçiş exact sayar.
    gradient_refs = 0
    for element in root.iter():
        for value in element.attrib.values():
            gradient_refs += sum(1 for ref in _URL_REF.findall(str(value)) if ref in gradient_ids)

    metrics["path_count"] = len(paths)
    metrics["node_count"] = sum(len(_PATH_COMMAND.findall(p.attrib.get("d", ""))) for p in paths)
    metrics["gradient_definition_count"] = (
        metrics["linear_gradient_count"] + metrics["radial_gradient_count"]
        + metrics["mesh_gradient_count"]
    )
    metrics["gradient_reference_count"] = gradient_refs
    # Geriye uyumlu alan: artık substring değil gerçek tanım sayısıdır.
    metrics["gradient_count"] = metrics["gradient_definition_count"]
    return metrics, fails, codes, root


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
    # Python resvg bağlaması production'daki birincil renderer'dır; yalnız CLI
    # aramak çoğu kurulumda parity'yi yanlışlıkla ölçülemez bırakıyordu.
    base = _render_resvg_py(svg_path, w, h)
    if base is None:
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
                       fixture_baseline: dict[str, Any] | None = None,
                       required_metrics: set[str] | None = None) -> FinalArtifactReport:
    """Path'teki kesin SVG'yi HAM BAYTLARI değiştirmeden değerlendirir."""
    svg_path = Path(svg_path)
    before = svg_path.stat()
    data = svg_path.read_bytes()
    after = svg_path.stat()
    read_stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(data)
        and before.st_mtime_ns == after.st_mtime_ns
    )
    return evaluate_final_svg_bytes(
        data, source_rgb, source_alpha=source_alpha, palette_rgb=palette_rgb,
        image_class=image_class, fixture_baseline=fixture_baseline,
        required_metrics=required_metrics, svg_path=svg_path,
        byte_read_stable=read_stable,
    )


def evaluate_final_svg_bytes(svg_bytes: bytes, source_rgb: np.ndarray,
                             source_alpha: np.ndarray | None = None,
                             palette_rgb: np.ndarray | None = None,
                             image_class: str = "clean_logo",
                             fixture_baseline: dict[str, Any] | None = None,
                             required_metrics: set[str] | None = None,
                             svg_path: Path | None = None,
                             byte_read_stable: bool = True) -> FinalArtifactReport:
    """Kesin SVG ham baytlarını kanonik olarak değerlendirir.

    ``svg_path`` yalnız renderer'lar için kullanılır. Hash, byte-size ve XML
    yapı metriklerinin tamamı doğrudan ``svg_bytes`` üzerinden hesaplanır.
    """
    data = bytes(svg_bytes)
    sha = hashlib.sha256(data).hexdigest()
    required = set(required_metrics or ())

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if svg_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="vektoryum-eval-")
        svg_path = Path(temp_dir.name) / "artifact.svg"
        svg_path.write_bytes(data)
    else:
        svg_path = Path(svg_path)

    h0, w0 = source_rgb.shape[:2]
    hard: list[str] = []
    hard_codes: list[str] = []
    soft: list[str] = []
    soft_codes: list[str] = []
    unmeasured: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}

    def add_hard(code: str, message: str) -> None:
        _add_failure(hard, hard_codes, code, message)

    def add_soft(code: str, message: str) -> None:
        if code not in soft_codes:
            soft_codes.append(code)
            soft.append(message)

    def add_unmeasured(name: str) -> None:
        if name not in unmeasured:
            unmeasured.append(name)

    # Uygulanabilir zorunlu metrikler yapısal/render erken dönüşlerinde de
    # kaybolmamalı. Bir başka hard fail, alpha/gradient ölçülmediği gerçeğini
    # rapordan silmez.
    source_has_alpha = bool(
        source_alpha is not None and np.asarray(source_alpha).size
        and (np.asarray(source_alpha) < 255).any()
    )
    if source_has_alpha:
        add_unmeasured("alpha_fidelity")
    if "gradient_fidelity" in required:
        add_unmeasured("gradient_fidelity")
    if image_class == "photo":
        add_unmeasured("photo_vector_fidelity")

    # --- A) yapı ---
    struct, sfails, scodes, _root = _structure_check(data)
    struct["sha256"] = sha
    struct["byte_size"] = len(data)
    struct["byte_read_stable"] = byte_read_stable
    struct["structural_safe"] = not scodes and byte_read_stable
    struct["structural_failure_codes"] = list(scodes)
    metrics["A_structure"] = struct
    hard.extend(sfails)
    hard_codes.extend(scodes)
    if not byte_read_stable:
        add_hard("byte_changed_during_read", "SVG değerlendirme sırasında değişti")

    # Güvenli parse/yapı geçmeden potansiyel aktif içeriği renderer'a verme.
    if hard:
        if temp_dir is not None:
            temp_dir.cleanup()
        return FinalArtifactReport(
            sha, byte_read_stable, "failed", hard, soft, unmeasured, metrics,
            hard_codes, soft_codes,
        )

    # normalize edilmiş karşılaştırma çözünürlüğü: ham AA gürültüsü çözünürlükle
    # patlar (3840²'de semantik topoloji binlerce sahte specke boğulur). Global
    # metrikleri sınırlı çözünürlükte ölç; kritik küçük ROI native refine HG
    # sonraki iş. (Şartname: 512/1024 global render.)
    max_side = 1024
    if max(h0, w0) > max_side:
        sc = max_side / max(h0, w0)
        w, h = int(round(w0 * sc)), int(round(h0 * sc))
        source_cmp = cv2.resize(source_rgb, (w, h), interpolation=cv2.INTER_AREA)
        source_alpha_cmp = (
            cv2.resize(source_alpha, (w, h), interpolation=cv2.INTER_AREA)
            if source_alpha is not None else None
        )
    else:
        w, h, source_cmp = w0, h0, source_rgb
        source_alpha_cmp = source_alpha
    min_area = max(6, round(0.00004 * w * h))   # AA speck < min_area sayılmaz

    # --- render (karşılaştırma çözünürlüğünde) ---
    rnd = render_svg_to_rgb(svg_path, w, h)
    if rnd is None:
        add_hard("render_failed", "SVG render edilemedi")
        add_unmeasured("render")
        if temp_dir is not None:
            temp_dir.cleanup()
        return FinalArtifactReport(
            sha, byte_read_stable, "failed", hard, soft, unmeasured, metrics,
            hard_codes, soft_codes,
        )
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
        add_unmeasured("boundary_offset")
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
        add_unmeasured("component_iou")

    # --- G) gradient / alpha ---
    g_group: dict[str, Any] = {}
    if source_alpha_cmp is not None:
        has_alpha = bool((source_alpha_cmp < 255).any())
        g_group["source_has_alpha"] = has_alpha
        g_group["alpha_fidelity_status"] = "unmeasured" if has_alpha else "not_applicable"
        if has_alpha:
            add_unmeasured("alpha_fidelity")
    else:
        g_group["source_has_alpha"] = False
        g_group["alpha_fidelity_status"] = "not_applicable"
    if "gradient_fidelity" in required:
        g_group["gradient_fidelity_status"] = "unmeasured"
        add_unmeasured("gradient_fidelity")
    else:
        g_group["gradient_fidelity_status"] = "not_required"
    # Foto/sürekli-ton sınıfının tam vektör doğruluğu henüz kanıtlanmış değildir.
    if image_class == "photo":
        add_unmeasured("photo_vector_fidelity")
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
            add_hard("topology_component_delta",
                     f"topoloji: bileşen farkı {comp_delta} > {thr['comp_delta']}")
        if hole_delta > thr["hole_delta"]:
            add_hard("topology_hole_delta",
                     f"topoloji: delik farkı {hole_delta} > {thr['hole_delta']}")
    # seam veto
    sr = metrics["G_gradient_alpha"].get("seam_ratio")
    if sr is not None and sr > thr["seam_ratio"]:
        add_hard("seam_gap", f"ağır seam/gap {sr:.4f} > {thr['seam_ratio']}")
    # renk kalite kapısı
    if cm["de00_p95"] > thr["de00_p95"]:
        add_soft("color_de00_p95", f"ΔE00 p95 {cm['de00_p95']:.2f} > {thr['de00_p95']}")
    if cm["worst_face_de00"] > thr["worst_face_de00"]:
        add_soft("worst_face_de00",
                 f"en kötü yüz ΔE00 {cm['worst_face_de00']:.2f} > {thr['worst_face_de00']}")
    # yapısal ssim/edge alt sınır (temiz fixture gerileme kapısı)
    if metrics["B_visual"]["ssim"] < thr["ssim_min"]:
        add_hard("ssim_below_min",
                 f"SSIM {metrics['B_visual']['ssim']:.4f} < {thr['ssim_min']}")
    if d_group["edge_f1_1px"] < thr["edge_f1_min"]:
        add_soft("edge_f1_below_min",
                 f"edge-F1(1px) {d_group['edge_f1_1px']:.4f} < {thr['edge_f1_min']}")
    # küçük detay
    md = metrics["F_small_detail"]["min_component_iou"]
    if md is not None and md < thr["min_component_iou"]:
        add_soft("component_iou_below_min",
                 f"min bileşen IoU {md:.3f} < {thr['min_component_iou']}")

    # verdict
    if hard:
        verdict = "failed"
    elif unmeasured or soft:
        verdict = "needs_review"
    else:
        verdict = "production_ready"

    if temp_dir is not None:
        temp_dir.cleanup()
    return FinalArtifactReport(
        sha, byte_read_stable, verdict, hard, soft, unmeasured, metrics,
        hard_codes, soft_codes,
    )


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
