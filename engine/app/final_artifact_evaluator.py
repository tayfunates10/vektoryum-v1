"""FinalArtifactEvaluator — exact exported SVG byte truth and quality gates.

FAZ 3 extends the exact-final contract with color-managed, alpha-aware source
truth. Transparent sources are judged as straight RGBA and on white, black and
checker backgrounds; a single white composite can never prove alpha fidelity.
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

from app.fidelity import (
    _edge_f1,
    _ms_ssim,
    _render_cairosvg,
    _render_resvg,
    _render_resvg_py,
    _render_svglib,
    _ssim,
    render_svg_to_rgb,
)
from app.source_truth import (
    alpha_plane_metrics,
    boundary_halo_metrics,
    composite_rgba,
    multibackground_pairs,
    render_svg_to_rgba,
    resize_rgba,
    roundtrip_metrics,
    source_rgba_from_white_composite,
)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 for (..., 3) true CIELAB arrays."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7 + 1e-12)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbarp, Cbarp = 0.5 * (L1 + L2), 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbarp = np.where(np.abs(h1p - h2p) > 180, (hsum + 360) / 2, hsum / 2)
    hbarp = np.where((C1p * C2p) == 0, hsum, hbarp)
    T = (
        1 - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbarp, 1 + 0.015 * Cbarp * T
    dTheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7 + 1e-12))
    Rt = -Rc * np.sin(np.radians(2 * dTheta))
    return np.sqrt(
        (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


@dataclass
class FinalArtifactReport:
    sha256: str
    byte_read_stable: bool
    verdict: str
    hard_fails: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    unmeasured_required: list[str] = field(default_factory=list)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    hard_fail_codes: list[str] = field(default_factory=list)
    soft_warning_codes: list[str] = field(default_factory=list)
    deterministic: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "byte_read_stable": self.byte_read_stable,
            "deterministic": self.deterministic,
            "verdict": self.verdict,
            "hard_fails": self.hard_fails,
            "soft_warnings": self.soft_warnings,
            "unmeasured_required": self.unmeasured_required,
            "metrics": self.metrics,
            "hard_fail_codes": self.hard_fail_codes,
            "soft_warning_codes": self.soft_warning_codes,
        }


_DANGEROUS_TAGS = {
    "script", "image", "feimage", "foreignobject", "iframe", "object", "embed",
    "animate", "animatetransform", "animatemotion", "set", "mpath",
}
_NUMERIC_ATTRS = {
    "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "width", "height", "stroke-width", "stroke-dashoffset", "opacity",
    "fill-opacity", "stroke-opacity", "offset",
}
_NONFINITE_TOKEN = re.compile(r"(?:nan|[-+]?inf(?:inity)?)", re.I)
_PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_URL_REF = re.compile(r"url\(\s*#([^)\s]+)\s*\)", re.I)
_URL_ANY = re.compile(r"url\(\s*([^)]*?)\s*\)", re.I)


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1] if "}" in name else name.split(":")[-1]


def _add_failure(fails: list[str], codes: list[str], code: str, message: str) -> None:
    if code not in codes:
        codes.append(code)
        fails.append(message)


def _structure_check(svg_bytes: bytes) -> tuple[dict[str, Any], list[str], list[str], Any | None]:
    fails: list[str] = []
    codes: list[str] = []
    try:
        root = SafeET.fromstring(svg_bytes)
    except Exception as exc:  # noqa: BLE001
        root = None
        _add_failure(fails, codes, "xml_parse_failed", f"XML parse başarısız: {exc}")

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
            valid = len(parts) == 4 and all(math.isfinite(x) for x in parts) and parts[2] > 0 and parts[3] > 0
        except (TypeError, ValueError):
            valid = False
        metrics["viewbox_valid"] = valid
        if not valid:
            _add_failure(fails, codes, "viewbox_invalid", "viewBox geçersiz veya sonlu değil")

    paths: list[Any] = []
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
            low = value.lower()
            if name.startswith("on"):
                _add_failure(fails, codes, "event_handler", "SVG olay işleyici attribute içeriyor")
            if name in {"href", "src"} and value and not value.startswith("#"):
                code = "embedded_raster" if low.startswith("data:image") else "external_reference"
                if code == "embedded_raster":
                    metrics["has_raster"] = True
                _add_failure(fails, codes, code, "gömülü raster bitmap içeriyor" if code == "embedded_raster" else "dış/tehlikeli kaynak referansı içeriyor")
            if "javascript:" in low or low.startswith("data:"):
                _add_failure(fails, codes, "unsafe_uri", "tehlikeli URI içeriyor")
            for url_target in _URL_ANY.findall(value):
                target = url_target.strip().strip("\"'")
                if target and not target.startswith("#"):
                    _add_failure(fails, codes, "external_reference", "CSS/SVG url() dış veya tehlikeli kaynak içeriyor")
            if name in _NUMERIC_ATTRS and _NONFINITE_TOKEN.search(value):
                metrics["nonfinite"] = True
                _add_failure(fails, codes, "nonfinite_geometry", "non-finite sayı (NaN/Inf)")
            if name in {"d", "points", "transform", "viewbox"} and _NONFINITE_TOKEN.search(value):
                metrics["nonfinite"] = True
                _add_failure(fails, codes, "nonfinite_geometry", "non-finite sayı (NaN/Inf)")

        text = str(element.text or "")
        low_text = text.lower()
        if "javascript:" in low_text or "@import" in low_text or "data:" in low_text:
            _add_failure(fails, codes, "unsafe_css", "tehlikeli inline CSS/metin içeriyor")
        for url_target in _URL_ANY.findall(text):
            target = url_target.strip().strip("\"'")
            if target and not target.startswith("#"):
                _add_failure(fails, codes, "external_reference", "Inline CSS dış veya tehlikeli url() içeriyor")

    refs = 0
    for element in root.iter():
        for value in element.attrib.values():
            refs += sum(1 for ref in _URL_REF.findall(str(value)) if ref in gradient_ids)
    metrics["path_count"] = len(paths)
    metrics["node_count"] = sum(len(_PATH_COMMAND.findall(p.attrib.get("d", ""))) for p in paths)
    metrics["gradient_definition_count"] = (
        metrics["linear_gradient_count"] + metrics["radial_gradient_count"] + metrics["mesh_gradient_count"]
    )
    metrics["gradient_reference_count"] = refs
    metrics["gradient_count"] = metrics["gradient_definition_count"]
    return metrics, fails, codes, root


def _topology_signature(labels: np.ndarray, ncolors: int, min_area: int) -> dict[str, int]:
    comps = holes = 0
    for cid in range(ncolors):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        comps += sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n))
        inv = (1 - mask).astype(np.uint8)
        padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        nb, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(padded, connectivity=4)
        outer = int(labels_h[0, 0])
        holes += sum(i != outer and int(stats_h[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, nb))
    return {"components": int(comps), "holes": max(0, int(holes))}


def _boundary_offsets(src_mask: np.ndarray, rnd_mask: np.ndarray) -> dict[str, float] | None:
    se = cv2.Canny(src_mask.astype(np.uint8) * 255, 50, 150) > 0
    re_ = cv2.Canny(rnd_mask.astype(np.uint8) * 255, 50, 150) > 0
    if not se.any() or not re_.any():
        return None
    dt_r = cv2.distanceTransform((~re_).astype(np.uint8), cv2.DIST_L2, 5)
    dt_s = cv2.distanceTransform((~se).astype(np.uint8), cv2.DIST_L2, 5)
    both = np.concatenate([dt_r[se], dt_s[re_]])
    return {
        "chamfer_mean": float(both.mean()),
        "chamfer_p95": float(np.percentile(both, 95)),
        "hausdorff_p95": float(np.percentile(both, 95)),
        "hausdorff_max": float(both.max()),
    }


def _seam_ratio(src: np.ndarray, rnd: np.ndarray) -> float:
    src_fg = np.any(np.abs(src.astype(np.int16) - 255) > 12, axis=2)
    rnd_white = np.all(rnd > 244, axis=2)
    denom = int(src_fg.sum())
    return 0.0 if denom == 0 else float((src_fg & rnd_white).sum()) / denom


def _cross_renderer_parity(svg_path: Path, w: int, h: int) -> dict[str, Any]:
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


def _color_metrics(src: np.ndarray, rnd: np.ndarray) -> dict[str, float]:
    flat = ciede2000(_lab(src), _lab(rnd)).reshape(-1)
    return {
        "de00_mean": float(flat.mean()),
        "de00_p50": float(np.percentile(flat, 50)),
        "de00_p95": float(np.percentile(flat, 95)),
        "de00_p99": float(np.percentile(flat, 99)),
        "de00_max": float(flat.max()),
    }


def _worst_face_de(src: np.ndarray, rnd: np.ndarray, labels: np.ndarray, ncolors: int, min_area: int) -> float:
    de = ciede2000(_lab(src), _lab(rnd))
    worst = 0.0
    for cid in range(ncolors):
        mask = (labels == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for index in range(1, n):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
                worst = max(worst, float(de[lab == index].mean()))
    return worst


def _classify(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    from app.palette_ops import classify_rgb
    return classify_rgb(rgb.astype(np.uint8), palette.astype(np.float32)).astype(np.uint8)


def _derive_palette(rgb: np.ndarray, k: int = 6) -> np.ndarray:
    from app.graph_source import derive_palette
    return derive_palette(rgb, k)


def _appearance_metrics(source_rgba: np.ndarray, render_rgba: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (src, rnd) in multibackground_pairs(source_rgba, render_rgba).items():
        src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        rnd_gray = cv2.cvtColor(rnd, cv2.COLOR_RGB2GRAY)
        diff = np.abs(src.astype(np.float32) - rnd.astype(np.float32)) / 255.0
        result[name] = {
            "ssim": float(_ssim(src_gray, rnd_gray)),
            "ms_ssim": float(_ms_ssim(src_gray, rnd_gray)),
            "rgb_mae": float(diff.mean()),
            "rgb_p95": float(np.percentile(diff, 95)),
        }
    return result


def evaluate_final_svg(
    svg_path: Path,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    palette_rgb: np.ndarray | None = None,
    image_class: str = "clean_logo",
    fixture_baseline: dict[str, Any] | None = None,
    required_metrics: set[str] | None = None,
) -> FinalArtifactReport:
    path = Path(svg_path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    stable = (
        before.st_dev == after.st_dev and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(data)
        and before.st_mtime_ns == after.st_mtime_ns
    )
    return evaluate_final_svg_bytes(
        data,
        source_rgb,
        source_alpha=source_alpha,
        palette_rgb=palette_rgb,
        image_class=image_class,
        fixture_baseline=fixture_baseline,
        required_metrics=required_metrics,
        svg_path=path,
        byte_read_stable=stable,
    )


def evaluate_final_svg_bytes(
    svg_bytes: bytes,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    palette_rgb: np.ndarray | None = None,
    image_class: str = "clean_logo",
    fixture_baseline: dict[str, Any] | None = None,
    required_metrics: set[str] | None = None,
    svg_path: Path | None = None,
    byte_read_stable: bool = True,
) -> FinalArtifactReport:
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

    src_rgb0 = np.asarray(source_rgb, dtype=np.uint8)
    h0, w0 = src_rgb0.shape[:2]
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

    def mark_measured(name: str) -> None:
        while name in unmeasured:
            unmeasured.remove(name)

    source_has_alpha = bool(
        source_alpha is not None and np.asarray(source_alpha).size and (np.asarray(source_alpha) < 255).any()
    )
    if source_has_alpha or "alpha_fidelity" in required:
        add_unmeasured("alpha_fidelity")
    if "gradient_fidelity" in required:
        add_unmeasured("gradient_fidelity")
    if image_class == "photo":
        add_unmeasured("photo_vector_fidelity")

    struct, sfails, scodes, _root = _structure_check(data)
    struct.update({
        "sha256": sha,
        "byte_size": len(data),
        "byte_read_stable": byte_read_stable,
        "structural_safe": not scodes and byte_read_stable,
        "structural_failure_codes": list(scodes),
    })
    metrics["A_structure"] = struct
    hard.extend(sfails)
    hard_codes.extend(scodes)
    if not byte_read_stable:
        add_hard("byte_changed_during_read", "SVG değerlendirme sırasında değişti")
    if hard:
        if temp_dir is not None:
            temp_dir.cleanup()
        return FinalArtifactReport(sha, byte_read_stable, "failed", hard, soft, unmeasured, metrics, hard_codes, soft_codes)

    max_side = 1024
    if max(h0, w0) > max_side:
        scale = max_side / max(h0, w0)
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        source_cmp = cv2.resize(src_rgb0, (w, h), interpolation=cv2.INTER_AREA)
        source_alpha_cmp = cv2.resize(np.asarray(source_alpha), (w, h), interpolation=cv2.INTER_AREA) if source_alpha is not None else None
    else:
        w, h = w0, h0
        source_cmp = src_rgb0.copy()
        source_alpha_cmp = np.asarray(source_alpha, dtype=np.uint8).copy() if source_alpha is not None else None
    min_area = max(6, round(0.00004 * w * h))

    render_rgba = render_svg_to_rgba(svg_path, w, h)
    if render_rgba is not None:
        rnd = composite_rgba(render_rgba, 255)
    else:
        rnd = render_svg_to_rgb(svg_path, w, h)
    if rnd is None:
        add_hard("render_failed", "SVG render edilemedi")
        add_unmeasured("render")
        if temp_dir is not None:
            temp_dir.cleanup()
        return FinalArtifactReport(sha, byte_read_stable, "failed", hard, soft, unmeasured, metrics, hard_codes, soft_codes)
    if rnd.shape[:2] != (h, w):
        rnd = cv2.resize(rnd, (w, h), interpolation=cv2.INTER_AREA)

    palette = palette_rgb if palette_rgb is not None else _derive_palette(source_cmp)
    ncolors = len(palette)
    co, cr = _classify(source_cmp, palette), _classify(rnd, palette)
    ga, gb = cv2.cvtColor(source_cmp, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rnd, cv2.COLOR_RGB2GRAY)
    metrics["B_visual"] = {
        "ssim": _ssim(ga, gb),
        "ms_ssim": _ms_ssim(ga, gb),
        "cross_renderer": _cross_renderer_parity(svg_path, min(w, 1024), min(h, 1024)),
    }
    cm = _color_metrics(source_cmp, rnd)
    cm["worst_face_de00"] = _worst_face_de(source_cmp, rnd, co, ncolors, min_area)
    cm["palette_agree"] = float((co == cr).mean())
    metrics["C_color"] = cm
    d_group: dict[str, Any] = {
        "edge_f1_1px": _edge_f1(gb, ga, tolerance=1),
        "edge_f1_2px": _edge_f1(gb, ga, tolerance=2),
    }
    supports = sorted(((int((co == c).sum()), c) for c in range(ncolors)), reverse=True)
    off = None
    for _support, cid in supports[:3]:
        off = _boundary_offsets(co == cid, cr == cid)
        if off is not None:
            break
    if off:
        d_group.update(off)
    else:
        add_unmeasured("boundary_offset")
    metrics["D_edge_geometry"] = d_group

    ts_src, ts_rnd = _topology_signature(co, ncolors, min_area), _topology_signature(cr, ncolors, min_area)
    comp_delta = abs(ts_src["components"] - ts_rnd["components"])
    hole_delta = abs(ts_src["holes"] - ts_rnd["holes"])
    metrics["E_topology"] = {
        "source": ts_src,
        "render": ts_rnd,
        "component_delta": comp_delta,
        "hole_delta": hole_delta,
    }
    ious: list[float] = []
    for cid in range(ncolors):
        a, b = co == cid, cr == cid
        union = int((a | b).sum())
        if union >= 20:
            ious.append(float((a & b).sum() / union))
    metrics["F_small_detail"] = {
        "min_component_iou": float(min(ious)) if ious else None,
        "mean_component_iou": float(np.mean(ious)) if ious else None,
    }
    if not ious:
        add_unmeasured("component_iou")

    g_group: dict[str, Any] = {
        "source_has_alpha": source_has_alpha,
        "gradient_fidelity_status": "unmeasured" if "gradient_fidelity" in required else "not_required",
        "seam_ratio": _seam_ratio(source_cmp, rnd),
    }
    if source_has_alpha:
        if render_rgba is None or source_alpha_cmp is None:
            g_group["alpha_fidelity_status"] = "unmeasured"
        else:
            if render_rgba.shape[:2] != (h, w):
                render_rgba = resize_rgba(render_rgba, w, h)
            source_rgba = source_rgba_from_white_composite(source_cmp, source_alpha_cmp)
            alpha_metrics = alpha_plane_metrics(source_alpha_cmp, render_rgba[:, :, 3])
            alpha_metrics.update(boundary_halo_metrics(source_rgba, render_rgba))
            alpha_metrics.update(roundtrip_metrics(source_rgba))
            alpha_metrics["backgrounds"] = _appearance_metrics(source_rgba, render_rgba)
            g_group.update(alpha_metrics)
            g_group["alpha_fidelity_status"] = "measured"
            mark_measured("alpha_fidelity")
    else:
        g_group["alpha_fidelity_status"] = "not_applicable"
        mark_measured("alpha_fidelity")
    metrics["G_gradient_alpha"] = g_group
    metrics["H_editability"] = {
        "node_count": struct["node_count"],
        "path_count": struct["path_count"],
        "nodes_per_path": round(struct["node_count"] / max(1, struct["path_count"]), 2),
    }

    thr = _thresholds(image_class, fixture_baseline)
    if image_class in ("clean_logo", "lineart", "geometric"):
        if comp_delta > thr["comp_delta"]:
            add_hard("topology_component_delta", f"topoloji: bileşen farkı {comp_delta} > {thr['comp_delta']}")
        if hole_delta > thr["hole_delta"]:
            add_hard("topology_hole_delta", f"topoloji: delik farkı {hole_delta} > {thr['hole_delta']}")
    if g_group["seam_ratio"] > thr["seam_ratio"]:
        add_hard("seam_gap", f"ağır seam/gap {g_group['seam_ratio']:.4f} > {thr['seam_ratio']}")
    if cm["de00_p95"] > thr["de00_p95"]:
        add_soft("color_de00_p95", f"ΔE00 p95 {cm['de00_p95']:.2f} > {thr['de00_p95']}")
    if cm["worst_face_de00"] > thr["worst_face_de00"]:
        add_soft("worst_face_de00", f"en kötü yüz ΔE00 {cm['worst_face_de00']:.2f} > {thr['worst_face_de00']}")
    if metrics["B_visual"]["ssim"] < thr["ssim_min"]:
        add_hard("ssim_below_min", f"SSIM {metrics['B_visual']['ssim']:.4f} < {thr['ssim_min']}")
    if d_group["edge_f1_1px"] < thr["edge_f1_min"]:
        add_soft("edge_f1_below_min", f"edge-F1(1px) {d_group['edge_f1_1px']:.4f} < {thr['edge_f1_min']}")
    md = metrics["F_small_detail"]["min_component_iou"]
    if md is not None and md < thr["min_component_iou"]:
        add_soft("component_iou_below_min", f"min bileşen IoU {md:.3f} < {thr['min_component_iou']}")

    if source_has_alpha and g_group.get("alpha_fidelity_status") == "measured":
        if float(g_group["alpha_iou"]) < thr["alpha_iou_min"]:
            add_hard("alpha_iou_below_min", f"alpha IoU {g_group['alpha_iou']:.6f} < {thr['alpha_iou_min']}")
        if float(g_group["alpha_mae"]) > thr["alpha_mae_max"]:
            add_hard("alpha_mae_above_max", f"alpha MAE {g_group['alpha_mae']:.6f} > {thr['alpha_mae_max']}")
        backgrounds = g_group.get("backgrounds") or {}
        for name in ("white", "black", "checker"):
            entry = backgrounds.get(name) or {}
            if float(entry.get("ssim", 0.0)) < thr["alpha_background_ssim_min"]:
                add_hard(f"alpha_{name}_ssim_below_min", f"{name} zemin SSIM {entry.get('ssim', 0.0):.6f} < {thr['alpha_background_ssim_min']}")
            if float(entry.get("rgb_mae", 1.0)) > thr["alpha_background_mae_max"]:
                add_hard(f"alpha_{name}_mae_above_max", f"{name} zemin RGB MAE {entry.get('rgb_mae', 1.0):.6f} > {thr['alpha_background_mae_max']}")
        g_group["alpha_fidelity_status"] = "failed" if any(code.startswith("alpha_") for code in hard_codes) else "passed"

    verdict = "failed" if hard else "needs_review" if unmeasured or soft else "production_ready"
    if temp_dir is not None:
        temp_dir.cleanup()
    return FinalArtifactReport(sha, byte_read_stable, verdict, hard, soft, unmeasured, metrics, hard_codes, soft_codes)


def _thresholds(image_class: str, baseline: dict[str, Any] | None) -> dict[str, float]:
    shared_alpha = dict(
        alpha_iou_min=0.995,
        alpha_mae_max=0.005,
        alpha_background_ssim_min=0.995,
        alpha_background_mae_max=0.008,
    )
    base = {
        "clean_logo": dict(comp_delta=0, hole_delta=0, seam_ratio=0.002, de00_p95=6.0, worst_face_de00=8.0, ssim_min=0.9897, edge_f1_min=0.9940, min_component_iou=0.90, **shared_alpha),
        "lineart": dict(comp_delta=0, hole_delta=0, seam_ratio=0.003, de00_p95=8.0, worst_face_de00=10.0, ssim_min=0.97, edge_f1_min=0.98, min_component_iou=0.85, **shared_alpha),
        "geometric": dict(comp_delta=0, hole_delta=0, seam_ratio=0.002, de00_p95=6.0, worst_face_de00=8.0, ssim_min=0.985, edge_f1_min=0.99, min_component_iou=0.90, **shared_alpha),
        "illustration": dict(comp_delta=3, hole_delta=2, seam_ratio=0.006, de00_p95=10.0, worst_face_de00=14.0, ssim_min=0.94, edge_f1_min=0.94, min_component_iou=0.70, alpha_iou_min=0.990, alpha_mae_max=0.010, alpha_background_ssim_min=0.985, alpha_background_mae_max=0.015),
        "photo": dict(comp_delta=9999, hole_delta=9999, seam_ratio=0.02, de00_p95=20.0, worst_face_de00=30.0, ssim_min=0.80, edge_f1_min=0.70, min_component_iou=0.40, alpha_iou_min=0.985, alpha_mae_max=0.015, alpha_background_ssim_min=0.975, alpha_background_mae_max=0.025),
    }
    thresholds = dict(base.get(image_class, base["illustration"]))
    if baseline:
        if "ssim" in baseline:
            thresholds["ssim_min"] = max(thresholds["ssim_min"], baseline["ssim"] - 0.002)
        if "edge_f1_1px" in baseline:
            thresholds["edge_f1_min"] = max(thresholds["edge_f1_min"], baseline["edge_f1_1px"] - 0.003)
        if "seam_ratio" in baseline:
            thresholds["seam_ratio"] = min(thresholds["seam_ratio"], baseline["seam_ratio"] + 0.0005)
    return thresholds
