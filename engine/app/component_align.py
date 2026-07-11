"""Küçük bileşen hizalama refiti (ölçüm kapılı).

İzleme tavanı (VEKTORYUM_TRACE_CAP) nedeniyle büyük girdiler küçültülmüş
rasterden izlenir ve SVG kaynak boyuta viewBox ile ölçeklenir. Küçük ama
anlamlı bileşenler (ör. ® simgesi) bu gidiş-dönüşte birkaç piksel kayabilir /
hafifçe büyüyebilir; halka gibi ince şekillerde bu, bölgesel IoU'yu çökertir
(LEGO ® vakası: global palet uyumu %99.5 iken halka bileşeni IoU %63).

Bu geçiş genel bir düzeltmedir — örneğe özel koordinat/renk sabiti içermez:

1. Kazanan SVG render edilir; ``fidelity._component_class_report`` zayıf küçük
   bileşen bölgelerini ve her bölge için kaynak/render sınıf momentlerinden
   kestirilen benzerlik dönüşümünü (dx, dy, s) verir.
2. Bölgenin içindeki KÜÇÜK path'lere dönüşüm svgpathtools ile koordinat
   düzeyinde uygulanır (transform özniteliği eklenmez; koordinatlar düz kalır).
   Bölgedeki tüm küçük path'ler birlikte taşınır: aynı izlemeden geldikleri
   için kayma ortaktır (siyah halka + R + kırmızı sayaç tek gövde gibi).
3. Sonuç yeniden ölçülür; sadakat düşer ya da en kötü bileşen IoU'su
   iyileşmezse dosya değişmeden bırakılır (ölçüm kapısı).

svgpathtools ya da render backend yoksa sessizce atlanır (çökme yok).
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from app.fidelity import (
    _component_class_report,
    compute_fidelity,
    load_reference_rgb,
    render_svg_to_rgb,
)

logger = logging.getLogger(__name__)

# kestirim sınırları: bunların dışındaki dönüşümler gürültü sayılır ve atlanır
_MAX_SHIFT_PX = 14.0      # karşılaştırma çözünürlüğünde (aşağıda ölçeklenir)
_SCALE_MIN, _SCALE_MAX = 0.94, 1.06
_ALIGN_COMPARE_SIDE = 1024  # hizalama kestirimi için karşılaştırma çözünürlüğü


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    vb = root.get("viewBox")
    if not vb:
        return None
    try:
        x, y, w, h = (float(v) for v in vb.replace(",", " ").split())
        return x, y, w, h
    except ValueError:
        return None


def _similarity_matrix(cx: float, cy: float, s: float, tx: float, ty: float) -> np.ndarray:
    """Bölge merkezi etrafında ölçek + öteleme: T(c+t) @ S(s) @ T(-c)."""
    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
    sc = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=float)
    t_pos = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=float)
    return t_pos @ sc @ t_neg


def apply_component_align(
    svg_path: Path,
    original_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Zayıf küçük bileşen bölgelerini hizalar; kapıyı geçemezse dokunmaz."""
    try:
        from svgpathtools import parse_path  # noqa: PLC0415
        from svgpathtools.path import transform as svgt_transform  # noqa: PLC0415
    except Exception:  # noqa: BLE001 (opsiyonel bağımlılık yoksa no-op)
        return {"applied": False, "reason": "svgpathtools_yok"}

    try:
        reference, (w, h) = load_reference_rgb(Path(original_path), max_side=_ALIGN_COMPARE_SIDE)
    except Exception as e:  # noqa: BLE001
        return {"applied": False, "reason": f"referans_yuklenemedi: {e}"}
    rendered = render_svg_to_rgb(Path(svg_path), w, h)
    if rendered is None:
        return {"applied": False, "reason": "render_yok"}

    report = _component_class_report(reference, rendered)
    weak = (report or {}).get("weak_components") or []
    if not weak:
        return {"applied": False, "reason": "zayif_bilesen_yok"}
    fid_before = compute_fidelity(reference, rendered)
    min_before = (report or {}).get("component_min_iou")

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    vb = _parse_viewbox(root)
    if vb is None:
        return {"applied": False, "reason": "viewBox_yok"}
    _, _, vbw, vbh = vb
    unit = vbw / float(w)  # karşılaştırma pikseli -> viewBox birimi

    def _translate_of(el: ET.Element) -> tuple[float, float] | None:
        """Yalnız translate(tx[,ty]) destekler; başka dönüşüm varsa None (atla)."""
        tr = (el.get("transform") or "").strip()
        if not tr:
            return 0.0, 0.0
        import re  # noqa: PLC0415
        m = re.fullmatch(r"translate\(\s*([-\d.eE]+)\s*[, ]?\s*([-\d.eE]+)?\s*\)", tr)
        if not m:
            return None
        return float(m.group(1)), float(m.group(2) or 0.0)

    path_els = [el for el in root.iter() if el.tag.endswith("path") and el.get("d")]
    parsed: list[tuple[ET.Element, Any, tuple[float, float, float, float], tuple[float, float]]] = []
    for el in path_els:
        t = _translate_of(el)
        if t is None:
            continue
        try:
            p = parse_path(el.get("d"))
            xmin, xmax, ymin, ymax = p.bbox()
            # bbox GLOBAL koordinatta (element translate'i dahil)
            parsed.append((el, p, (xmin + t[0], xmax + t[0], ymin + t[1], ymax + t[1]), t))
        except Exception:  # noqa: BLE001 (bozuk path'i olduğu gibi bırak)
            continue

    regions: list[dict[str, Any]] = []
    moved_total = 0
    moved_els: set[int] = set()  # çakışan bölgelerde aynı path'e çifte dönüşüm yok
    for comp in sorted(weak, key=lambda c: c["iou"])[:3]:
        dx, dy, s = comp["dx"], comp["dy"], comp["scale"]
        if abs(dx) > _MAX_SHIFT_PX or abs(dy) > _MAX_SHIFT_PX:
            continue
        if not (_SCALE_MIN <= s <= _SCALE_MAX):
            continue
        x, y, ww, hh = comp["bbox"]
        pad = 0.30 * max(ww, hh)
        rx0, ry0 = (x - pad) * unit, (y - pad) * unit
        rx1, ry1 = (x + ww + pad) * unit, (y + hh + pad) * unit
        rcx, rcy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        rarea = max(1.0, (rx1 - rx0) * (ry1 - ry0))
        mat = _similarity_matrix(rcx, rcy, s, dx * unit, dy * unit)
        moved = 0
        for el, p, (xmin, xmax, ymin, ymax), (tx, ty) in parsed:
            inside = xmin >= rx0 and xmax <= rx1 and ymin >= ry0 and ymax <= ry1
            small_enough = (xmax - xmin) * (ymax - ymin) <= rarea * 1.6
            if not (inside and small_enough) or id(el) in moved_els:
                continue
            try:
                # dönüşüm global koordinatta tanımlı; path yerel (translate'li)
                # koordinatta yazılı: M_local = T(-t) @ M @ T(t)
                t_pos = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)
                t_neg = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]], dtype=float)
                p2 = svgt_transform(p, t_neg @ mat @ t_pos)
                el.set("d", p2.d())
                moved_els.add(id(el))
                moved += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("Path dönüşümü atlandı: %s", e)
        if moved:
            moved_total += moved
            regions.append({"bbox": comp["bbox"], "iou": comp["iou"],
                            "dx": dx, "dy": dy, "scale": s, "paths": moved})

    if moved_total == 0:
        return {"applied": False, "reason": "uygun_path_yok", "weak": weak[:3]}

    tree.write(str(out_path), encoding="unicode", xml_declaration=False)

    # ölçüm kapısı: sadakat korunmalı VE en kötü bileşen IoU'su iyileşmeli
    rendered_after = render_svg_to_rgb(Path(out_path), w, h)
    if rendered_after is None:
        out_path.unlink(missing_ok=True)
        return {"applied": False, "reason": "yeni_render_yok"}
    fid_after = compute_fidelity(reference, rendered_after)
    report_after = _component_class_report(reference, rendered_after)
    min_after = (report_after or {}).get("component_min_iou")

    fid_ok = fid_after["fidelity_score"] >= fid_before["fidelity_score"] - 0.02
    comp_ok = (
        min_before is not None and min_after is not None
        and min_after >= min_before + 0.02
    )
    if not (fid_ok and comp_ok):
        out_path.unlink(missing_ok=True)
        return {
            "applied": False, "reason": "olcum_kapisi",
            "fidelity_before": fid_before["fidelity_score"],
            "fidelity_after": fid_after["fidelity_score"],
            "component_min_before": min_before, "component_min_after": min_after,
            "regions": regions,
        }
    return {
        "applied": True,
        "fidelity_before": fid_before["fidelity_score"],
        "fidelity_after": fid_after["fidelity_score"],
        "component_min_before": min_before, "component_min_after": min_after,
        "regions": regions, "moved_paths": moved_total,
    }
