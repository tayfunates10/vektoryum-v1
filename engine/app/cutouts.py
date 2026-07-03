"""Shape stacking dönüşümü: stacked -> cut-outs.

VTracer çıktısı KATMANLIDIR (stacked): şekiller belge sırasıyla üst üste
boyanır; alttaki şeklin görünmeyen bölümü de path'inde durur. CUT-OUTS
gösteriminde her path yalnız GÖRÜNEN bölgesini içerir: üstündeki şekillerin
birleşimi kendisinden çıkarılır. Böylece bir bileşeni seçip taşımak, altında
başka şekil kalmadan mümkün olur (düzenlenebilirlik).

Uygulama notları:
* Boolean işlemler ``pyclipper`` ile tamsayı uzayında yapılır (ölçek 100 =
  0.01px hassasiyet). Eğriler yoğun örneklemeyle (adım ~0.8px) poligonlara
  düzleştirilir — cut-outs geometrisi kesim sınırları içerdiğinden dosya
  büyür; bu, gösterimin doğası gereğidir.
* DİKİŞ ÖNLEME: üstteki birleşim çıkarılmadan önce ~0.25px İÇERİ ofsetlenir;
  parçalar bu kadar binişir ve render anti-alias'ında kılcal zemin sızması
  (seam) oluşmaz.
* ``pyclipper`` yoksa dönüşüm ``skipped`` döner; stacked çıktı aynen kalır
  (çökme yok, zorunlu bağımlılık değil).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
_SCALE = 100.0          # tamsayı uzay ölçeği (0.01px)
_OVERLAP_PX = 0.25      # dikiş önleme binişi
_FLATTEN_STEP = 0.8     # eğri düzleştirme adımı (px)

try:
    import pyclipper
except ImportError:  # pragma: no cover
    pyclipper = None

try:
    from svgpathtools import parse_path
except ImportError:  # pragma: no cover
    parse_path = None


def is_available() -> bool:
    return pyclipper is not None and parse_path is not None


def _flatten_to_rings(
    d: str, xf: tuple[float, float, float, float, float, float] | None = None
) -> list[list[tuple[int, int]]] | None:
    """d-string'in KAPALI alt yollarını tamsayı halkalarına düzleştirir.

    ``xf`` verilirse (a,b,c,d,e,f) affine dönüşümü uygulanır: path'ler farklı
    ``transform`` taşıyabilir; boolean işlemler TEK kullanıcı uzayında yapılmalı
    (yerel koordinat karışımı yanlış bölge çıkarıyordu — gerçek bir hataydı).
    """
    try:
        rings: list[list[tuple[int, int]]] = []
        for sub in parse_path(d).continuous_subpaths():
            try:
                if not sub.isclosed():
                    return None  # açık alt yol: bu path dönüştürülmez
                length = sub.length()
            except Exception:  # noqa: BLE001
                return None
            n = int(max(8, min(4000, (length or 8) / _FLATTEN_STEP)))
            ring = []
            for i in range(n):
                p = sub.point(i / n)
                x, y = p.real, p.imag
                if xf is not None:
                    a, b, c, dd, e, f = xf
                    x, y = a * x + c * y + e, b * x + dd * y + f
                ring.append((int(round(x * _SCALE)), int(round(y * _SCALE))))
            # ardışık tekrarları at
            dedup = [ring[0]]
            for q in ring[1:]:
                if q != dedup[-1]:
                    dedup.append(q)
            if len(dedup) >= 3:
                rings.append(dedup)
        return rings if rings else None
    except Exception:  # noqa: BLE001
        return None


def _rings_to_d(rings: list[list[tuple[int, int]]]) -> str:
    parts: list[str] = []
    for ring in rings:
        pts = [(x / _SCALE, y / _SCALE) for x, y in ring]
        parts.append(f"M {pts[0][0]:.2f} {pts[0][1]:.2f}")
        parts.extend(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
        parts.append("Z")
    return " ".join(parts)


def _union(subject: list, clip: list) -> list:
    if not subject:
        return [list(r) for r in clip]
    if not clip:
        return subject
    pc = pyclipper.Pyclipper()
    pc.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    pc.AddPaths(clip, pyclipper.PT_CLIP, True)
    return pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)


def _difference(subject: list, clip: list) -> list:
    if not clip:
        return subject
    pc = pyclipper.Pyclipper()
    pc.AddPaths(subject, pyclipper.PT_SUBJECT, True)
    pc.AddPaths(clip, pyclipper.PT_CLIP, True)
    return pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_EVENODD, pyclipper.PFT_NONZERO)


def _inset(rings: list, delta_px: float) -> list:
    """Halkaları içeri ofsetler (dikiş önleme binişi için)."""
    if not rings:
        return rings
    po = pyclipper.PyclipperOffset()
    po.AddPaths(rings, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    return po.Execute(-delta_px * _SCALE)


def convert_svg_to_cutouts(svg_path: Path) -> dict[str, Any]:
    """Stacked SVG'yi yerinde cut-outs gösterimine çevirir.

    Her path'ten, belge sırasında ÜSTÜNDE kalan path'lerin (hafif içeri
    ofsetli) birleşimi çıkarılır. Tamamen örtülen path'ler silinir. Eğri
    içermeyen görünüm birebir korunur (0.01px hassasiyet + 0.25px biniş).
    """
    if not is_available():
        return {"status": "skipped", "error": "pyclipper/svgpathtools yok"}
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}

    # path öğelerini (ebeveynleriyle) belge sırasında topla; transform'lar
    # kullanıcı uzayına uygulanır ki tüm halkalar aynı koordinat uzayında olsun
    from app.exporters import _parse_transform

    items: list[tuple[Any, Any, list | None]] = []  # (parent, el, rings)
    for parent in root.iter():
        # ebeveyn (grup) transform'u kaldırılamaz (kardeşleri etkiler); böyle
        # path'ler dönüştürülmez, stacked kalır ve birleşime katılmaz
        parent_has_xf = parent.get("transform") is not None
        for el in list(parent):
            if el.tag.split("}")[-1] != "path":
                continue
            d = el.get("d")
            if parent_has_xf:
                items.append((parent, el, None))
                continue
            xf = _parse_transform(el.get("transform")) if el.get("transform") else None
            rings = _flatten_to_rings(d, xf) if d else None
            items.append((parent, el, rings))
    if len(items) < 2:
        return {"status": "no_change", "paths": len(items)}

    try:
        upper_union: list = []
        removed = 0
        changed = 0
        # üstten (belgede son) alta doğru
        for parent, el, rings in reversed(items):
            if rings is None:
                continue  # dönüştürülemeyen path aynen kalır; birleşime katılmaz
            visible = _difference(rings, _inset(upper_union, _OVERLAP_PX))
            if not visible:
                parent.remove(el)
                removed += 1
            else:
                new_d = _rings_to_d(visible)
                if new_d:
                    el.set("d", new_d)
                    el.set("fill-rule", "evenodd")
                    # yeni d KULLANICI uzayında: transform kaldırılmalı,
                    # yoksa çift uygulanır
                    if el.get("transform") is not None:
                        del el.attrib["transform"]
                    changed += 1
            upper_union = _union(upper_union, rings)
    except Exception as e:  # noqa: BLE001
        logger.warning("Cut-outs dönüşümü başarısız, stacked korunuyor: %s", e)
        return {"status": "failed", "error": str(e)}

    try:
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}
    return {"status": "completed", "paths_changed": changed, "paths_removed": removed}
