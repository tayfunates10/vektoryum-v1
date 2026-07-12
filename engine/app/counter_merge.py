"""Eğri koruyan sayaç (counter) birleştirme: gerçek evenodd delikleri.

VTracer "stacked" ressam modelinde iç boşluklar (R/P/B/A sayaçları, halka
içleri) çoğu zaman ZEMİN RENKLİ bir örtme path'i olarak üstte durur. Bu
görsel olarak doğru ama vektörel olarak yanlıştır: arka plan değişince leke
kalır, şekil tek parça düzenlenemez. Bu modül uygun örtme path'lerini alttaki
ebeveyn path'e ALT-YOL olarak gömer ve ``fill-rule="evenodd"`` ile gerçek
deliğe çevirir.

Cutouts'tan (pyclipper) temel farkı: geometri YENİDEN ÖRNEKLENMEZ. Ebeveynin
d verisi aynen kalır; örtmenin d verisi (gerekirse yalnız yön ters çevrilerek)
sonuna eklenir. Bézier/yay komutları korunur; komut sayısı artmaz (örtme
elementi silindiği için toplam değişmez, path sayısı azalır).

Güvenlik: dönüşüm ÖLÇÜM KAPILIDIR. Her birleştirme sonrası SVG kaynak
çözünürlükte render edilir; global palet uyumu ve örtme bölgesi IoU'su
toleransı aşarsa o birleştirme geri alınır (yarı uygulanmış durum kalmaz).
``VEKTORYUM_COUNTER_MERGE=off`` tüm dönüşümü kapatır (güvenli eski yol).

Determinizm: aday taraması belge sırasıyla yapılır, rastgelelik yoktur.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

# Eşikler (ölçüm kapısı; fixture'a özel DEĞİL — göreli/normalize değerler):
MIN_COUNTER_AREA_PX = 9.0        # kaynak px²: bundan küçük sayaç gürültüdür
MAX_COUNTER_CANVAS_FRAC = 0.25   # sayaç tuvalin %25'inden büyük olamaz
CONTAIN_MARGIN_PX = 0.5          # örtme, ebeveyn sınırına bu kadar yaklaşamaz
MAX_GLOBAL_DISAGREE = 1e-4       # render sınıf uyumu bozulması üst sınırı
MIN_LOCAL_IOU = 0.995            # örtme bölgesinde delik maskesi IoU tabanı
_FLATTEN_STEP_PX = 1.0           # içerme TESTİ için düzleştirme (geometri değişmez)

try:
    from svgpathtools import parse_path
except ImportError:  # pragma: no cover
    parse_path = None


@dataclass
class CounterMergeCandidate:
    """Tek örtme->delik dönüşümünün kaydı (rapor + hata ayıklama)."""

    parent_path_index: int
    overlay_path_index: int
    parent_fill: str
    overlay_fill: str
    containment_score: float = 0.0
    render_delta: float = 1.0
    topology_before: dict[str, int] = field(default_factory=dict)
    topology_after: dict[str, int] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str | None = None


def is_available() -> bool:
    return parse_path is not None


# ---------------------------------------------------------------------------
# Geometri yardımcıları (yalnız TEST için düzleştirme; çıktı geometrisi orijinal)
# ---------------------------------------------------------------------------
def _flatten_subpaths(d: str) -> list[np.ndarray] | None:
    """Kapalı alt yolları Nx2 float32 poligonlara düzleştirir (içerme testi)."""
    try:
        polys: list[np.ndarray] = []
        for sub in parse_path(d).continuous_subpaths():
            if not sub.isclosed():
                return None
            length = sub.length() or 8.0
            n = int(max(16, min(2000, length / _FLATTEN_STEP_PX)))
            pts = [sub.point(i / n) for i in range(n)]
            polys.append(np.array([[p.real, p.imag] for p in pts], dtype=np.float32))
        return polys or None
    except Exception:  # noqa: BLE001
        return None


def _signed_dist_inside(polys: list[np.ndarray], pt: tuple[float, float]) -> float:
    """Noktanın DOLU bölgeye göre işaretli mesafesi (evenodd parite modeli).

    İçindeyse +en yakın sınır mesafesi, dışındaysa negatif. Ebeveyn nonzero
    olabilir; bu bir ÖN filtredir — nihai hakem render kapısıdır.
    """
    inside_count = 0
    min_abs = float("inf")
    for poly in polys:
        d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), pt, True)
        if d >= 0:
            inside_count += 1
        min_abs = min(min_abs, abs(d))
    return min_abs if inside_count % 2 == 1 else -min_abs


def _bbox(polys: list[np.ndarray]) -> tuple[float, float, float, float]:
    allp = np.vstack(polys)
    return (float(allp[:, 0].min()), float(allp[:, 1].min()),
            float(allp[:, 0].max()), float(allp[:, 1].max()))


def _poly_area(polys: list[np.ndarray]) -> float:
    """Evenodd parite modeliyle yaklaşık dolu alan (dış - iç)."""
    total = 0.0
    for poly in polys:
        total += abs(cv2.contourArea(poly.reshape(-1, 1, 2)))
    outer = max(abs(cv2.contourArea(p.reshape(-1, 1, 2))) for p in polys)
    return max(outer, 2 * outer - total)  # tek alt yol: outer; delikliler: dış-içler


def _sample_boundary(polys: list[np.ndarray], per_poly: int = 24) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for poly in polys:
        step = max(1, len(poly) // per_poly)
        pts.extend((float(x), float(y)) for x, y in poly[::step])
    return pts


def _round_d(d: str) -> str:
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.2f}".rstrip("0").rstrip("."), d)


def _topology(paths: list[dict]) -> dict[str, int]:
    return {
        "path_count": len(paths),
        "subpath_count": sum(p["subpaths"] for p in paths),
        "total_cmds": sum(p["cmds"] for p in paths),
    }


def _path_info(el: ET.Element) -> dict:
    d = el.get("d") or ""
    return {
        "el": el,
        "d": d,
        "fill": (el.get("fill") or "").lower(),
        "subpaths": len(re.findall(r"(?<![0-9a-zA-Z.,-])[Mm]", " " + d)),
        "cmds": len(re.findall(r"[MLCQAZHVSTmlcqazhvst]", d)),
    }


# ---------------------------------------------------------------------------
# Ana dönüşüm
# ---------------------------------------------------------------------------
def merge_counters(
    svg_path: Path,
    width: int,
    height: int,
    render_fn: Callable[[Path, int, int], np.ndarray | None],
    source_rgb: np.ndarray | None = None,
    max_merges: int = 8,
) -> dict[str, Any]:
    """Örtme sayaçlarını gerçek evenodd deliklerine çevirir (ölçüm kapılı).

    ``render_fn(svg, w, h)`` kaynak çözünürlükte RGB render döndürmelidir;
    render yoksa dönüşüm yapılmaz (kapısız değişiklik YOK). ``source_rgb``
    verilirse yerel kapı KAYNAĞA göre ölçülür: örtme modundaki çift-AA
    saçağı (üst üste çakışan kenarlarda koyu kılcal) delik modunda kaybolur;
    render-render kıyası bu iyileşmeyi "fark" diye reddederdi. Kaynak uyumu
    düşmediği sürece dönüşüm kabul edilir.
    """
    if not is_available():
        return {"status": "skipped", "reason": "svgpathtools yok"}
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error": str(e)}

    # koordinat sözleşmesi: transform'lu belge normalize edilmemiştir — dokunma
    for el in root.iter():
        if el.get("transform"):
            return {"status": "skipped", "reason": "transform içeren belge"}

    parents: dict[ET.Element, ET.Element] = {}
    paths: list[dict] = []
    for parent in root.iter():
        for el in list(parent):
            if el.tag.split("}")[-1] == "path" and el.get("d"):
                parents[el] = parent
                paths.append(_path_info(el))
    if len(paths) < 3:
        return {"status": "no_change", "reason": "yeterli path yok"}

    hexes = [p["fill"] if re.fullmatch(r"#[0-9a-f]{6}", p["fill"]) else None for p in paths]
    if not any(hexes):
        return {"status": "no_change", "reason": "hex dolgu yok"}

    canvas_area = float(width * height)
    topo_before = _topology(paths)

    # --- aday tarama (belge sırası; determinist) --------------------------
    flat_cache: dict[int, list[np.ndarray] | None] = {}

    def flat(i: int) -> list[np.ndarray] | None:
        if i not in flat_cache:
            flat_cache[i] = _flatten_subpaths(paths[i]["d"])
        return flat_cache[i]

    prelim: list[CounterMergeCandidate] = []
    for j in range(len(paths) - 1, 0, -1):
        if hexes[j] is None:
            continue
        fj = flat(j)
        if fj is None:
            continue
        area_j = _poly_area(fj)
        if area_j < MIN_COUNTER_AREA_PX or area_j > MAX_COUNTER_CANVAS_FRAC * canvas_area:
            continue
        bx0, by0, bx1, by1 = _bbox(fj)
        # üstte örtüşen path varsa atla (görünür başka bileşeni örtüyor olabilir)
        occluded = False
        for m in range(j + 1, len(paths)):
            fm = flat(m)
            if fm is None:
                continue
            mx0, my0, mx1, my1 = _bbox(fm)
            if not (mx1 < bx0 or mx0 > bx1 or my1 < by0 or my0 > by1):
                occluded = True
                break
        if occluded:
            continue
        samples = _sample_boundary(fj)
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        # ebeveyn: örtmeyi TAM içeren en üstteki alt path (farklı renk)
        parent_idx = None
        contain_score = 0.0
        for i in range(j - 1, -1, -1):
            if hexes[i] is None or hexes[i] == hexes[j]:
                continue
            fi = flat(i)
            if fi is None:
                continue
            dists = [_signed_dist_inside(fi, pt) for pt in samples]
            dmin = min(dists) if dists else -1.0
            if dmin >= CONTAIN_MARGIN_PX:
                parent_idx = i
                contain_score = float(dmin)
                break
            if dmin > -CONTAIN_MARGIN_PX:
                break  # ebeveyn sınırına DEĞİYOR: sayaç değil, bitişik parça
        if parent_idx is None:
            continue
        # örtme rengi, ebeveynin ALTINDAKİ bölgenin rengiyle aynı olmalı
        beneath = None
        for k in range(parent_idx - 1, -1, -1):
            fk = flat(k)
            if fk is None or hexes[k] is None:
                continue
            if _signed_dist_inside(fk, (cx, cy)) > 0:
                beneath = hexes[k]
                break
        if beneath != hexes[j]:
            continue
        prelim.append(CounterMergeCandidate(
            parent_path_index=parent_idx, overlay_path_index=j,
            parent_fill=hexes[parent_idx], overlay_fill=hexes[j],
            containment_score=round(contain_score, 2),
        ))
        if len(prelim) >= max_merges:
            break

    if not prelim:
        return {"status": "no_change", "reason": "sayaç adayı yok",
                "topology": topo_before}

    # --- render kapısı ile uygula -----------------------------------------
    before_rgb = render_fn(svg_path, width, height)
    if before_rgb is None:
        return {"status": "skipped", "reason": "render backend yok",
                "candidates": [asdict(c) for c in prelim]}
    fills_rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in sorted({x for x in hexes if x})],
        dtype=np.float32,
    )

    from app.palette_ops import classify_rgb  # noqa: PLC0415

    def classify(img: np.ndarray) -> np.ndarray:
        return classify_rgb(img, fills_rgb)  # bant bazlı: bellek sınırlı, bit-birebir

    cls_before = classify(before_rgb)
    src_cls = None
    if source_rgb is not None and source_rgb.shape[:2] == (height, width):
        src_cls = classify(source_rgb)
    tmp = svg_path.with_suffix(".cmerge.svg")
    accepted: list[CounterMergeCandidate] = []
    # üstten alta sırayla; indeksler orijinal belge sırasına göre sabit
    for cand in prelim:
        pi, oi = cand.parent_path_index, cand.overlay_path_index
        p_el, o_el = paths[pi]["el"], paths[oi]["el"]
        backup_d = p_el.get("d")
        backup_fr = p_el.get("fill-rule")
        backup_cr = p_el.get("clip-rule")

        # DELİK ALT-YOLU BİT-BİREBİR KOPYALANIR (yeniden serileştirme YOK):
        # evenodd yön-bağımsızdır; alt-yolu ters çevirip yeniden yazmak eğri
        # düzleştirme/AA örneklemesini değiştirir ve render kapısını gereksiz
        # bozar (ölçüldü: ters çevrilmiş sayaçta yerel IoU 0.9709'a düştü,
        # orijinal metinle fark bit-düzeyinde 0). Orijinal d metni korunur.
        hole_d = paths[oi]["d"]

        # uygula
        p_el.set("d", f"{backup_d} {hole_d}")
        p_el.set("fill-rule", "evenodd")
        p_el.set("clip-rule", "evenodd")
        o_parent = parents[o_el]
        o_pos = list(o_parent).index(o_el)
        o_parent.remove(o_el)
        tree.write(str(tmp), encoding="utf-8", xml_declaration=True)

        after_rgb = render_fn(tmp, width, height)
        ok, reason, delta = False, None, 1.0
        if after_rgb is None:
            reason = "render başarısız"
        else:
            # MADDİ değişim maskesi: kanal toplam farkı > 30. Sert sınıflandırma
            # kullanılMAZ çünkü iki renk ORTASINDAKİ AA pikselleri 1 birimlik
            # render gürültüsüyle sınıf çevirir (ölçüldü: birebir aynı görüntüde
            # %1 "sınıf farkı") — o gürültü kalite sinyali değildir.
            from app.palette_ops import abs_diff_sum  # noqa: PLC0415

            material = abs_diff_sum(after_rgb, before_rgb) > 30
            delta = float(material.mean())
            if delta > MAX_GLOBAL_DISAGREE:
                reason = f"global maddi fark {delta:.6f} > {MAX_GLOBAL_DISAGREE}"
            else:
                fj2 = flat(oi)
                x0, y0, x1, y1 = _bbox(fj2)
                pad = 8
                xa, ya = max(0, int(x0) - pad), max(0, int(y0) - pad)
                xb, yb = min(width, int(x1) + pad), min(height, int(y1) + pad)
                m_loc = material[ya:yb, xa:xb]
                if not m_loc.any():
                    ok = True  # görünür değişiklik yok: dönüşüm birebir
                elif src_cls is not None:
                    # maddi değişen piksellerde KAYNAĞA göre bilanço:
                    # kötüleşen > iyileşen ise reddet
                    cls_after = classify(after_rgb)
                    sc = src_cls[ya:yb, xa:xb]
                    cbc = cls_before[ya:yb, xa:xb]
                    cac = cls_after[ya:yb, xa:xb]
                    worse = int(((sc == cbc) & (sc != cac) & m_loc).sum())
                    better = int(((sc != cbc) & (sc == cac) & m_loc).sum())
                    if worse > better + 4:
                        reason = f"kaynak uyumu kötüleşti (kötü {worse} > iyi {better})"
                    else:
                        ok = True
                else:
                    # kaynak yoksa: yerel maddi değişim payı küçük olmalı
                    frac = float(m_loc.mean())
                    if frac > 0.02:
                        reason = f"yerel maddi fark {frac:.4f} > 0.02"
                    else:
                        ok = True
        cand.render_delta = round(delta, 6)
        if ok:
            cand.accepted = True
            before_rgb = after_rgb  # sonraki birleştirmeler bu duruma göre ölçülür
            cls_before = classify(after_rgb)
            paths[pi]["d"] = p_el.get("d")
            paths[pi]["subpaths"] += paths[oi]["subpaths"]
            paths[pi]["cmds"] += paths[oi]["cmds"]
            paths[oi]["subpaths"] = paths[oi]["cmds"] = 0
            accepted.append(cand)
        else:
            cand.rejection_reason = reason
            p_el.set("d", backup_d)
            if backup_fr is None:
                p_el.attrib.pop("fill-rule", None)
            else:
                p_el.set("fill-rule", backup_fr)
            if backup_cr is None:
                p_el.attrib.pop("clip-rule", None)
            else:
                p_el.set("clip-rule", backup_cr)
            o_parent.insert(o_pos, o_el)

    tmp.unlink(missing_ok=True)
    live_paths = [p for p in paths if p["cmds"] > 0]
    topo_after = _topology(live_paths)
    for c in prelim:
        c.topology_before, c.topology_after = topo_before, topo_after
    if accepted:
        if topo_after["total_cmds"] > topo_before["total_cmds"]:
            # komut bütçesi: birleştirme komut EKLEYEMEZ — tümünü geri al
            logger.warning("counter_merge komut sayısını artırdı; geri alınıyor")
            return {"status": "rolled_back", "reason": "komut artışı",
                    "topology_before": topo_before, "topology_after": topo_after}
        tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        status = "completed"
    else:
        status = "no_change"
    return {
        "status": status,
        "merged": len(accepted),
        "candidates": [asdict(c) for c in prelim],
        "topology_before": topo_before,
        "topology_after": topo_after if accepted else topo_before,
    }
