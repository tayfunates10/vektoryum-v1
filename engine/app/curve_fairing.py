"""Eğri pürüzsüzleştirme (curve fairing): spline eklemlerinde teğet hizalama.

VTracer'ın spline çıktısında ardışık iki kübik Bezier'in birleştiği eklemde
teğetler çoğu zaman TAM hizalı değildir: izleme gürültüsü küçük açılı bir
"kink" (kırıklık) bırakır ve büyütmede eğri boyunca hafif köşelenmeler görünür.
Referans vektörleştiricilerin "curve fairing / tangent matching" dediği adımın
karşılığı: niyeti PÜRÜZSÜZ olan eklemlerde (dönüş açısı küçük) giriş/çıkış
kontrol noktaları ortak teğet doğrultusuna döndürülür (G1 süreklilik).

Güvenlik ilkeleri:
* Uç noktalar ASLA taşınmaz; yalnız kontrol noktaları eklem çevresinde döner.
  Sapma, açı küçük olduğundan piksel-altı düzeydedir.
* Dönüş açısı ``corner_deg`` üzerindeki eklemler KASITLI köşedir; dokunulmaz.
* Çizgi (L) içeren eklemler atlanır: düz çizgiler tasarım öğesidir, bükülmez.
* Yalnız mutlak M/L/C/Z komutlarından oluşan path'ler işlenir (VTracer çıktısı);
  başka komut görülürse path aynen korunur. Hata durumunda dosya değişmez.
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD_RE = re.compile(r"([MLCZmlczHhVvSsQqTtAa])([^MLCZmlczHhVvSsQqTtAa]*)")

Point = tuple[float, float]


def _fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _unit(v: Point) -> Point | None:
    n = _norm(v)
    if n < 1e-9:
        return None
    return (v[0] / n, v[1] / n)


def _angle_between(a: Point, b: Point) -> float:
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return 0.0
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    return math.degrees(math.acos(dot))


def _parse_subpaths(d: str) -> list[dict[str, Any]] | None:
    """d-string'i mutlak M/L/C/Z alt path'lerine ayırır; başka komut -> None.

    Her alt path: {"start": P, "segs": [("L", end) | ("C", c1, c2, end)],
    "closed": bool}
    """
    subpaths: list[dict[str, Any]] = []
    cur: Point | None = None
    sp: dict[str, Any] | None = None

    for m in _CMD_RE.finditer(d or ""):
        cmd = m.group(1)
        nums = [float(x) for x in _NUM_RE.findall(m.group(2))]
        if cmd == "M":
            if len(nums) < 2:
                return None
            if sp is not None:
                subpaths.append(sp)
            cur = (nums[0], nums[1])
            sp = {"start": cur, "segs": [], "closed": False}
            # M sonrası örtük L'ler
            for j in range(2, len(nums) - 1, 2):
                nxt = (nums[j], nums[j + 1])
                sp["segs"].append(("L", nxt))
                cur = nxt
        elif cmd == "L":
            if sp is None or len(nums) < 2:
                return None
            for j in range(0, len(nums) - 1, 2):
                nxt = (nums[j], nums[j + 1])
                sp["segs"].append(("L", nxt))
                cur = nxt
        elif cmd == "C":
            if sp is None or len(nums) < 6 or len(nums) % 6 != 0:
                return None
            for j in range(0, len(nums) - 5, 6):
                c1 = (nums[j], nums[j + 1])
                c2 = (nums[j + 2], nums[j + 3])
                end = (nums[j + 4], nums[j + 5])
                sp["segs"].append(("C", c1, c2, end))
                cur = end
        elif cmd == "Z":
            if sp is None:
                return None
            sp["closed"] = True
        else:
            return None  # desteklenmeyen komut -> path'e dokunma
    if sp is not None:
        subpaths.append(sp)
    return subpaths


def _serialize_subpaths(subpaths: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for sp in subpaths:
        parts.append(f"M{_fmt(sp['start'][0])} {_fmt(sp['start'][1])}")
        for seg in sp["segs"]:
            if seg[0] == "L":
                parts.append(f"L{_fmt(seg[1][0])} {_fmt(seg[1][1])}")
            else:
                _, c1, c2, end = seg
                parts.append(
                    f"C{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} "
                    f"{_fmt(end[0])} {_fmt(end[1])}"
                )
        if sp["closed"]:
            parts.append("Z")
    return " ".join(parts)


def _joint_tangents(
    prev_seg: tuple, next_seg: tuple, joint: Point
) -> tuple[Point, Point] | None:
    """Eklemdeki gelen/giden teğet vektörleri (yalnız C-C eklemleri)."""
    if prev_seg[0] != "C" or next_seg[0] != "C":
        return None
    v_in = (joint[0] - prev_seg[2][0], joint[1] - prev_seg[2][1])   # c2 -> P
    v_out = (next_seg[1][0] - joint[0], next_seg[1][1] - joint[1])  # P -> c1
    return v_in, v_out


def fair_subpath(sp: dict[str, Any], fair_max_deg: float = 25.0, min_deg: float = 1.5) -> int:
    """Bir alt path'in pürüzsüz-niyetli C-C eklemlerinde teğetleri hizalar.

    Dönüş açısı (min_deg, fair_max_deg] aralığındaki eklemler kink sayılır ve
    G1 sürekliliğe çekilir; daha büyük açılar kasıtlı köşe olarak korunur.
    Değişen eklem sayısını döndürür.
    """
    segs = sp["segs"]
    n = len(segs)
    if n < 2:
        return 0
    changed = 0

    # eklem listesi: (prev_idx, next_idx, joint). Kapalı path'te son->ilk eklemi
    # de dahildir (joint = start).
    joints: list[tuple[int, int, Point]] = []
    for i in range(n - 1):
        joints.append((i, i + 1, segs[i][-1]))
    if sp["closed"] and n >= 2:
        last_end = segs[-1][-1]
        if _norm((last_end[0] - sp["start"][0], last_end[1] - sp["start"][1])) < 1e-6:
            joints.append((n - 1, 0, sp["start"]))

    for pi, ni, joint in joints:
        tans = _joint_tangents(segs[pi], segs[ni], joint)
        if tans is None:
            continue
        v_in, v_out = tans
        turn = _angle_between(v_in, v_out)
        if not (min_deg < turn <= fair_max_deg):
            continue
        u_in, u_out = _unit(v_in), _unit(v_out)
        if u_in is None or u_out is None:
            continue
        # ortak teğet: birim vektörlerin ortalaması
        bis = _unit((u_in[0] + u_out[0], u_in[1] + u_out[1]))
        if bis is None:
            continue
        # kontrol noktalarını eklem çevresinde ortak doğrultuya döndür
        _, c1p, c2p, endp = segs[pi]
        len_in = _norm((joint[0] - c2p[0], joint[1] - c2p[1]))
        new_c2 = (joint[0] - bis[0] * len_in, joint[1] - bis[1] * len_in)
        _, c1n, c2n, endn = segs[ni]
        len_out = _norm((c1n[0] - joint[0], c1n[1] - joint[1]))
        new_c1 = (joint[0] + bis[0] * len_out, joint[1] + bis[1] * len_out)
        segs[pi] = ("C", c1p, new_c2, endp)
        segs[ni] = ("C", new_c1, c2n, endn)
        changed += 1
    return changed


def count_curve_kinks(d: str, fair_max_deg: float = 25.0, min_deg: float = 1.5) -> tuple[int, int]:
    """(kink_sayısı, C-C eklem sayısı) döndürür — ölçüm/regresyon için."""
    subpaths = _parse_subpaths(d)
    if not subpaths:
        return (0, 0)
    kinks = 0
    total = 0
    for sp in subpaths:
        segs = sp["segs"]
        n = len(segs)
        joints: list[tuple[int, int, Point]] = [(i, i + 1, segs[i][-1]) for i in range(n - 1)]
        if sp["closed"] and n >= 2:
            last_end = segs[-1][-1]
            if _norm((last_end[0] - sp["start"][0], last_end[1] - sp["start"][1])) < 1e-6:
                joints.append((n - 1, 0, sp["start"]))
        for pi, ni, joint in joints:
            tans = _joint_tangents(segs[pi], segs[ni], joint)
            if tans is None:
                continue
            total += 1
            turn = _angle_between(*tans)
            if min_deg < turn <= fair_max_deg:
                kinks += 1
    return (kinks, total)


def fair_svg_curves(svg_path: Path, fair_max_deg: float = 25.0) -> dict[str, Any]:
    """SVG'deki tüm uygun path'lerde eğri eklemlerini pürüzsüzleştirir.

    Yalnız ``d`` attribute'ları yeniden yazılır; fill/transform/sıra korunur.
    Hata durumunda dosya değiştirilmez (çökme yok).
    """
    svg_path = Path(svg_path)
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        logger.debug("Curve fairing: SVG parse edilemedi, atlandı: %s", e)
        return {"status": "skipped", "error": str(e)}

    paths_processed = 0
    joints_faired = 0
    changed_any = False
    for el in root.iter():
        if el.tag.split("}")[-1] != "path":
            continue
        d = el.get("d")
        if not d or "C" not in d:
            continue
        try:
            subpaths = _parse_subpaths(d)
        except Exception:  # noqa: BLE001
            subpaths = None
        if not subpaths:
            continue
        changed = 0
        for sp in subpaths:
            changed += fair_subpath(sp, fair_max_deg=fair_max_deg)
        paths_processed += 1
        if changed:
            el.set("d", _serialize_subpaths(subpaths))
            joints_faired += changed
            changed_any = True

    if changed_any:
        try:
            tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Curve fairing: SVG yazılamadı: %s", e)
            return {"status": "failed", "error": str(e)}

    return {
        "status": "completed" if changed_any else "no_change",
        "paths_processed": paths_processed,
        "joints_faired": joints_faired,
    }
