"""Opsiyonel kenar temizleme geçişi (edge_cleanup): tırtıklı kenar + katman lekesi.

Bu geçiş VARSAYILAN KAPALIDIR ve yalnız kullanıcı açtığında (API/arayüz) çalışır;
böylece varsayılan çıktı ve tüm regresyon fixture'ları bit-bit korunur
("diğerlerini etkilemeden" güvencesi). İki cerrahi düzeltmeyi birleştirir:

1) KONTUR YUMUŞATMA (contour_smooth): tırtıklı organik kenarları özellik-koruyan
   Taubin ile pürüzsüzleştirir. Piksel-sadakat metriği gürültülü JPEG'i birebir
   eşleşmeyi ödüllendirdiğinden yumuşatma metriği hep hafif düşürür (metrik-göz
   ayrışması); bu yüzden fidelity kapısı DEĞİL, bilinçli-tercih (opt-in) kapısı
   kullanılır — ama yine de büyük düşüş güvenlik toleransıyla reddedilir.

2) ADA-YUTMA (absorb_islands): büyük düz bir bölgenin içine hapsolmuş küçük,
   rengi kopuk artefakt-adaları (JPEG bloğu / renk sızması) baskın komşusunun
   rengine çekilir. ÖLÇÜM KORUMALI: yalnız fidelity artar ya da nötr kalırsa
   uygulanır (bu adalar genelde orijinalde yoktur -> düzeltmek sadakati artırır).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

_ISLAND_MAX_PX = 1400       # bundan büyük görünür bölge "ada" sayılmaz
_ISLAND_MIN_PX = 6          # bundan küçük zaten despeckle kapsamı
_BORDER_FRAC = 0.55         # sınırın en az bu kadarı baskın komşuya değmeli
                            # (dokulu bölgede tam kuşatma nadir; ölçüm kapısı korur)
_NEIGHBOR_RATIO = 2.5       # baskın komşu, adadan en az bu kat büyük olmalı
_ABSORB_MAX_DE = 60.0       # ada ile komşu farkı bundan büyükse bilinçli tasarım


def absorb_islands_svg(svg_path: Path, original_path: Path, out_path: Path) -> dict[str, Any]:
    """Büyük düz bölgeye gömülü küçük artefakt-adalarını komşu rengine çeker.

    ID-render ile her path'in görünür bölgesi bulunur; sınırının >= _BORDER_FRAC
    kadarı tek, çok daha büyük bir komşuya değen küçük ada, o komşunun dolgusuna
    boyanır (geometri silinmez — stack sürprizi olmaz). Benimseme ölçümle
    çağırana bırakılır; bu modül yalnız yazar. Dönen: {"absorbed": n}.
    """
    from app.color_refit import (_build_id_svg, _decode_id_map, _hex,  # noqa: PLC0415
                                 _iter_paths, _parse_fill)
    from app.fidelity import load_reference_rgb, render_svg_to_rgb  # noqa: PLC0415

    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(svg_path))
    except Exception as e:  # noqa: BLE001
        return {"absorbed": 0, "error": f"parse: {e}"}
    paths = _iter_paths(tree.getroot())
    if len(paths) < 2:
        return {"absorbed": 0}
    try:
        ref, (w, h) = load_reference_rgb(Path(original_path), max_side=1400)
    except Exception as e:  # noqa: BLE001
        return {"absorbed": 0, "error": f"referans: {e}"}

    id_tree, _ = _build_id_svg(tree)
    id_svg = Path(out_path).with_suffix(".idm.svg")
    try:
        id_tree.write(str(id_svg), encoding="utf-8", xml_declaration=True)
        id_rgb = render_svg_to_rgb(id_svg, w, h)
    finally:
        id_svg.unlink(missing_ok=True)
    if id_rgb is None:
        return {"absorbed": 0, "error": "render yok"}
    id_map = _decode_id_map(id_rgb)
    n_bins = len(paths) + 2
    counts = np.bincount(id_map.reshape(-1).clip(0, n_bins - 1), minlength=n_bins)

    fills = [_parse_fill(el.get("fill")) for el in paths]
    lab_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    absorbed = 0
    for i, el in enumerate(paths):
        code = i + 1
        if code >= 0xFFFFFF:
            code += 1
        area = int(counts[code]) if code < n_bins else 0
        if area < _ISLAND_MIN_PX or area > _ISLAND_MAX_PX:
            continue
        if fills[i] is None:
            continue
        mask = (id_map == code).astype(np.uint8)
        border = (cv2.dilate(mask, kernel) - mask).astype(bool)
        nb_codes = id_map[border]
        nb_codes = nb_codes[(nb_codes != code)]
        if len(nb_codes) < 8:
            continue
        vals, cnts = np.unique(nb_codes, return_counts=True)
        j = int(vals[int(np.argmax(cnts))])          # baskın komşu kodu
        frac = float(cnts.max()) / float(len(nb_codes))
        if frac < _BORDER_FRAC:
            continue
        nb_idx = j - 1
        if nb_idx < 0 or nb_idx >= len(paths) or fills[nb_idx] is None:
            continue
        if counts[j] < _NEIGHBOR_RATIO * area:        # komşu yeterince büyük mü
            continue
        # ada, komşusuyla anlamlı renk farkı taşımalı (yoksa zaten görünmez);
        # ama devasa fark (>_ABSORB_MAX_DE) bilinçli kontrast olabilir -> dokunma
        from app.color_refit import _delta_e  # noqa: PLC0415
        de_nb = _delta_e(fills[i], fills[nb_idx])
        if de_nb < 3.0 or de_nb > _ABSORB_MAX_DE:
            continue
        # ARTEFAKT testi: ada rengi, orijinaldeki o bölgenin gerçek renginden de
        # uzak mı? (orijinalde gerçekten varsa detaydır -> koru)
        me_lab = lab_ref[mask.astype(bool)].mean(axis=0)
        isl_lab = np.array(_lab_of(fills[i]))
        if float(np.linalg.norm(me_lab - isl_lab)) < 10.0:
            continue  # ada rengi orijinalle uyumlu -> gerçek detay, dokunma
        el.set("fill", _hex(fills[nb_idx]))
        absorbed += 1

    if absorbed == 0:
        return {"absorbed": 0}
    try:
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    except Exception as e:  # noqa: BLE001
        return {"absorbed": 0, "error": f"yazma: {e}"}
    return {"absorbed": absorbed}


def _lab_of(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def apply_edge_cleanup(
    svg_path: Path, original_path: Path, out_path: Path
) -> dict[str, Any]:
    """Kontur yumuşatma + ada-yutmayı sırayla uygular (ölçüm korumalı).

    Her adım ayrı ölçülür; yumuşatma büyük fidelity düşüşünde (tolerans
    _SMOOTH_TOL) reddedilir, ada-yutma yalnız fidelity artar/nötr kalırsa tutulur.
    Dönen rapor adım bazlı sonuçları içerir. Hiçbir adım tutmazsa girdi aynen
    kopyalanır ve applied=False döner.
    """
    from shutil import copyfile  # noqa: PLC0415

    from app.contour_smooth import smooth_svg_contours  # noqa: PLC0415
    from app.curve_refit import refit_svg_curves  # noqa: PLC0415
    from app.fidelity import (compute_fidelity, load_reference_rgb,  # noqa: PLC0415
                              render_svg_to_rgb)

    _SMOOTH_TOL = 0.6  # yumuşatma için kabul edilebilir azami fidelity düşüşü
    _REFIT_TOL = 1.5   # Bézier basitleştirme için kabul edilebilir azami düşüş
                       # (metrik-göz ayrışması: dosya/segment %25 düşerken
                       # sadakat ~1 puan düşer; görsel olarak eğri daha akıcı)

    src = Path(svg_path)
    dst = Path(out_path)
    report: dict[str, Any] = {"applied": False, "smoothed": 0, "refit": 0, "absorbed": 0}

    try:
        ref, (w, h) = load_reference_rgb(Path(original_path), max_side=1024)
    except Exception:  # noqa: BLE001
        ref = None

    def _fid(p: Path) -> float | None:
        if ref is None:
            return None
        r = render_svg_to_rgb(p, w, h)
        return compute_fidelity(ref, r)["fidelity_score"] if r is not None else None

    base_fid = _fid(src)
    cur = src
    cur_fid = base_fid          # cur'ün sadakati her aşamada izlenir -> gereksiz
                                # render yok (büyük SVG'de render darboğazdır)
    tmp_a = dst.with_suffix(".smooth.svg")
    tmp_r = dst.with_suffix(".refit.svg")
    tmp_b = dst.with_suffix(".island.svg")

    # 1) kontur yumuşatma (opt-in; büyük düşüşte reddet)
    try:
        rs = smooth_svg_contours(cur, tmp_a)
        if rs.get("smoothed_paths"):
            f = _fid(tmp_a)
            # None yalnız TABAN da ölçülemediğinde kabul edilir (ölçüm imkânsız).
            # Taban ölçülüp temiz SVG render edilemiyorsa (f is None) REDDET —
            # render edilebilir kazananı doğrulanmamış dosyayla değiştirme.
            if base_fid is None or (f is not None and f >= base_fid - _SMOOTH_TOL):
                report["smoothed"] = rs["smoothed_paths"]
                report["smooth_fidelity"] = f
                cur, cur_fid = tmp_a, f
    except Exception as e:  # noqa: BLE001
        logger.debug("kontur yumuşatma atlandı: %s", e)

    # 2) eğri basitleştirme (Schneider kübik Bézier; tolerans kapılı). Aşırı
    #    segmentasyonu akıcı Bézier'lere indirger — dosya/segment küçülür, çentik
    #    azalır. Metrik-göz ayrışması nedeniyle fidelity kapısı DEĞİL, tolerans
    #    kapısı kullanılır; büyük düşüş yine reddedilir.
    try:
        f_pre = cur_fid            # cur zaten ölçülü; yeniden render etme
        rr = refit_svg_curves(cur, tmp_r)
        if rr.get("refit_paths"):
            f = _fid(tmp_r)
            # None yalnız önceki aşama da ölçülemediyse kabul edilir; ölçülü
            # cur render edilebilirken refit render edilemiyorsa (f is None) REDDET.
            if f_pre is None or (f is not None and f >= f_pre - _REFIT_TOL):
                report["refit"] = rr["refit_paths"]
                report["refit_seg_before"] = rr.get("seg_before")
                report["refit_seg_after"] = rr.get("seg_after")
                report["refit_fidelity"] = f
                cur, cur_fid = tmp_r, f
    except Exception as e:  # noqa: BLE001
        logger.debug("eğri basitleştirme atlandı: %s", e)

    # 3) ada-yutma (ölçüm korumalı: yalnız fidelity artar/nötr)
    try:
        ri = absorb_islands_svg(cur, original_path, tmp_b)
        if ri.get("absorbed"):
            f_before = cur_fid     # cur zaten ölçülü; yeniden render etme
            f_after = _fid(tmp_b)
            # eşik gevşetildiğinden (dokulu bölge) ada-yutma yalnız fidelity'yi
            # GERÇEKTEN artırırsa tutulur — gerçek artefakt-adası kaldırmak
            # sadakati yükseltir, meşru detayı ezmek düşürür (ölçüm ayırır)
            if f_before is not None and f_after is not None and f_after > f_before:
                report["absorbed"] = ri["absorbed"]
                report["island_fidelity"] = f_after
                cur, cur_fid = tmp_b, f_after
    except Exception as e:  # noqa: BLE001
        logger.debug("ada-yutma atlandı: %s", e)

    if cur == src:
        copyfile(src, dst)
        return report
    if cur != dst:
        from shutil import copyfile as _cp  # noqa: PLC0415
        _cp(cur, dst)
    report["applied"] = True
    report["final_fidelity"] = cur_fid     # cur'ün ölçülü sadakati (render yok)
    for t in (tmp_a, tmp_r, tmp_b):
        if t != dst:
            t.unlink(missing_ok=True)
    return report
