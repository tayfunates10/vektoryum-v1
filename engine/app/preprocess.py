"""Profil bazlı ön işleme katmanı.

Her vektörleştirme profili için ayrı bir ön işleme fonksiyonu vardır. Tek
giriş noktası ``preprocess_for_mode`` bu fonksiyonlara dağıtım yapar ve
``(çıktı_yolu, rapor)`` döndürür.

Tasarım ilkeleri:
* Geometrik logolarda düz çizgi ve köşeleri korumak için yalnızca çok hafif
  kenar-koruyan filtre + flat palet indirgeme uygulanır (Gaussian blur yok).
* Palet sertleştirme yalnızca kanonik siyah/beyaz/kırmızıya çok yakın renkleri
  yaslar; kurumsal diğer renkler korunur.
* Çizgi kalınlığını değiştiren morfoloji kullanılmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Kanonik renkler (RGB)
_CANON = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "dark_gray": (45, 45, 45),
}


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _rgba_to_rgb_on_white(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 3:
        return arr
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    rgb = arr[..., :3].astype(np.float32)
    white = np.full_like(rgb, 255.0)
    return (rgb * alpha + white * (1 - alpha)).astype(np.uint8)


def _foreground_mask_from_alpha(arr: np.ndarray) -> np.ndarray | None:
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    alpha = arr[..., 3]
    if np.all(alpha > 250):
        return None
    return (alpha > 128).astype(np.uint8) * 255


def _quantize_flat(rgb: np.ndarray, colors: int, dither: bool = False) -> np.ndarray:
    pil = Image.fromarray(rgb)
    q = pil.quantize(
        colors=max(2, colors),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE,
    )
    return np.array(q.convert("RGB"))


def _quantize_flat_fg_aware(rgb: np.ndarray, colors: int) -> np.ndarray:
    """Zemin-farkındalıklı düz palet kuantizasyonu.

    MEDIANCUT nüfus tabanlıdır: büyük tek renk zeminli logolarda kutuların çoğu
    zemin tonlarına gider ve İNCE ÇİZGİLER tek çamur renge çöküp kesiklenir
    (siyah+kırmızı+mavi çizgilerin tek renkte birleşmesi gerçek bir hataydı).
    Burada zemin (köşe medyanı) ayrılır ve kümeler YALNIZCA ön plan piksellerine
    harcanır (LAB k-means). Zemin düz değilse eski global yola düşülür.
    """
    h, w = rgb.shape[:2]
    pw, ph = max(8, w // 12), max(8, h // 12)
    corners = np.concatenate([
        rgb[:ph, :pw].reshape(-1, 3), rgb[:ph, -pw:].reshape(-1, 3),
        rgb[-ph:, :pw].reshape(-1, 3), rgb[-ph:, -pw:].reshape(-1, 3),
    ]).astype(np.float32)
    bg = np.median(corners, axis=0)
    uniform = float((np.linalg.norm(corners - bg, axis=1) < 18).mean())
    if uniform < 0.85:
        return _quantize_flat(rgb, colors)

    flat = rgb.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(flat - bg, axis=1)
    fg_idx = dist > 40
    fg_ratio = float(fg_idx.mean())
    n_fg = int(fg_idx.sum())
    if n_fg < 50 or fg_ratio > 0.6:
        return _quantize_flat(rgb, colors)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    fg_lab = lab[fg_idx]
    unique_n = len(np.unique((fg_lab // 8).astype(np.int32), axis=0))
    K = max(2, min(colors - 1, unique_n))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if len(fg_lab) > 60000:
        rng = np.random.default_rng(0)
        fit = fg_lab[rng.choice(len(fg_lab), 60000, replace=False)]
    else:
        fit = fg_lab
    _c, _l, centers = cv2.kmeans(fit, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.float32)

    labels = np.empty(len(fg_lab), dtype=np.int32)
    chunk = 200000
    for s in range(0, len(fg_lab), chunk):
        block = fg_lab[s:s + chunk]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[s:s + chunk] = np.argmin(d2, axis=1)
    centers_rgb = cv2.cvtColor(
        centers.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    ).reshape(-1, 3)

    out = np.empty_like(flat, dtype=np.uint8)
    out[~fg_idx] = np.clip(np.round(bg), 0, 255).astype(np.uint8)
    out[fg_idx] = centers_rgb[labels]
    return out.reshape(rgb.shape)


def _harden_palette(rgb: np.ndarray, tol: int = 42) -> np.ndarray:
    """Kanonik siyah/beyaza ve BASKIN kırmızıya yakın renkleri tam değere yaslar.

    Kırmızı yaslama hedefi görüntünün kendi baskın kırmızısıdır: anti-alias
    pembeleri temizlenir ama koyu/marka kırmızısı (ör. 214,40,40) zorla saf
    (255,0,0)'a çevrilip RENK BOZULMAZ. Baskın kırmızı zaten saf kırmızıya çok
    yakınsa kanonik değere yaslanır (eski davranış korunur).
    """
    out = rgb.copy()
    flat = out.reshape(-1, 3).astype(np.int32)

    def _near(target: tuple[int, int, int], t: int) -> np.ndarray:
        diff = flat - np.array(target, dtype=np.int32)
        return np.sqrt((diff * diff).sum(axis=1)) <= t

    # beyaz (geniş tolerans), siyah
    flat[_near(_CANON["white"], tol + 18)] = _CANON["white"]
    flat[_near(_CANON["black"], tol)] = _CANON["black"]
    # kırmızı: doygun VE açık (anti-alias pembe) kırmızılar. Mutlak g/b sınırı
    # yerine bağıl baskınlık kullanılır; böylece ince kırmızı çizgiler (anti-alias
    # nedeniyle pembeleşmiş) kaybolmadan kırmızıya yaslanır.
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    red_like = (r >= 130) & (r > g + 45) & (r > b + 45) & (np.abs(g.astype(np.int32) - b.astype(np.int32)) <= 60)
    if np.any(red_like):
        reds = flat[red_like]
        sat = reds.max(axis=1) - reds.min(axis=1)
        # hedefi çizgi ÇEKİRDEĞİ belirlesin, anti-alias saçağı değil: en doygun
        # kırmızıların (>= 0.6 * maks doygunluk) en kalabalık 16'lık kovası
        core = reds[sat >= 0.6 * int(sat.max())] if int(sat.max()) > 0 else reds
        buckets = core // 16
        uniq, counts = np.unique(buckets, axis=0, return_counts=True)
        modal = uniq[np.argmax(counts)]
        members = core[(buckets == modal).all(axis=1)]
        target = members.mean(axis=0) if len(members) else core.mean(axis=0)
        canon_red = np.array(_CANON["red"], dtype=np.float64)
        if float(np.linalg.norm(target - canon_red)) <= tol + 22:
            target = canon_red
        flat[red_like] = np.round(target).astype(np.int32)
    return flat.reshape(out.shape).astype(np.uint8)


def _reduce_to_dominant(
    rgb: np.ndarray,
    k: int,
    protect_chromatic: bool = True,
    protect_min_ratio: float = 0.0004,
    max_protected: int = 3,
    erode_iters: int = 1,
) -> np.ndarray:
    """Her pikseli en sık görülen k renkten en yakınına atar (sert palet).

    Çıktıda yalnızca k (artı korunan renkler) düz renk kalır; ara/anti-alias
    tonları yok olur. Sert kenarlar oluştuğundan VTracer renkleri yeniden
    çoğaltamaz.

    İki koruma, alanı küçük diye GERÇEK tasarım öğelerinin silinmesini önler:

    * CANLI renkler (doygunluk >= 60, ör. ince kırmızı/mavi çizgi) k kotasına
      giremese de palete eklenir.
    * BAĞIMSIZ İNCE KONTURLAR: yalnızca ince şerit olarak yaşayan (1px erozyonda
      kaybolan) ve çevresi ZEMİNLE çevrili nötr renkler (ör. beyaz zeminde 1px
      gri çizgi). Bunlar en yakın baskın renge gömülürse çizgi ZEMİNE düşüp
      tamamen silinebilir; korunarak palete eklenir.

    Toplam korunan renk sayısı ``max_protected`` ile sınırlıdır.
    """
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    if len(colors) <= k:
        return rgb
    order = np.argsort(counts)[::-1]
    colors = colors[order]
    counts = counts[order]
    total = float(flat.shape[0])
    dominant_list = [colors[i].astype(np.int32) for i in range(min(k, len(colors)))]
    bg = dominant_list[0]

    if protect_chromatic and len(colors) > k:
        kernel = np.ones((3, 3), np.uint8)
        added = 0
        for i in range(k, len(colors)):
            if added >= max_protected:
                break
            if counts[i] / total < protect_min_ratio or counts[i] < 60:
                continue
            c = colors[i].astype(np.int32)
            sat = int(c.max()) - int(c.min())
            if sat >= 60:
                dominant_list.append(c)
                added += 1
                continue
            # nötr aday: bağımsız ince kontur mu? (ince + zeminle çevrili)
            exact = colors[i].astype(rgb.dtype)
            mask = cv2.inRange(rgb, exact, exact)
            n_mask = int(np.count_nonzero(mask))
            if n_mask == 0:
                continue
            survival = float(np.count_nonzero(cv2.erode(mask, kernel, iterations=erode_iters))) / float(n_mask)
            if survival > 0.25:
                continue  # kalın bölge; kotaya giremediyse gerçekten önemsiz
            ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
            n_ring = int(np.count_nonzero(ring))
            if n_ring == 0:
                continue
            bg_exact = np.array(bg, dtype=rgb.dtype)
            bg_frac = float(np.count_nonzero(
                ring & np.all(rgb == bg_exact[None, None, :], axis=2)
            )) / float(n_ring)
            if bg_frac >= 0.7:
                dominant_list.append(c)
                added += 1

    dominant = np.array(dominant_list, dtype=np.int32)
    # her piksel için en yakın dominant renk (parça parça, bellek dostu)
    out = np.empty_like(flat)
    chunk = 200000
    for start in range(0, len(flat), chunk):
        block = flat[start:start + chunk].astype(np.int32)
        d = ((block[:, None, :] - dominant[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk] = dominant[np.argmin(d, axis=1)]
    return out.reshape(rgb.shape)


def _mixture_anchors(
    c: np.ndarray,
    anchors: list[np.ndarray],
    tol: float,
) -> list[int] | None:
    """c rengi, verilen çapa renklerin İKİLİ ya da ÜÇLÜ karışımı mı?

    Önce çiftler denenir (doğru parçası üzerinde), sonra üçlüler (üçgen içinde).
    Kırmızı-siyah sınırındaki koyu bordo gibi ÜÇ rengin (kırmızı+siyah+beyaz)
    karışımı olan anti-alias tonları çift testinden kaçar; üçgen testi yakalar.
    Karışımsa çapa indekslerinin listesi, değilse None döner.
    """
    n = len(anchors)
    for j in range(n):
        for m in range(j + 1, n):
            a, b = anchors[j], anchors[m]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue
            t = float(np.dot(c - a, ab) / denom)
            if not (0.04 <= t <= 0.96):
                continue
            if float(np.linalg.norm(c - (a + t * ab))) <= tol:
                return [j, m]
    for j in range(n):
        for m in range(j + 1, n):
            for p in range(m + 1, n):
                a, b, d = anchors[j], anchors[m], anchors[p]
                u, v, w_ = b - a, d - a, c - a
                uu, uv, vv = float(np.dot(u, u)), float(np.dot(u, v)), float(np.dot(v, v))
                wu, wv = float(np.dot(w_, u)), float(np.dot(w_, v))
                det = uu * vv - uv * uv
                if det < 1e-9:
                    continue
                s = (vv * wu - uv * wv) / det
                t = (uu * wv - uv * wu) / det
                if s < -0.02 or t < -0.02 or s + t > 1.02:
                    continue
                proj = a + s * u + t * v
                if float(np.linalg.norm(c - proj)) <= tol:
                    return [j, m, p]
    return None


def _absorb_aa_films(
    rgb: np.ndarray,
    colinear_tol: float = 26.0,
    max_survival: float = 0.25,
    anchor_factor: float = 0.5,
    erode_iters: int = 1,
) -> np.ndarray:
    """Quantize sonrası anti-alias FİLM renklerini komşu baskın renge gömer.

    Film: karşılaştırılabilir baskınlıktaki çapa renklerin KARIŞIMI olan
    (ikili: doğru üzerinde, üçlü: üçgen içinde) ve görüntüde yalnızca ince şerit
    olarak yaşayan (1px erozyonda kaybolan) renk — siyah kenar çevresindeki gri
    şerit, kırmızı çizgi çevresindeki pembe saçak, kırmızı-siyah sınırındaki
    koyu bordo. Bu tonlar elenmezse ``_reduce_to_dominant`` k kotasını işgal
    edip GERÇEK renkleri (ince mavi çizgi gibi) dışarı itebilir ve çıktı SVG'de
    kenarlar boyunca kirli ara-ton bandı kalır.

    İki koruma gerçek tasarım öğelerini tutar:
    * Kalın bölgeler erozyonda hayatta kalır -> gerçek (gri dahil) renk alanları
      film sayılmaz.
    * KOMŞULUK KORUMASI: gerçek bir film, karıştığı MÜREKKEP renklerinin
      sınırında yaşar. Çevresi yalnızca zeminle çevrili ince bir renk bağımsız
      bir ÇİZGİDİR (ör. beyaz zeminde tek başına gri/bordo çizgi); gömülürse
      çizgi tamamen SİLİNECEĞİNDEN korunur.
    """
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    if not (3 <= len(colors) <= 16):
        return rgb
    order = np.argsort(counts)[::-1]
    colors = colors[order].astype(np.int32)
    counts = counts[order]
    bg_exact = colors[0].astype(rgb.dtype)  # en kalabalık renk = zemin
    out = rgb
    kernel = np.ones((3, 3), np.uint8)
    absorbed: set[int] = set()

    for i in range(len(colors) - 1, -1, -1):  # en seyrek renkten başla
        c = colors[i].astype(np.float64)
        candidates = [
            j for j in range(len(colors))
            if j != i and j not in absorbed and counts[j] >= counts[i] * anchor_factor
        ]
        anchor_rgbs = [colors[j].astype(np.float64) for j in candidates]
        mix = _mixture_anchors(c, anchor_rgbs, colinear_tol)
        if mix is None:
            continue
        anchor_idx = [candidates[j] for j in mix]
        exact = colors[i].astype(rgb.dtype)
        mask = cv2.inRange(out, exact, exact)
        total = int(np.count_nonzero(mask))
        if total == 0:
            continue
        # süperörneklemede AA filmleri 2px genişler; erozyon ölçekle uyumlu
        # olmazsa film 'kalın bölge' sanılıp palete sızar (halo artışı ölçüldü)
        survival = float(np.count_nonzero(cv2.erode(mask, kernel, iterations=erode_iters))) / float(total)
        if survival > max_survival:
            continue  # kalın bölge -> gerçek renk, koru
        # komşuluk koruması: halka, zemin DIŞI çapa renklerden iz taşımalı
        ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
        n_ring = int(np.count_nonzero(ring))
        ink_anchors = [
            colors[j] for j in anchor_idx
            if not np.array_equal(colors[j].astype(rgb.dtype), bg_exact)
        ]
        if n_ring > 0 and ink_anchors:
            ink_hits = np.zeros(ring.shape, dtype=bool)
            for a in ink_anchors:
                a_exact = a.astype(rgb.dtype)
                ink_hits |= np.all(out == a_exact[None, None, :], axis=2)
            frac = float(np.count_nonzero(ring & ink_hits)) / float(n_ring)
            if frac < 0.08:
                continue  # bağımsız ince çizgi -> koru
        # en yakın çapaya göm
        dists = [float(np.linalg.norm(c - colors[j].astype(np.float64))) for j in anchor_idx]
        target = colors[anchor_idx[int(np.argmin(dists))]].astype(rgb.dtype)
        if out is rgb:
            out = rgb.copy()
        out[mask > 0] = target
        absorbed.add(i)
    return out


def _remove_speckles(rgb: np.ndarray, min_area: int = 6, protect_chromatic: bool = False) -> np.ndarray:
    """Renk bölgelerindeki çok küçük izole lekeleri komşuya gömerek temizler.

    ``protect_chromatic``: CANLI (doygunluk >= 60) renklerin benekleri korunur.
    Ateş közleri/kıvılcımlar gibi küçük ama kasıtlı parlak noktalar tasarımın
    parçasıdır; gürültü benekleri ise tipik olarak nötr/gri tonlardadır
    (közlerin silinip sadakati düşürmesi gerçek bir hataydı).

    PERFORMANS: her küçük bileşen, tüm görsel yerine kendi bounding-box (ROI)
    üzerinde işlenir. Eski sürüm her bileşen için tam görselde dilate/maske
    yapıyordu (1500² × binlerce bileşen) ve pipeline süresinin ~%75'ini yiyordu.
    ROI ile maliyet bileşen boyutuyla orantılı olur.
    """
    out = rgb.copy()
    h, w = out.shape[:2]
    colors = np.unique(out.reshape(-1, 3), axis=0)
    if len(colors) > 48:
        return out  # çok renkli görselde atla (48: hata-güdümlü ek kümeler dahil)
    kernel = np.ones((3, 3), np.uint8)
    for color in colors:
        if protect_chromatic and int(color.max()) - int(color.min()) >= 60:
            continue
        mask = cv2.inRange(out, color, color)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # bileşeni +1px pad'li yerel pencerede işle
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(w, x + cw + 1), min(h, y + ch + 1)
            comp = labels[y0:y1, x0:x1] == i
            dil = cv2.dilate(comp.astype(np.uint8), kernel) > 0
            ring = dil & ~comp
            if not ring.any():
                continue
            sub = out[y0:y1, x0:x1]
            fill = np.median(sub[ring].reshape(-1, 3), axis=0).astype(np.uint8)
            sub[comp] = fill
    return out


# ---------------------------------------------------------------------------
# Profil fonksiyonları
# ---------------------------------------------------------------------------
def preprocess_geometric_logo(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    # süperörneklenmiş girdide AA filmleri ölçekle genişler; incelik testleri
    # erozyonu aynı ölçekle yapmalı
    scale = 2 if report.get("supersampled") else 1
    # ince çizgileri birbirine karıştırmamak için çok hafif kenar-koruyan filtre
    filtered = cv2.bilateralFilter(rgb, d=3, sigmaColor=18, sigmaSpace=18)
    report["steps"].append("bilateral_very_light")
    quant = _quantize_flat_fg_aware(filtered, colors=8)
    report["steps"].append("quantize_fg_aware_8")
    # AA film tonlarını (kenar grileri/pembe saçak) gerçek renklerden önce göm;
    # yoksa reduce kotasını işgal edip ince renkli çizgileri dışarı itebilirler
    quant = _absorb_aa_films(quant, erode_iters=scale)
    report["steps"].append("absorb_aa_films")
    hard = _harden_palette(quant, tol=46)
    report["steps"].append("palette_harden_bwr")
    # küçük lekeleri temizle (ince çizgileri korumak için küçük eşik)
    cleaned = _remove_speckles(hard, min_area=3)
    report["steps"].append("despeckle")
    # SON adım: sert palet -> en baskın 4 renk (+ korunan canlı aksanlar);
    # ara/median tonları garanti elenir
    reduced = _reduce_to_dominant(cleaned, k=4, erode_iters=scale)
    report["steps"].append("reduce_to_dominant_4")
    report["palette"] = _palette_list(reduced)
    return reduced


def preprocess_minimal_ai(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    scale = 2 if report.get("supersampled") else 1
    filtered = cv2.bilateralFilter(rgb, d=5, sigmaColor=35, sigmaSpace=35)
    report["steps"].append("bilateral_light")
    quant = _quantize_flat_fg_aware(filtered, colors=6)
    report["steps"].append("quantize_fg_aware_6")
    quant = _absorb_aa_films(quant, erode_iters=scale)
    report["steps"].append("absorb_aa_films")
    hard = _harden_palette(quant, tol=38)
    report["steps"].append("palette_harden_bwr")
    reduced = _reduce_to_dominant(hard, k=5, erode_iters=scale)
    report["steps"].append("reduce_to_dominant_5")
    report["palette"] = _palette_list(reduced)
    return reduced


def _assign_to_centers(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Tüm örnekleri en yakın merkeze atar (parça parça, bellek dostu)."""
    labels = np.empty(len(samples), dtype=np.int32)
    chunk = 200000
    for s in range(0, len(samples), chunk):
        block = samples[s:s + chunk]
        d2 = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[s:s + chunk] = np.argmin(d2, axis=1)
    return labels


def _kmeans_quantize_lab(
    rgb: np.ndarray,
    k: int,
    edge_preserve: bool = True,
    refine_high_error: bool = True,
    noise_merge_tol: float = 11.0,
) -> np.ndarray:
    """LAB renk uzayında k-means ile algısal kuantizasyon.

    RGB'de değil LAB'de kümelendiği için renkler insan algısına göre ayrılır;
    çıktı az sayıda DÜZ ve temiz renk bölgesidir (Vectorizer.AI benzeri).

    İki ek adım çıktı NETLİĞİNİ korur:

    * **Hata-güdümlü ek merkezler** (``refine_high_error``): k-means nüfus
      yanlıdır; alanı küçük ama FARKLI tondaki öğeler (turuncu portakal, kırmızı
      domates, yeşil biber) kümesiz kalıp komşu tona boyanır — gerçek bir renk
      hatasıydı. Atama sonrası hâlâ yüksek hatalı (ΔE>18) kalan piksellerden
      yeni merkezler açılır (en fazla +8) ve herkes yeniden atanır.
    * **Hedefli gürültü birleştirme** (``noise_merge_tol``): İKİSİ de nötr
      (düşük kroma) olan ve en az biri leke deseninde (çok parçalı) dağılan iki
      yakın merkez birleştirilir — koyu panel/zemin vinyetinin JPEG gürültüsüyle
      benekli lekeye bölünmesini engeller. Kasıtlı gradyan bantları (ateş,
      duman, krom) birleştirilmez; genel bir merkez birleştirme SSIM'i düşürdüğü
      için YOKTUR (gerçek bir gerilemeydi).
    """
    img = rgb
    if edge_preserve:
        # kenar-koruyan güçlü düzleştirme -> düz renk bölgeleri (gürültü/gradyan azalır)
        img = cv2.bilateralFilter(rgb, d=9, sigmaColor=45, sigmaSpace=45)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # DERİN KENAR HARİTASI (opsiyonel, HED): anlamsal nesne/yazı sınırlarında
    # güçlü, JPEG gürültüsü ve doku pürüzünde sessiz. Varsa Sobel ile HARMANLANIR
    # (aşağıdaki _energy); yoksa None kalır ve tüm kararlar salt Sobel'le, önceki
    # davranışla birebir aynı alınır (çökme yok, zorunlu bağımlılık yok).
    from app.dl_segmentation import compute_edge_map
    dl_edge = compute_edge_map(img)

    def _energy(gmag_arr: np.ndarray) -> np.ndarray:
        """Kenar enerjisi: <=1 düz bölge, >1 yapılı bölge (Sobel + derin kenar)."""
        e = gmag_arr / 35.0
        if dl_edge is not None:
            e = 0.5 * e + 0.5 * (dl_edge.reshape(gmag_arr.shape) / 0.12)
        return e

    # DÜZ-NÖTR bölgelerde gürültüyü KAYNAĞINDA düzleştir: vinyetli koyu panel
    # zeminindeki JPEG gürültüsü, kümeleme sınırlarını benekli lekeye çevirir.
    # Düz bölgede (düşük kroma + düşük kenar enerjisi) kaybolacak detay yoktur;
    # kenar/yazıdan >= 7px içeri erozyonla çekildiği için keskin öğeler etkilenmez.
    L0 = lab[:, :, 0].astype(np.float32)
    g0x = cv2.Sobel(L0, cv2.CV_32F, 1, 0, ksize=3)
    g0y = cv2.Sobel(L0, cv2.CV_32F, 0, 1, ksize=3)
    gmag0 = np.sqrt(g0x * g0x + g0y * g0y)
    chroma0 = np.hypot(lab[:, :, 1].astype(np.float32) - 128.0,
                       lab[:, :, 2].astype(np.float32) - 128.0)
    flat_zone0 = ((chroma0 <= 12.0) & (_energy(gmag0) <= 1.0)).astype(np.uint8)
    safe_zone = cv2.erode(flat_zone0, np.ones((13, 13), np.uint8)) > 0
    if float(safe_zone.mean()) > 0.03:
        blurred = cv2.GaussianBlur(lab, (15, 15), 0)
        lab = lab.copy()
        lab[safe_zone] = blurred[safe_zone]

    samples = lab.reshape(-1, 3).astype(np.float32)
    unique_n = len(np.unique(samples.astype(np.uint8), axis=0))
    K = max(2, min(int(k), unique_n))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    rng = np.random.default_rng(0)

    # HIZ: merkezleri alt-örneklemde bul, sonra TÜM pikselleri en yakın merkeze ata
    if len(samples) > 60000:
        idx = rng.choice(len(samples), 60000, replace=False)
        fit = samples[idx]
    else:
        fit = samples
    _compactness, _labels, centers = cv2.kmeans(fit, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.float32)
    labels_full = _assign_to_centers(samples, centers)

    # yerel kenar enerjisi (hem augmentasyon hem gürültü birleştirme kapısı):
    # düz alan = vinyet/gürültü bölgesi; yapılı alan = kasıtlı detay.
    # Sobel + (varsa) derin kenar haritası harmanı (_energy): <=1 düz, >1 yapılı.
    L_ch = lab[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(L_ch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_ch, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    energy = _energy(gmag)
    energy_flat = energy.reshape(-1)
    chroma_flat = np.hypot(samples[:, 1] - 128.0, samples[:, 2] - 128.0)

    if refine_high_error:
        max_extra = 10
        # DÜZ-NÖTR alandaki DAĞINIK yüksek-hata pikselleri yeni merkez açamaz:
        # koyu panel vinyetinin gürültüsü ek gri merkezlerle yeniden bölünüp
        # leke üretiyordu (augmentasyon ile gürültü birleştirme çatışması).
        # BÜYÜK ve BÜTÜNSEL (>= 400px bağlı bileşen) yüksek-hata bölgeleri ise
        # k-means'in kaçırdığı GERÇEK öğelerdir (tabela çerçevesi gibi) ve her
        # zaman küme açabilir — kapı yalnız kırıntıları eler.
        flat_neutral = (chroma_flat <= 12.0) & (energy_flat <= 1.0)
        img_h, img_w = lab.shape[:2]
        for _round in range(3):
            if len(centers) - K >= max_extra:
                break
            err = np.linalg.norm(samples - centers[labels_full], axis=1)
            hi = err > 18.0
            hi_img = hi.reshape(img_h, img_w).astype(np.uint8)
            n_comp, comp_lbl, comp_stats, _ = cv2.connectedComponentsWithStats(hi_img, connectivity=8)
            big = np.zeros(n_comp, dtype=bool)
            if n_comp > 1:
                big[1:] = comp_stats[1:, cv2.CC_STAT_AREA] >= 400
            hi_big = big[comp_lbl].reshape(-1)
            hi = hi & (hi_big | ~flat_neutral)
            hi_ratio = float(hi.mean())
            if hi_ratio < 0.001:
                break
            hi_samples = samples[hi]
            if len(hi_samples) > 40000:
                hi_samples = hi_samples[rng.choice(len(hi_samples), 40000, replace=False)]
            k2 = int(min(4, max_extra - (len(centers) - K),
                         len(np.unique(hi_samples.astype(np.uint8), axis=0))))
            if k2 < 1:
                break
            try:
                _c2, _l2, new_centers = cv2.kmeans(
                    np.ascontiguousarray(hi_samples), k2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
                )
            except cv2.error:
                break
            centers = np.vstack([centers, np.clip(new_centers, 0, 255).astype(np.float32)])
            labels_full = _assign_to_centers(samples, centers)

    if noise_merge_tol > 0 and len(centers) > 2:
        counts = np.bincount(labels_full, minlength=len(centers)).astype(np.float64)
        remap = np.arange(len(centers))
        merged = centers.astype(np.float64).copy()
        label_img = labels_full.reshape(lab.shape[:2])
        # etiket bölgesi altındaki yerel L-gradyanı (yukarıda hesaplandı):
        # gürültü lekesi görsel olarak DÜZ (vinyetli panel) alanda yaşar ->
        # gradyan küçük; duman/doku gibi kasıtlı gri detaylar yapılıdır ->
        # gradyan büyük ve KORUNUR.
        frag_cache: dict[int, bool] = {}
        grad_cache: dict[int, float] = {}

        def _is_fragmented(idx: int) -> bool:
            """Etiketin bölgesi çok sayıda küçük parçaya mı dağılmış (leke deseni)?"""
            if idx in frag_cache:
                return frag_cache[idx]
            mask = (label_img == idx).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            frag = (n - 1) >= 8 and float(np.median(areas)) <= 500.0 if n > 1 else False
            frag_cache[idx] = frag
            return frag

        def _mean_energy(idx: int) -> float:
            if idx in grad_cache:
                return grad_cache[idx]
            m = label_img == idx
            g = float(energy[m].mean()) if m.any() else 0.0
            grad_cache[idx] = g
            return g

        def _noise_pair(i: int, j: int) -> bool:
            # yalnız İKİSİ de nötr (düşük kroma) VE düz alanda yaşayan (düşük
            # kenar enerjisi) VE en az biri leke deseninde dağılan çiftler
            # birleştirilir. OpenCV LAB'de a=b=128 nötrdür. Duman gibi yapılı
            # gri detayların enerjisi yüksektir; onlara dokunulmaz (dumanın
            # zemine gömülmesi gerçek bir gerilemeydi).
            a, b = merged[i], merged[j]
            if float(np.linalg.norm(a - b)) > noise_merge_tol:
                return False
            ca = float(np.hypot(a[1] - 128.0, a[2] - 128.0))
            cb = float(np.hypot(b[1] - 128.0, b[2] - 128.0))
            if ca > 12.0 or cb > 12.0:
                return False
            if _mean_energy(i) > 1.0 or _mean_energy(j) > 1.0:
                return False
            return _is_fragmented(i) or _is_fragmented(j)

        # TEK geçiş: yinelemeli/zincirli birleştirme merkezleri kaydıra kaydıra
        # gerçek öğeleri (çerçeve tonu gibi) zemine gömüyordu — gerçek bir
        # gerilemeydi. Tek geçişte yalnız doğrudan komşu ton çiftleri birleşir.
        for i in range(len(centers)):
            if remap[i] != i:
                continue
            for j in range(i + 1, len(centers)):
                if remap[j] != j:
                    continue
                if _noise_pair(i, j):
                    total = counts[i] + counts[j]
                    if total > 0:
                        merged[i] = (merged[i] * counts[i] + merged[j] * counts[j]) / total
                    counts[i] = total
                    remap[j] = i
        if (remap != np.arange(len(centers))).any():
            # remap'i köke indir (i sonradan h'ye katılmışsa j -> i -> h)
            for idx in range(len(remap)):
                root = idx
                while remap[root] != root:
                    root = remap[root]
                remap[idx] = root
            centers = np.clip(merged, 0, 255).astype(np.float32)
            labels_full = remap[labels_full]

    # Merkezleri üye MEDYANIYLA yeniden hesapla: k-means ortalaması, bölgenin
    # parlak/doygun çekirdeğini gölgeli kenar pikselleriyle SOLUKLAŞTIRIR
    # (parlak kırmızı dilimin kiremit rengine dönmesi gerçek bir canlılık
    # kaybıydı). Medyan, bölgenin baskın gerçek rengini temsil eder.
    for idx in np.unique(labels_full):
        members = samples[labels_full == idx]
        if len(members):
            centers[idx] = np.clip(np.median(members, axis=0), 0, 255)

    quant_lab = centers.astype(np.uint8)[labels_full].reshape(lab.shape)
    out = cv2.cvtColor(quant_lab, cv2.COLOR_LAB2RGB)
    return out


def preprocess_logo_color(arr: np.ndarray, report: dict, n_colors: int = 20) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    # LAB k-means: algısal, temiz, düz renk bölgeleri (+ hata-güdümlü aksan
    # kümeleri + nötr leke birleştirme, bkz. _kmeans_quantize_lab)
    quant = _kmeans_quantize_lab(rgb, k=n_colors, edge_preserve=True)
    report["steps"].append(f"lab_kmeans_{n_colors}")
    # çok küçük NÖTR lekeleri komşuya gömerek bölge sınırlarını temizle;
    # canlı renk benekleri (köz/kıvılcım) tasarımın parçasıdır, korunur.
    # Çok renkli/dokulu görsellerde (>28 renk: grunge fırça dokusu, illüstrasyon)
    # nötr benekler de tasarımdır -> tümüyle atlanır.
    n_quant = int(len(np.unique(quant.reshape(-1, 3), axis=0)))
    cleaned = _remove_speckles(quant, min_area=8, protect_chromatic=True) if n_quant <= 28 else quant
    report["steps"].append("despeckle")
    # SVG palet konsolidasyonu tavanı: hata-güdümlü eklenen aksan kümeleri
    # (domates kırmızısı, portakal turuncusu) istenen k'yı aşabilir; tavan
    # gerçek renk sayısına bağlanır ki konsolidasyon aksanları geri kırpmasın.
    report["actual_color_count"] = int(len(np.unique(cleaned.reshape(-1, 3), axis=0)))
    report["palette"] = _palette_list(cleaned, limit=24)
    return cleaned


def preprocess_lineart(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Otsu; ince çizgileri korumak için morfoloji yok
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    report["steps"].append("otsu_threshold")
    # çok küçük izole benekleri temizle: ince çizgili görsellerde VTracer speckle
    # filtresi kapatıldığından (çizgi silinmesin) gürültü kontrolü burada yapılır
    inv = 255 - binary
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            inv[labels == i] = 0
    report["steps"].append("despeckle")
    return 255 - inv


def preprocess_single_color(arr: np.ndarray, report: dict) -> np.ndarray:
    fg = _foreground_mask_from_alpha(arr)
    if fg is not None:
        report["steps"].append("alpha_foreground_mask")
        binary = 255 - fg  # konu siyah, zemin beyaz
    else:
        rgb = _rgba_to_rgb_on_white(arr)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        report["steps"].append("otsu_threshold")
    # küçük lekeleri temizle
    inv = 255 - binary
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 8:
            inv[labels == i] = 0
    report["steps"].append("despeckle")
    return 255 - inv


def preprocess_centerline(arr: np.ndarray, report: dict) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    report["steps"].append("otsu_threshold_for_skeleton")
    return binary


def preprocess_photo_poster(arr: np.ndarray, report: dict, n_colors: int = 16) -> np.ndarray:
    rgb = _rgba_to_rgb_on_white(arr)
    quant = _kmeans_quantize_lab(rgb, k=n_colors, edge_preserve=True)
    report["steps"].append(f"lab_kmeans_{n_colors}")
    report["palette"] = _palette_list(quant, limit=n_colors)
    report["note"] = "Fotoğraf modu: çıktı posterize bir yaklaşımdır, tam sadakat garanti edilmez."
    return quant


def _palette_list(rgb: np.ndarray, limit: int = 12) -> list[str]:
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:limit]
    return [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in colors[order]]


_DISPATCH = {
    "geometric_logo": preprocess_geometric_logo,
    "minimal_ai": preprocess_minimal_ai,
    "flat_logo": preprocess_minimal_ai,
    "logo_color": preprocess_logo_color,
    "lineart": preprocess_lineart,
    "single_color": preprocess_single_color,
    "centerline": preprocess_centerline,
    "photo_poster": preprocess_photo_poster,
}


def _auto_color_count(analysis: dict[str, Any] | None) -> int:
    """Analizdeki renk zenginliğine göre logo_color için k seçer (16-64).

    Gerçek görsel survey'i, sabit ~22 renk cap'inin renk-zengini logolarda ΔE'yi
    11-14'e fırlattığını gösterdi (renk açlığı). Tavan yükseltildi; cap bu değere
    bağlanır (bkz. pipeline) ki üretilen renkler kırpılıp boşa gitmesin.

    Kademeli ek bütçe (referans vektörleştiricilerin içerik-ölçekli renk
    sayısıyla uyumlu; ör. foto-zengin amblemde ~77 renk):
    * est >= 18 (ton-zengini illüstrasyon): +8 — koyu tonlar turuncu-orta
      tonlara çekilip derinlik kaybolmasın (SSIM ölçülür biçimde yükselir).
    * est >= 22 (foto-zengin görsel): +16 daha — sebze/meyve gibi çok tonlu
      fotoğrafik bölgelerde ton merdiveni zenginleşir (ΔE 5.1 -> 4.6 ölçüldü).
    """
    if not analysis:
        return 22
    est = int(analysis.get("estimated_color_count", 14))
    k = est + 10 + (8 if est >= 18 else 0) + (16 if est >= 22 else 0)
    return int(max(16, min(64, k)))


def preprocess_for_mode(
    image_path: Path,
    mode: str,
    output_dir: Path,
    analysis: dict[str, Any] | None = None,
    color_override: int | None = None,
    output_suffix: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Görseli seçilen moda göre ön işler ve PNG olarak kaydeder.

    ``color_override`` verilirse renkli modlarda (logo_color/photo_poster)
    otomatik renk sayısı yerine bu değer kullanılır — refinement döngüsü ΔE'yi
    düşürmek için daha yüksek k ile yeniden ön işleme yapabilsin diye.
    ``output_suffix`` çıktı dosya adına eklenir (refinement varyantları orijinali
    ezmesin diye).
    """
    image = Image.open(image_path).convert("RGBA")

    # ALT-PİKSEL SINIR YAKLAŞIMI (süperörnekleme): küçük girdilerde bölge
    # sınırları piksel ızgarasına oturur ve eğriler tırtıklı izlenir. Girdi
    # 2x LANCZOS ile büyütülürse anti-alias gradyanı sınırı büyütülmüş ızgarada
    # ara konuma yerleştirir; kuantizasyon + izleme bu ince ızgarada çalışır ve
    # eğriler orijinal ızgaraya göre yarım-piksel hassasiyet kazanır. Büyük
    # girdiler zaten yeterli örneklem taşır; maliyet nedeniyle uygulanmaz.
    max_side = max(image.size)
    report_supersample = None
    if max_side < 700:
        new_size = (image.size[0] * 2, image.size[1] * 2)
        image = image.resize(new_size, Image.LANCZOS)
        report_supersample = {"from": max_side, "scale": 2}

    # PERFORMANS: çok büyük girdileri trace öncesi küçült (vektör çıktı sonsuz
    # ölçeklenir; sadakat karşılaştırması zaten 512px). Renkli modlar k-means +
    # despeckle nedeniyle pahalı -> daha agresif sınır; sade modlar keskin kenar
    # için biraz daha yüksek kalsın.
    cap = 1100 if mode in ("logo_color", "photo_poster") else 1400
    max_side = max(image.size)
    report_resize = None
    if max_side > cap:
        scale = cap / max_side
        new_size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
        image = image.resize(new_size, Image.LANCZOS)
        report_resize = {"from": [max_side, max_side], "to": list(image.size)}

    arr = np.array(image)

    report: dict[str, Any] = {"mode": mode, "steps": []}
    if report_supersample:
        report["supersampled"] = report_supersample
        report["steps"].append("supersample_2x")
    if report_resize:
        report["resized"] = report_resize
    func = _DISPATCH.get(mode, preprocess_minimal_ai)
    if mode == "logo_color":
        n_colors = int(color_override) if color_override else _auto_color_count(analysis)
        n_colors = max(8, min(64, n_colors))
        report["auto_color_count"] = n_colors
        processed = preprocess_logo_color(arr, report, n_colors=n_colors)
    elif mode == "photo_poster" and color_override:
        k = max(8, min(64, int(color_override)))
        report["auto_color_count"] = k
        processed = preprocess_photo_poster(arr, report, n_colors=k)
    else:
        processed = func(arr, report)

    output_path = output_dir / f"preprocessed_{mode}{output_suffix}.png"
    if processed.ndim == 2:
        Image.fromarray(processed, mode="L").save(output_path)
    else:
        Image.fromarray(processed).save(output_path)

    report["output"] = str(output_path)
    return output_path, report
