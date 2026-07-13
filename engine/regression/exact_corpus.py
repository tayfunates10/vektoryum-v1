"""FAZ 0 — Exact-final SVG hata reproduction korpusu (ÖZGÜN sentetik + oracle).

Canlı testte kanıtlanan hata ailelerini (T1 topology, T2 gradient+alpha, T3
micro-detail, T5 low-res over-vectorization) DETERMINIST sentetik fixture'larla
üretir; her fixture yanında KAYNAK ORACLE metadata'sı tutulur (beklenen bileşen/
delik sayısı, alpha/gradient varlığı, korunacak ROI). Kapalı üründen kopya YOK.

Bu korpus, exact final SVG değerlendirici (FinalArtifactEvaluator) + sonraki
faz motorlarının regresyon oracle'ıdır. Fixture adı/oracle production pipeline'a
verilmez (anti-overfit) — yalnız test doğrulaması içindir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Marka palet (sentetik)
BLACK = (10, 10, 10)
RED = (227, 0, 11)
YELLOW = (255, 237, 0)
WHITE = (250, 250, 250)
BLUE = (20, 60, 190)


@dataclass
class Fixture:
    name: str
    rgb: np.ndarray                       # (H,W,3) uint8 (beyaz zemin görünümü)
    alpha: np.ndarray | None              # (H,W) uint8 veya None
    oracle: dict[str, Any] = field(default_factory=dict)


def t1_topology(n: int = 300) -> Fixture:
    """Düz renk + nested hole + bitişik renk yüzleri + T-junction + küçük noktalar."""
    img = np.full((n, n, 3), WHITE, np.uint8)
    # bitişik iki renk yüz (ortak sınır) — sol siyah, sağ kırmızı
    cv2.rectangle(img, (int(n * .1), int(n * .1)), (int(n * .5), int(n * .6)), BLACK, -1)
    cv2.rectangle(img, (int(n * .5), int(n * .1)), (int(n * .9), int(n * .6)), RED, -1)
    # alt tek yüz (sarı) → üç yüz bir çizgide (T-junction hattı)
    cv2.rectangle(img, (int(n * .1), int(n * .6)), (int(n * .9), int(n * .85)), YELLOW, -1)
    # nested hole: siyah blok içinde beyaz delik + içinde kırmızı ada
    cv2.rectangle(img, (int(n * .18), int(n * .18)), (int(n * .34), int(n * .42)), WHITE, -1)
    cv2.rectangle(img, (int(n * .22), int(n * .22)), (int(n * .30), int(n * .38)), RED, -1)
    # küçük noktalar (2/4/6px yarıçap benzeri kare) — korunacak
    for i, s in enumerate((2, 4, 6)):
        cx = int(n * (0.2 + i * 0.12)); cy = int(n * 0.92)
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), BLUE, -1)
    return Fixture("t1_topology", img, None, {
        "class": "geometric", "source_has_alpha": False,
        "expected_gradients": 0,
        "protected_rois": ["dot_2px", "dot_4px", "dot_6px", "nested_red_island"],
        "topology_note": "bileşen/delik source==render (delta 0) zorunlu",
    })


def t2_gradient_alpha(n: int = 300) -> Fixture:
    """İki linear + bir radial gradient + alpha gradient + yarı saydam örtüşme."""
    img = np.zeros((n, n, 3), np.float32)
    x = np.linspace(0, 1, n)[None, :].repeat(n, 0)
    y = np.linspace(0, 1, n)[:, None].repeat(n, 1)
    # yatay linear (kırmızı→sarı)
    img[..., 0] = 227 * (1 - x) + 255 * x
    img[..., 1] = 0 * (1 - x) + 237 * x
    img[..., 2] = 11 * (1 - x) + 0 * x
    # radial (merkez mavi) üstüne
    cy, cx = n * 0.5, n * 0.5
    r = np.sqrt((np.arange(n)[:, None] - cy) ** 2 + (np.arange(n)[None, :] - cx) ** 2)
    rad = np.clip(1 - r / (n * 0.35), 0, 1)
    for c, v in zip(range(3), BLUE):
        img[..., c] = img[..., c] * (1 - rad) + v * rad
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    # alpha gradient (dikey): üst şeffaf, alt opak
    alpha = np.clip((y * 255), 0, 255).astype(np.uint8)
    return Fixture("t2_gradient_alpha", rgb, alpha, {
        "class": "clean_logo", "source_has_alpha": True,
        "expected_gradients": 3, "expected_linear": 2, "expected_radial": 1,
        "gradient_note": "düz-band flatten kabul edilmez; SSIM-black+white birlikte",
        "alpha_gate": {"iou_min": 0.995, "mae_max": 0.005},
    })


def t3_micro_detail(n: int = 320) -> Fixture:
    """1px halkalar, küçük ®/glyph benzeri, 3/5/7px noktalar, nested circles."""
    img = np.full((n, n, 3), WHITE, np.uint8)
    # ince halka (1px) — kapalı kalmalı
    cv2.circle(img, (int(n * .3), int(n * .3)), int(n * .18), BLACK, 1)
    # nested circles (counter)
    cv2.circle(img, (int(n * .7), int(n * .3)), int(n * .16), RED, -1)
    cv2.circle(img, (int(n * .7), int(n * .3)), int(n * .08), WHITE, -1)
    # ® benzeri: küçük daire + içinde işaret
    rc = (int(n * .82), int(n * .82)); rr = max(6, int(n * .05))
    cv2.circle(img, rc, rr, BLACK, 1)
    cv2.line(img, (rc[0] - rr // 2, rc[1]), (rc[0] + rr // 2, rc[1]), BLACK, 1)
    # 3/5/7px noktalar (ayrı bileşen)
    for i, s in enumerate((3, 5, 7)):
        cx = int(n * (0.2 + i * 0.14)); cy = int(n * 0.7)
        cv2.circle(img, (cx, cy), s, BLUE, -1)
    return Fixture("t3_micro_detail", img, None, {
        "class": "lineart", "source_has_alpha": False,
        "protected_rois": ["thin_ring_1px", "registered_mark", "dot_3px", "dot_5px", "dot_7px"],
        "counter_note": "nested counter/hole korunmalı; ® kaybolmamalı",
    })


def t5_lowres_jpeg(n: int = 160) -> tuple[Fixture, bytes]:
    """Düşük çöz. mikro rozet; JPEG Q32 sıkıştırma → over-vectorization tuzağı."""
    img = np.full((n, n, 3), WHITE, np.uint8)
    cv2.circle(img, (n // 2, n // 2), int(n * .4), RED, -1)
    cv2.circle(img, (n // 2, n // 2), int(n * .22), WHITE, -1)
    cv2.rectangle(img, (int(n * .42), int(n * .42)), (int(n * .58), int(n * .58)), BLACK, -1)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 32])
    jpeg_bytes = enc.tobytes() if ok else b""
    fx = Fixture("t5_lowres_jpeg", img, None, {
        "class": "photo", "source_has_alpha": False,
        "source_bytes": len(jpeg_bytes),
        "complexity_budget": {"max_paths": 500, "max_bytes": 150_000},
        "note": "JPEG bloklarını path'e çevirme; over-vectorization hard fail",
    })
    return fx, jpeg_bytes


def all_fixtures() -> list[Fixture]:
    return [t1_topology(), t2_gradient_alpha(), t3_micro_detail(), t5_lowres_jpeg()[0]]
