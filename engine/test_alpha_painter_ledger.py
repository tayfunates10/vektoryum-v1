"""FAZ 3B.1 — painter encoding attempt ledger + hata önceliği + seçim politikası.

Kök observability hatası şuydu: ``apply_candidate_painter_reconstruction`` içindeki
tek ``last_error`` değişkeni her encoding ve stroke denemesinde eziliyordu; en son
denenen ``rect`` byte hatası, contour'un GERÇEK doğrulama/journal reddini gizliyordu
(public-05: yanıltıcı ``rect:792169>416187``). Bu modül üç bağımsız garantiyi test
eder:

1. LEDGER — her (encoding, stroke) denemesi yapılandırılmış, güvenli sayısal
   telemetriyle kaydedilir (ham SVG / path ``d`` / kaynak byte YOK), deterministik
   JSON (``sort_keys`` + sabit ayraç) ile stderr'e yazılır ve içerik SHA'sı verilir.
2. HATA ÖNCELİĞİ — hiçbir aday kabul edilmezse ana hata, son (rect byte) denemesi
   değil, öncelik merdivenine göre EN ANLAMLI olandır (validation başlatan ilk exact
   → ilk quantized → en küçük byte'lı byte-rejected exact → quantized → yok).
3. SEÇİM POLİTİKASI — iki aşamalı: önce exact adaylar; yalnız hiçbir exact geçmezse
   quantized. Bütçeye giren + TÜM değişmemiş kapıları (alfa + journal geometri) geçen
   adaylardan kazanan en küçük byte'lıdır. Bir exact geçerse quantized denenmez.

Eşik/evaluator/TransformJournal/corpus değişmez; fail-open yoktur.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.alpha_candidate_knockout import _write_tree_to_temp
from app.alpha_candidate_painter import (
    _emit_painter_attempts,
    _painter_primary_error,
    _requantize_alpha,
    _short_error_code,
    apply_candidate_painter_reconstruction,
    build_painter_reconstruction_tree,
)
from app.source_truth import render_svg_to_rgba

# Ledger'a yazılması İZİNLİ tek anahtar kümesi. Herhangi bir ham SVG/path/kaynak
# sızıntısı bu kümeyi ihlal eder (test 5 bunu doğrular).
_ALLOWED_LEDGER_KEYS = {
    "stroke_width",
    "encoding_label",
    "encoding_family",
    "exact_or_quantized",
    "source_alpha_level_count",
    "encoded_alpha_level_count",
    "actual_serialized_bytes",
    "byte_limit",
    "projected_path_count",
    "actual_path_count",
    "path_limit",
    "projected_node_count",
    "actual_node_count",
    "node_limit",
    "preflight_status",
    "validation_started",
    "validation_stage",
    "status",
    "exact_error_code",
    "native_alpha_iou",
    "native_alpha_mae",
    "bounded_alpha_iou",
    "bounded_alpha_mae",
    "evaluator_alpha_iou",
    "evaluator_alpha_mae",
    "artwork_fingerprint_match",
    "journal_gate_started",
    "journal_passed",
    "journal_reason_codes",
}

_ATTEMPTS_PREFIX = "source_alpha_candidate_painter_attempts="


# --------------------------------------------------------------------------- #
# Kaynak/parent üreticileri (self-contained).                                 #
# --------------------------------------------------------------------------- #
def _simple_soft_disc_source(path: Path, side: int = 96) -> None:
    """Tek yumuşak disk: az bileşen → polygon/rect bütçeye sığar (bir exact geçer)."""
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = cy = side / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip((side * 0.34 - dist) * 40.0, 0, 255).astype(np.uint8)
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[:, :, :3] = (180, 20, 30)
    rgba[:, :, 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(path)


_COMPLEX_SIDE = 360


def _complex_multi_glyph_source(path: Path, side: int = _COMPLEX_SIDE) -> None:
    """Yoğun döşeli yumuşak-halka glifleri: polygon/rect byte'ı aşar, contour bütçeye
    girer ama node sayısı journal node kapısını patlatır (node_complexity_explosion).
    Alfa 0.995-üretilebilir (büyük binary gövde IoU'yu sabitler)."""
    step, bsize = 12, 7
    alpha = np.zeros((side, side), dtype=np.float32)
    for by in range(6, side - bsize - 4, step):
        for bx in range(6, side - bsize - 4, step):
            body = np.zeros((side, side), dtype=bool)
            body[by : by + bsize, bx : bx + bsize] = True
            alpha[body] = 255.0
            for distance, value in ((1, 176.0), (2, 64.0)):
                grown = np.zeros((side, side), dtype=bool)
                grown[
                    by - distance : by + bsize + distance,
                    bx - distance : bx + bsize + distance,
                ] = True
                ring = grown & ~body & (alpha == 0.0)
                alpha[ring] = value
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    rgba[:, :, :3] = (180, 20, 30)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _covering_parent(path: Path, side: int) -> bytes:
    inner = side - 8
    data = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}">'
        f'<path d="M0 0H{side}V{side}H0Z" fill="#FFFFFF"/>'
        f'<path d="M4 4H{inner}V{inner}H4Z" fill="#1E5AC8"/>'
        '</svg>'
    ).encode("utf-8")
    path.write_bytes(data)
    return data


def _capture_attempts(func) -> tuple[Any, list[list[dict[str, Any]]], BaseException | None]:
    """``func``'u çağır, stderr'e yazılan attempt ledger JSON satırlarını ayrıştır.

    (dönüş_değeri, [ledger, ...], yakalanan_exception) döner — exception ledger
    yazımını engellemez (hata yolunda da emit edilir)."""
    stderr = io.StringIO()
    value: Any = None
    error: BaseException | None = None
    with contextlib.redirect_stderr(stderr):
        try:
            value = func()
        except BaseException as exc:  # noqa: BLE001 - test tüm hata yolunu inceler
            error = exc
    ledgers: list[list[dict[str, Any]]] = []
    for line in stderr.getvalue().splitlines():
        if line.startswith(_ATTEMPTS_PREFIX):
            ledgers.append(json.loads(line[len(_ATTEMPTS_PREFIX):]))
    return value, ledgers, error


# --------------------------------------------------------------------------- #
# Saf hata-önceliği / emit birim üreticileri.                                 #
# --------------------------------------------------------------------------- #
def _ledger_entry(**override: Any) -> dict[str, Any]:
    """Tüm izinli alanları taşıyan nötr bir ledger kaydı (testler alanı ezer)."""
    entry: dict[str, Any] = {
        "stroke_width": 1.0,
        "encoding_label": "polygon",
        "encoding_family": "polygon",
        "exact_or_quantized": "exact",
        "source_alpha_level_count": 3,
        "encoded_alpha_level_count": 3,
        "actual_serialized_bytes": 1000,
        "byte_limit": 250000,
        "projected_path_count": 2,
        "actual_path_count": None,
        "path_limit": 502,
        "projected_node_count": None,
        "actual_node_count": None,
        "node_limit": 2510,
        "preflight_status": "within_budget",
        "validation_started": True,
        "validation_stage": "accepted",
        "status": "accepted",
        "exact_error_code": "",
        "native_alpha_iou": None,
        "native_alpha_mae": None,
        "bounded_alpha_iou": None,
        "bounded_alpha_mae": None,
        "evaluator_alpha_iou": None,
        "evaluator_alpha_mae": None,
        "artwork_fingerprint_match": None,
        "journal_gate_started": False,
        "journal_passed": None,
        "journal_reason_codes": [],
    }
    entry.update(override)
    return entry


# --------------------------------------------------------------------------- #
# Encoding-eşliği yardımcıları (polygon vs contour maske alfa düzlemi).       #
# --------------------------------------------------------------------------- #
def _equivalence_parent(side: int) -> ET.Element:
    return ET.fromstring(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}">'
        f'<path d="M0 0H{side}V{side}H0Z" fill="#123456"/></svg>'
    )


def _render_encoding_alpha(
    parent: ET.Element,
    quantized: np.ndarray,
    opacity_by_level: dict[int, float],
    encoding: str,
    render_side: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "candidate.svg"
        target.write_text("<svg/>")
        root, _geometry = build_painter_reconstruction_tree(
            parent, None, quantized, opacity_by_level, 1.5,
            mask_encoding=encoding, transaction_id="equiv-txn",
        )
        svg = _write_tree_to_temp(root, target)
        rendered = render_svg_to_rgba(svg, render_side, render_side)
        svg.unlink(missing_ok=True)
    assert rendered is not None, f"{encoding}@{render_side} render edilemedi"
    return rendered[:, :, 3].astype(np.int16)


class PainterLedgerErrorPriorityTests(unittest.TestCase):
    """Saf hata-önceliği: rect byte, contour'un gerçek doğrulama hatasını EZMEZ."""

    def test_01_rect_byte_does_not_overwrite_contour_validation_error(self) -> None:
        # polygon+rect byte-rejected; contour validation başlatıp journal reddi aldı.
        attempts = [
            _ledger_entry(
                encoding_label="polygon", encoding_family="polygon",
                actual_serialized_bytes=368259, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:polygon:368259>250215",
            ),
            _ledger_entry(
                encoding_label="contour", encoding_family="contour",
                actual_serialized_bytes=68940, validation_started=True,
                validation_stage="journal_geometry", status="geometry_rejected",
                journal_gate_started=True, journal_passed=False,
                journal_reason_codes=["node_complexity_explosion"],
                exact_error_code="source_alpha_candidate_painter_journal_geometry_rejected:node_complexity_explosion",
            ),
            _ledger_entry(
                encoding_label="rect", encoding_family="rect",
                actual_serialized_bytes=339374, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:rect:339374>250215",
            ),
        ]
        message = _painter_primary_error(attempts, "deadbeef")
        self.assertIn("primary=contour", message)
        self.assertIn("node_complexity_explosion", message)
        self.assertNotIn("primary=rect", message)
        self.assertIn("attempts_sha256=deadbeef", message)

    def test_09_contour_native_iou_reject_is_primary(self) -> None:
        # polygon byte-rejected; contour bütçeye girip native IoU kapısında düştü →
        # ana hata contour'un GERÇEK IoU kodu olmalı (rect byte değil).
        attempts = [
            _ledger_entry(
                encoding_label="polygon", encoding_family="polygon",
                actual_serialized_bytes=400000, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:polygon:400000>250000",
            ),
            _ledger_entry(
                encoding_label="contour", encoding_family="contour",
                actual_serialized_bytes=60000, validation_started=True,
                validation_stage="native_alpha", status="native_alpha_rejected",
                native_alpha_iou=0.981234,
                exact_error_code="source_alpha_candidate_painter_native_iou_gate_failed:0.981234<0.995",
            ),
            _ledger_entry(
                encoding_label="rect", encoding_family="rect",
                actual_serialized_bytes=330000, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:rect:330000>250000",
            ),
        ]
        message = _painter_primary_error(attempts, "cafef00d")
        self.assertIn("primary=contour", message)
        self.assertIn("native_iou_gate_failed:0.981234", message)
        self.assertNotIn("primary=rect", message)

    def test_10_all_exact_over_byte_selects_smallest_byte_primary(self) -> None:
        # Hiçbir aday validation başlatmadı (hepsi byte-rejected) → öncelik en küçük
        # byte'lı exact byte-rejected adaydır (contour = 300000).
        attempts = [
            _ledger_entry(
                encoding_label="polygon", encoding_family="polygon",
                actual_serialized_bytes=400000, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:polygon:400000>250000",
            ),
            _ledger_entry(
                encoding_label="contour", encoding_family="contour",
                actual_serialized_bytes=300000, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:contour:300000>250000",
            ),
            _ledger_entry(
                encoding_label="rect", encoding_family="rect",
                actual_serialized_bytes=350000, preflight_status="over_budget",
                validation_started=False, validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:rect:350000>250000",
            ),
        ]
        message = _painter_primary_error(attempts, "0011")
        self.assertIn("primary=contour", message)
        self.assertIn("300000>250000", message)

    def test_priority_quantized_used_only_when_no_exact_validated(self) -> None:
        # Exact hepsi byte-rejected; quantized validation başlatıp düştü → validation
        # başlatan quantized, byte-rejected exact'tan ÖNCELİKLİDİR (öncelik 2 > 3).
        attempts = [
            _ledger_entry(
                encoding_label="polygon", encoding_family="polygon",
                exact_or_quantized="exact", actual_serialized_bytes=400000,
                preflight_status="over_budget", validation_started=False,
                validation_stage=None, status="byte_rejected",
                exact_error_code="source_alpha_candidate_painter_byte_budget_rejected:polygon:400000>250000",
            ),
            _ledger_entry(
                encoding_label="contour-q32", encoding_family="contour",
                exact_or_quantized="quantized", actual_serialized_bytes=40000,
                validation_started=True, validation_stage="bounded_alpha",
                status="bounded_alpha_rejected",
                exact_error_code="source_alpha_candidate_painter_iou_gate_failed:0.990000<0.995",
            ),
        ]
        message = _painter_primary_error(attempts, "abcd")
        self.assertIn("primary=contour-q32", message)
        self.assertIn("iou_gate_failed", message)

    def test_priority_empty_attempts_reports_no_candidate(self) -> None:
        message = _painter_primary_error([], "ffff")
        self.assertIn("primary=none", message)
        self.assertIn("no_candidate", message)
        self.assertIn("attempts_sha256=ffff", message)

    def test_short_error_code_strips_prefix(self) -> None:
        self.assertEqual(
            _short_error_code(
                "source_alpha_candidate_painter_journal_geometry_rejected:seam_regression"
            ),
            "journal_geometry_rejected:seam_regression",
        )
        self.assertEqual(_short_error_code("foreign_code"), "foreign_code")


class PainterLedgerEmitTests(unittest.TestCase):
    """Emit: güvenli telemetri, deterministik JSON, ham veri sızıntısı yok."""

    def test_04_emit_is_deterministic_across_key_order(self) -> None:
        # Aynı içerik, farklı dict-anahtar ekleme sırası → aynı payload + aynı SHA.
        entry_a = _ledger_entry(encoding_label="contour", actual_serialized_bytes=42)
        entry_b = {k: entry_a[k] for k in reversed(list(entry_a.keys()))}
        first, ledgers_a, _ = _capture_attempts(lambda: _emit_painter_attempts([entry_a]))
        second, ledgers_b, _ = _capture_attempts(lambda: _emit_painter_attempts([entry_b]))
        self.assertEqual(first, second)
        self.assertEqual(ledgers_a, ledgers_b)
        # Dönen SHA, yayınlanan payload'un gerçek sha256'sıdır.
        payload = json.dumps([entry_a], sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_05_emit_contains_no_raw_svg_or_path_or_source(self) -> None:
        entry = _ledger_entry(
            encoding_label="contour",
            exact_error_code="source_alpha_candidate_painter_native_iou_gate_failed:0.9<0.995",
        )
        _sha, ledgers, _ = _capture_attempts(lambda: _emit_painter_attempts([entry]))
        self.assertEqual(len(ledgers), 1)
        emitted = ledgers[0]
        self.assertEqual(len(emitted), 1)
        # Yalnız izinli sayısal/enum/hata-kodu anahtarları; ham geometri anahtarı yok.
        self.assertEqual(set(emitted[0]), _ALLOWED_LEDGER_KEYS)
        payload = json.dumps(emitted, separators=(",", ":"))
        for forbidden in ("<svg", "<path", '"d":', "points", "viewBox", "M0", "H", "rgb("):
            self.assertNotIn(forbidden, payload)


class PainterLedgerIntegrationTests(unittest.TestCase):
    """Gerçek turnuva çalıştırıp ledger içeriğini ve seçimi doğrular."""

    def test_02_ledger_records_every_exact_encoding(self) -> None:
        # Option B üç kademe: basit vaka Kademe 1'de biter (contour hiç denenmez).
        # Bu yüzden "her kodlamayı kaydeder" için tüm kademelere ulaşan node-yoğun
        # vaka kullanılır: polygon+rect byte-red (Kademe 1), contour geometri-red
        # (Kademe 2), quantized (Kademe 3). Exact aileler = {polygon, rect, contour}.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            _covering_parent(svg, _COMPLEX_SIDE)
            _report, ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(len(ledgers), 1)
            exact_families = {
                entry["encoding_family"]
                for entry in ledgers[0]
                if entry["exact_or_quantized"] == "exact"
            }
            self.assertEqual(exact_families, {"polygon", "rect", "contour"})

    def test_03_ledger_records_all_stroke_widths_when_byte_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            _covering_parent(svg, _COMPLEX_SIDE)
            _report, ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(len(ledgers), 1)
            polygon_strokes = {
                entry["stroke_width"]
                for entry in ledgers[0]
                if entry["encoding_family"] == "polygon"
            }
            # Byte-rejected encoding tüm stroke'ları dener (break yok) → 4 genişlik.
            self.assertEqual(polygon_strokes, {1.0, 1.5, 2.0, 3.0})

    def test_06_passing_exact_prevents_quantized_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            report, ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsNone(error)
            self.assertTrue(report["applied"])
            quantized = [
                entry for entry in ledgers[0]
                if entry["exact_or_quantized"] == "quantized"
            ]
            self.assertEqual(quantized, [])

    def test_07_smallest_passing_exact_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _simple_soft_disc_source(source)
            _covering_parent(svg, 96)
            report, ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsNone(error)
            passing = [
                entry for entry in ledgers[0]
                if entry["status"] == "accepted" and entry["journal_passed"]
            ]
            self.assertTrue(passing)
            smallest = min(passing, key=lambda e: e["actual_serialized_bytes"])
            self.assertEqual(report["painter_encoding_label"], smallest["encoding_label"])

    def test_08_all_exact_fail_triggers_quantized_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            _covering_parent(svg, _COMPLEX_SIDE)
            _report, ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsInstance(error, RuntimeError)
            families = {
                entry["exact_or_quantized"] for entry in ledgers[0]
            }
            # Exact hepsi başarısız → quantized aşaması denenmiş olmalı.
            self.assertIn("quantized", families)
            quantized_labels = {
                entry["encoding_label"] for entry in ledgers[0]
                if entry["exact_or_quantized"] == "quantized"
            }
            self.assertEqual(quantized_labels, {"contour-q128", "contour-q64", "contour-q32"})

    def test_16_no_admissible_reconstruction_rolls_back_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "s.png"
            svg = root / "c.svg"
            _complex_multi_glyph_source(source)
            original = _covering_parent(svg, _COMPLEX_SIDE)
            _report, _ledgers, error = _capture_attempts(
                lambda: apply_candidate_painter_reconstruction(svg, source, "logo_color")
            )
            self.assertIsInstance(error, RuntimeError)
            message = str(error)
            self.assertIn("no_admissible_reconstruction", message)
            self.assertIn("primary=contour", message)
            self.assertNotIn("primary=rect", message)
            # Fail-closed: kaynak SVG byte-birebir korunur.
            self.assertEqual(svg.read_bytes(), original)


class PainterEncodingEquivalenceTests(unittest.TestCase):
    """polygon (varsayılan) vs contour (grouped-evenodd) maske alfa düzlemi eşliği.

    Kanıtlanan (ve ölçülen) semantik gerçek: her iki kodlama da AYNI hücre-kenarı
    geometrisini üretir, bu yüzden ÖLÇÜM (native) ızgarasında piksel-BİREBİRdir
    (max_diff == 0) — seviye-0 kapalı göller, iç içe delikler, köşe-değen bileşenler
    ve çok-seviyeli paint sırası dahil. Bu, contour'un alfa kapılarını polygon ile
    aynı geçmesinin nedenidir.

    ANCAK native OLMAYAN (kesirli) ölçeklerde iki kodlama seviye-0/delik SINIRLARINDA
    ıraksar: polygon deliği gri-doldurup-siyah-üzerine-boyar, contour even-odd ile
    boş bırakır; ölçekli AA bu iki yolu farklı yuvarlar (ölçülen: ortalama < 1 gri
    seviye, < %2 piksel; tam-kat ölçeklerde 0). Bu ıraksama TAM OLARAK contour
    kabulünün byte/alfa ile DEĞİL, yukarı-ölçekte değerlendiren journal seam/geometri
    kapısıyla yönetilmesi gerektiğini gösterir (public-05: node/seam reddi).
    """

    def _render_pair(
        self, quantized: np.ndarray, opacity: dict[int, float], side: int, render_side: int
    ) -> tuple[np.ndarray, np.ndarray]:
        parent = _equivalence_parent(side)
        poly = _render_encoding_alpha(parent, quantized, opacity, "polygon", render_side)
        cont = _render_encoding_alpha(parent, quantized, opacity, "contour", render_side)
        return poly, cont

    def _assert_native_identical(
        self, quantized: np.ndarray, opacity: dict[int, float], side: int
    ) -> None:
        poly, cont = self._render_pair(quantized, opacity, side, side)
        max_diff = int(np.abs(poly - cont).max())
        self.assertEqual(
            max_diff, 0,
            f"native ({side}px) polygon vs contour alfa farkı = {max_diff} (0 bekleniyor)",
        )

    def test_11_level_zero_enclosed_island_native_identical(self) -> None:
        # Dış seviye-1 kare içinde kapalı seviye-0 (şeffaf) göl.
        side = 40
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[6:34, 6:34] = 1
        quantized[16:24, 16:24] = 0  # kapalı şeffaf ada
        self._assert_native_identical(quantized, {0: 0.0, 1: 1.0}, side)

    def test_12_nested_hole_with_island_native_identical(self) -> None:
        # seviye-1 gövde → seviye-0 delik → deliğin içinde seviye-1 ada (3 iç içe).
        side = 48
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[6:42, 6:42] = 1
        quantized[14:34, 14:34] = 0
        quantized[20:28, 20:28] = 1
        self._assert_native_identical(quantized, {0: 0.0, 1: 1.0}, side)

    def test_13_corner_touching_components_native_identical(self) -> None:
        # Köşeden değen iki seviye-1 kare (even-odd vs overpaint ayrımı sınanır).
        side = 40
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[6:20, 6:20] = 1
        quantized[20:34, 20:34] = 1
        self._assert_native_identical(quantized, {0: 0.0, 1: 1.0}, side)

    def test_14_multi_alpha_level_native_identical(self) -> None:
        # İç içe üç gri seviye (kısmi alfa gradyanı vekili) — paint sırası doğruluğu.
        side = 48
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[4:44, 4:44] = 1
        quantized[12:36, 12:36] = 2
        quantized[20:28, 20:28] = 3
        self._assert_native_identical(
            quantized, {0: 0.0, 1: 0.25, 2: 0.6, 3: 1.0}, side
        )

    def test_15_multi_scale_divergence_is_boundary_localized(self) -> None:
        # 192/256/512/native Resvg karşılaştırması: native TAM eşit; yukarı ölçekte
        # ıraksama YALNIZ ince sınır bandında (bütünde ortalama<2 gri, <%4 piksel).
        # Bu, contour'un seam/geometri kapısıyla yönetilmesini doğrular.
        side = 48
        quantized = np.zeros((side, side), dtype=np.int32)
        quantized[6:42, 6:42] = 1
        quantized[16:32, 16:32] = 0  # kesirli-ölçekte ıraksayan seviye-0 delik
        opacity = {0: 0.0, 1: 1.0}
        total = side * side
        for render_side in (side, 192, 256, 512):
            poly, cont = self._render_pair(quantized, opacity, side, render_side)
            diff = np.abs(poly - cont)
            differing_fraction = float((diff > 0).sum()) / float(diff.size)
            if render_side == side:
                self.assertEqual(
                    int(diff.max()), 0,
                    "native ölçekte kodlamalar TAM eşit olmalı",
                )
            else:
                # Bulk uyum: ıraksama sistemik dolgu farkı değil, sınır AA bandıdır.
                self.assertLess(
                    float(diff.mean()), 2.0,
                    f"@{render_side}px ortalama fark {diff.mean():.3f} — sınır-yerel değil",
                )
                self.assertLess(
                    differing_fraction, 0.04,
                    f"@{render_side}px farklı piksel oranı {differing_fraction:.4f} — çok geniş",
                )
        # Kaydedilen not: tam-kat olmayan ölçekte max fark > 0 olabilir; bu, journal
        # seam kapısının contour kabulünü neden nihai yetki olduğu gerçeğidir.
        _ = total


if __name__ == "__main__":
    unittest.main()
