"""Artefakt regresyon testi: kırık çizgi / çizik / kenarlık / renk hatası YOK.

``regression/artifact_probe.py`` vakalarını uçtan uca çalıştırır ve çıktının
"net" olduğunu SAYISAL kabul kriterleriyle kilitler:

* ink_recall >= 0.995      — orijinaldeki hiçbir çizgi/şekil kopmaz, silinmez
* ink_precision >= 0.975   — çıktıda hayalet çizik/leke yoktur
* component_delta == 0     — şekiller parçalanmaz ve birbirine yapışmaz
* seam_ratio <= 0.002      — bitişik renkler arasında zemin sızması (çizik) yok
* halo_ratio <= 0.02       — beklenen palet dışında renk bandı/kirlilik yok
* fidelity >= vaka tabanı  — genel algısal sadakat gerilemez

Ek olarak kalite kapısının (quality gate) kırık çıktıyı asla
``production_ready`` işaretlemediği birim düzeyinde doğrulanır.

Çalıştırma::

    .venv/bin/python test_artifact_quality.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(ENGINE_DIR / "regression"))

from artifact_probe import CASES, run_case  # noqa: E402

from app.quality import basic_svg_quality_check  # noqa: E402

# Vaka bazlı asgari sadakat (ölçülen değerlerin ~1.5 puan altı: gürültü payı)
MIN_FIDELITY = {
    "thin_lines": 90.5,
    "border_frames": 89.5,
    "adjacent_colors": 94.5,
    "curves_smooth": 96.0,
    "small_glyphs": 97.5,
}

# Renk koruma: bu vakalar İKİLİ (siyah-beyaz) modlara düşmemeli
EXPECTED_MODES = {
    "thin_lines": {"geometric_logo", "minimal_ai", "logo_color"},
    "border_frames": {"geometric_logo", "minimal_ai"},
}

GLOBAL_LIMITS = {
    "ink_recall_min": 0.995,
    "ink_precision_min": 0.975,
    "component_delta": 0,
    "seam_ratio_max": 0.002,
    "halo_ratio_max": 0.02,
}


def _check(results: list[tuple[str, bool, str]], name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"   [{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


def run_probe_cases(results: list[tuple[str, bool, str]]) -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        for case in CASES:
            r = run_case(case, out_dir)
            if not r.get("ok"):
                _check(results, f"{case}: pipeline üretimi", False, str(r.get("error")))
                continue
            print(f"=== {case} (mode={r['mode_used']}, best={r['best']}) "
                  f"fid={r['fidelity']:.1f} recall={r['ink_recall']:.4f} "
                  f"prec={r['ink_precision']:.4f} dComp={r['component_delta']:+d} "
                  f"seam={r['seam_ratio']:.5f} halo={r['halo_ratio']}")
            _check(results, f"{case}: kırık/eksik çizgi yok (ink_recall)",
                   r["ink_recall"] >= GLOBAL_LIMITS["ink_recall_min"],
                   f"{r['ink_recall']:.4f} >= {GLOBAL_LIMITS['ink_recall_min']}")
            _check(results, f"{case}: hayalet çizik yok (ink_precision)",
                   r["ink_precision"] >= GLOBAL_LIMITS["ink_precision_min"],
                   f"{r['ink_precision']:.4f} >= {GLOBAL_LIMITS['ink_precision_min']}")
            _check(results, f"{case}: şekil parçalanması yok (component_delta)",
                   r["component_delta"] == GLOBAL_LIMITS["component_delta"],
                   f"{r['component_delta']:+d} == 0")
            _check(results, f"{case}: bitişik renkte zemin sızması yok (seam)",
                   r["seam_ratio"] <= GLOBAL_LIMITS["seam_ratio_max"],
                   f"{r['seam_ratio']:.5f} <= {GLOBAL_LIMITS['seam_ratio_max']}")
            if r.get("halo_ratio") is not None:
                _check(results, f"{case}: palet dışı renk bandı yok (halo)",
                       r["halo_ratio"] <= GLOBAL_LIMITS["halo_ratio_max"],
                       f"{r['halo_ratio']:.5f} <= {GLOBAL_LIMITS['halo_ratio_max']}")
            _check(results, f"{case}: sadakat tabanı",
                   r["fidelity"] >= MIN_FIDELITY[case],
                   f"{r['fidelity']:.1f} >= {MIN_FIDELITY[case]}")
            if case in EXPECTED_MODES:
                _check(results, f"{case}: renk koruyan mod",
                       r["mode_used"] in EXPECTED_MODES[case],
                       f"{r['mode_used']} in {sorted(EXPECTED_MODES[case])}")


def run_quality_gate_checks(results: list[tuple[str, bool, str]]) -> None:
    """Kalite kapısı: kırık yapı raporu ASLA production_ready olamaz."""
    base = dict(path_count=40, node_count=400, unique_colors=4, has_bitmap=False)

    broken = basic_svg_quality_check(
        score_details=base, mode="geometric_logo", total_score=92.0, fidelity_score=93.0,
        structure_report={"ink_recall": 0.90, "ink_precision": 0.99,
                          "components_original": 8, "components_rendered": 5, "component_delta": -3},
    )
    _check(results, "quality gate: kopuk çizgi -> needs_review",
           broken["status"] == "needs_review" and any("missing or broken" in w for w in broken["warnings"]),
           f"status={broken['status']}")

    ghost = basic_svg_quality_check(
        score_details=base, mode="geometric_logo", total_score=92.0, fidelity_score=93.0,
        structure_report={"ink_recall": 0.999, "ink_precision": 0.90,
                          "components_original": 8, "components_rendered": 9, "component_delta": 1},
    )
    _check(results, "quality gate: hayalet çizik -> needs_review",
           ghost["status"] == "needs_review",
           f"status={ghost['status']}")

    fragmented = basic_svg_quality_check(
        score_details=base, mode="geometric_logo", total_score=92.0, fidelity_score=93.0,
        structure_report={"ink_recall": 0.999, "ink_precision": 0.999,
                          "components_original": 6, "components_rendered": 14, "component_delta": 8},
    )
    _check(results, "quality gate: parçalanma -> needs_review",
           fragmented["status"] == "needs_review",
           f"status={fragmented['status']}")

    clean = basic_svg_quality_check(
        score_details=base, mode="geometric_logo", total_score=92.0, fidelity_score=93.0,
        structure_report={"ink_recall": 0.9995, "ink_precision": 0.999,
                          "components_original": 8, "components_rendered": 8, "component_delta": 0},
    )
    _check(results, "quality gate: sağlam yapı -> production_ready",
           clean["status"] == "production_ready" and not clean["warnings"],
           f"status={clean['status']} warnings={clean['warnings']}")


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    run_quality_gate_checks(results)
    run_probe_cases(results)

    passed = sum(1 for _, ok, _ in results if ok)
    print("=" * 60)
    print(f"SONUC: {passed}/{len(results)} kontrol gecti")
    print("=" * 60)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
