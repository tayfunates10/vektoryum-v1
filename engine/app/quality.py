"""Kalite raporu katmanı.

``basic_svg_quality_check`` sade logolarda düşük path sayısını hata saymaz;
çok renkli logolarda detay/renk dengesini denetler ve gömülü bitmap'i ciddi
sorun olarak işaretler.
"""

from __future__ import annotations

from typing import Any

_FLAT_MODES = {"geometric_logo", "minimal_ai", "flat_logo", "single_color", "lineart", "centerline"}

# Bu algısal sadakatin altında çıktı "yaklaşık" sayılır (foto/sürekli-tonlu girdi).
# Gerçek logolar tipik olarak 85+ alır; survey'de zor girdiler 57-76 aldı.
_LOW_FIDELITY_THRESHOLD = 78.0


# Yapı bütünlüğü eşikleri: orijinaldeki konturların en az bu oranı render'da
# karşılanmalı (1px tolerans). Altına düşmek = kırık/eksik çizgi demektir.
_STRUCTURE_RECALL_WARN = 0.985
_STRUCTURE_RECALL_SEVERE = 0.955
_STRUCTURE_PRECISION_WARN = 0.96


def basic_svg_quality_check(
    score_details: dict[str, Any],
    mode: str,
    geometry_report: dict[str, float] | None = None,
    total_score: float = 0.0,
    fidelity_score: float | None = None,
    structure_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """En iyi adayın yapısal istatistiklerine göre kalite raporu üretir.

    ``fidelity_score`` (algısal sadakat) verilirse, yapısal sezgiler ona göre
    yumuşatılır: ör. düşük path sayısı, sadakat yüksekse "detay kaybı" sayılmaz
    (az path = daha düzenlenebilir, bir kusur değil).

    ``structure_report`` (bkz. ``fidelity.score_structure_integrity``) verilirse
    kırık/eksik çizgi ve hayalet çizik denetimi yapılır: kontur karşılama oranı
    eşiğin altındaysa çıktı ASLA ``production_ready`` işaretlenmez.
    """
    path_count = int(score_details.get("path_count", 0))
    unique_colors = int(score_details.get("unique_colors", 0))
    has_bitmap = bool(score_details.get("has_bitmap", False))
    node_count = int(score_details.get("node_count", 0))
    has_gradient = bool(score_details.get("has_gradient", False))

    warnings: list[str] = []

    if has_bitmap:
        warnings.append("SVG embeds a bitmap image; output is not fully vector.")

    if path_count == 0:
        warnings.append("No vector paths were produced.")

    flat_mode = mode in _FLAT_MODES
    # Sade logolarda az path uyarısı verme kuralı
    low_path_exempt = flat_mode and unique_colors <= 6 and path_count >= 10

    # yüksek sadakat varsa düşük path bir kusur değildir (sadık + düzenlenebilir)
    fidelity_ok = fidelity_score is not None and fidelity_score >= 85.0
    # yapı bütünlüğü ölçüldü ve konturlar tam karşılanıyorsa şekil eksik DEĞİLDİR;
    # "az path = eksik olabilir" sezgisi ölçüme karşı gelemez (tek-renk kesim /
    # stencil meşru olarak 1-2 path üretir)
    structure_ok = (
        structure_report is not None
        and float(structure_report.get("ink_recall", 0.0)) >= _STRUCTURE_RECALL_WARN
    )

    if mode == "logo_color":
        if 0 < path_count < 250 and not fidelity_ok:
            warnings.append("Low path count for a color logo; some detail may be lost.")
        if unique_colors > 64 and not has_gradient:
            warnings.append("High color count; consider reducing palette for production.")
    elif not low_path_exempt:
        if 0 < path_count < 4 and not fidelity_ok and not structure_ok:
            warnings.append("Very low path count; the shape may be incomplete.")

    if path_count > 3000:
        warnings.append("Very high path count; the file may be heavy and hard to edit.")

    # genel renk uyarısı (sade modlar için)
    if flat_mode and unique_colors > 8:
        warnings.append("More colors than expected for a flat logo; palette cleanup recommended.")

    # Algısal sadakat düşükse: görsel büyük olasılıkla sürekli-tonlu/fotografik.
    # Gerçek logolar 85-98 sadakat alır; bu eşik yalnızca vektörleştirmenin doğal
    # tavanındaki zor girdileri (foto/karmaşık illüstrasyon) işaretler — renkli
    # logoları etkilemez. Foto↔logo'yu önden sınıflandırmaktan (kırılgan) daha
    # güvenilir bir sinyal: ölçülen gerçek sadakat.
    low_fidelity = fidelity_score is not None and fidelity_score < _LOW_FIDELITY_THRESHOLD
    if low_fidelity:
        warnings.append(
            "Low perceptual fidelity; the image looks photographic or continuous-tone. "
            "The vector output is an approximation — a cleaner logo or higher-quality "
            "source image will give a better result."
        )

    # Yapı bütünlüğü: kırık/eksik çizgi ve hayalet çizik denetimi
    structure_broken = False
    structure_block = None
    if structure_report:
        recall = float(structure_report.get("ink_recall", 1.0))
        precision = float(structure_report.get("ink_precision", 1.0))
        comp_delta = int(structure_report.get("component_delta", 0))
        comps_orig = int(structure_report.get("components_original", 0))
        if recall < _STRUCTURE_RECALL_WARN:
            warnings.append(
                "Some strokes or shapes from the original are missing or broken "
                "in the vector output."
            )
            if recall < _STRUCTURE_RECALL_SEVERE:
                structure_broken = True
        if precision < _STRUCTURE_PRECISION_WARN:
            warnings.append("The vector output contains stray marks not present in the original.")
            structure_broken = True
        # parçalanma: bileşen sayısı belirgin arttıysa şekiller bölünmüş demektir
        if comps_orig > 0 and comp_delta > max(2, int(0.5 * comps_orig)):
            warnings.append("Shapes appear fragmented into more pieces than the original.")
            structure_broken = True
        structure_block = structure_report

    geo = geometry_report or {}
    geometry_block = {
        "straight_edge_score": round(float(geo.get("straight_edge_score", 0.0)) * 100, 2),
        "corner_cleanliness_score": round(float(geo.get("corner_cleanliness_score", 0.0)) * 100, 2),
        "axis_alignment_score": round(float(geo.get("axis_alignment_score", 0.0)) * 100, 2),
        "geometry_score": round(float(geo.get("geometry_score", 0.0)) * 100, 2),
    }

    # durum kararı
    serious = has_bitmap or path_count == 0
    if serious:
        status = "failed" if path_count == 0 else "needs_review"
    elif structure_broken:
        # kırık çizgi / hayalet çizik varken çıktı üretime hazır sayılamaz
        status = "needs_review"
    elif low_fidelity:
        # ölçülen sadakat düşükse "üretime hazır" diyemeyiz (dürüst beklenti)
        status = "needs_review"
    elif not warnings and (
        total_score >= 80.0
        or (fidelity_score is not None and fidelity_score >= 90.0)
    ):
        # total_score yapısal bir SEZGİDİR; ölçülen algısal sadakat asıl ürün
        # metriğidir. Uyarısız + yapısı sağlam + sadakati >= 90 çıktı (ör.
        # 1-path gradyan: yapısal skoru düşük ama render birebir) üretime
        # hazırdır — aksi kullanıcıya yanlış "gözden geçirin" sinyali veriyordu.
        status = "production_ready"
    elif total_score >= 70.0 and len(warnings) <= 1:
        status = "production_ready" if not warnings else "needs_review"
    else:
        status = "needs_review"

    return {
        "status": status,
        "warnings": warnings,
        "metrics": {
            "path_count": path_count,
            "node_count": node_count,
            "unique_color_count": unique_colors,
            "has_bitmap": has_bitmap,
            "fidelity_score": fidelity_score,
        },
        "geometry_report": geometry_block,
        "structure_report": structure_block,
    }
