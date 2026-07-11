"""Export katmanı: SVG temizleme + PDF / EPS / DXF üretimi.

Dayanıklılık ilkeleri:
* SVG her zaman üretilir (gömülü bitmap'ler temizlenir).
* PDF: CairoSVG (Cairo DLL varsa) -> Inkscape -> svglib+reportlab -> hata.
* EPS: CairoSVG (svg2ps) -> Inkscape -> svglib+reportlab -> hata.
* DXF: svgpathtools + ezdxf (saf Python, Windows'ta güvenilir).
* Hangi format başarısız olursa olsun diğerleri ve API çalışmaya devam eder;
  hata mesajları ``output_errors`` içinde döner.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"


# ---------------------------------------------------------------------------
# Inkscape tespiti
# ---------------------------------------------------------------------------
def get_inkscape_path() -> str | None:
    env = os.environ.get("INKSCAPE_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("inkscape")
    if found:
        return found
    # Windows tipik kurulum yolları
    for candidate in (
        r"C:\Program Files\Inkscape\bin\inkscape.exe",
        r"C:\Program Files\Inkscape\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# SVG temizleme / export
# ---------------------------------------------------------------------------
def clean_svg(src: Path, dst: Path, candidate_id: str | None = None) -> Path:
    """SVG'yi temizleyip hedefe yazar: bitmap image tag'lerini ve boş path'leri siler."""
    try:
        ET.register_namespace("", SVG_NS)
        tree = ET.parse(str(src))
        root = tree.getroot()

        # gömülü bitmap'leri kaldır
        for parent in root.iter():
            to_remove = [c for c in list(parent) if c.tag.split("}")[-1] == "image"]
            for child in to_remove:
                parent.remove(child)

        if candidate_id:
            root.set("data-candidate", candidate_id)

        # Bileşik (çok alt-yollu) path'lerde fill-rule AÇIK yazılır: delikler
        # sarım yönüyle kurulur (nonzero varsayılanı doğru render eder) ama
        # örtük bırakmak editör/araç uyumluluğunda belirsizlik yaratır.
        # "nonzero"yu açıkça yazmak görüntüyü değiştirmez (varsayılanla aynı),
        # sözleşmeyi netleştirir; evenodd YAZILMAZ — aynı yönlü örtüşen
        # alt-yolları olan birleşimlerde deliğe dönüşürdü.
        import re as _re  # noqa: PLC0415

        for el in root.iter():
            if el.tag.split("}")[-1] != "path" or el.get("fill-rule"):
                continue
            d = el.get("d") or ""
            if len(_re.findall(r"(?<![0-9a-zA-Z.,-])[Mm]", " " + d)) >= 2:
                el.set("fill-rule", "nonzero")

        tree.write(str(dst), encoding="utf-8", xml_declaration=True)
        return dst
    except Exception as e:  # noqa: BLE001
        logger.warning("SVG temizlenemedi (%s), ham kopya kullanılıyor.", e)
        shutil.copyfile(src, dst)
        return dst


def export_svg(src: Path, dst: Path, candidate_id: str | None = None) -> Path:
    return clean_svg(Path(src), Path(dst), candidate_id)


# ---------------------------------------------------------------------------
# Render fallback yardımcıları
# ---------------------------------------------------------------------------
def _cairosvg_convert(src: Path, dst: Path, fmt: str) -> bool:
    """CairoSVG ile PDF/PS üretmeyi dener. Cairo DLL yoksa False döner.

    Not: Windows'ta cairo DLL eksikse ``import cairosvg`` *import anında* OSError
    fırlatabilir (cairocffi DLL'i import sırasında yükler). Bu yüzden import
    geniş ``Exception`` ile sarmalanır.
    """
    try:
        import cairosvg  # noqa: PLC0415
    except Exception:  # noqa: BLE001  (ImportError veya cairo DLL OSError)
        return False
    try:
        if fmt == "pdf":
            cairosvg.svg2pdf(url=str(src), write_to=str(dst))
        elif fmt == "eps":
            # CairoSVG PostScript üretir; .eps uzantısıyla yaz
            cairosvg.svg2ps(url=str(src), write_to=str(dst))
        else:
            return False
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception as e:  # noqa: BLE001  (Cairo DLL eksikse OSError)
        logger.info("CairoSVG %s render başarısız: %s", fmt, e)
        return False


def _inkscape_convert(src: Path, dst: Path, fmt: str) -> bool:
    inkscape = get_inkscape_path()
    if not inkscape:
        return False
    export_type = "pdf" if fmt == "pdf" else "eps"
    try:
        subprocess.run(
            [inkscape, str(src), f"--export-type={export_type}", f"--export-filename={dst}"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        logger.info("Inkscape %s render başarısız: %s", fmt, e)
        return False


def _svglib_convert(src: Path, dst: Path, fmt: str) -> bool:
    """svglib + reportlab ile saf Python PDF/EPS fallback'i (kuruluysa)."""
    try:
        from svglib.svglib import svg2rlg  # noqa: PLC0415
        from reportlab.graphics import renderPDF, renderPS  # noqa: PLC0415
    except ImportError:
        return False
    try:
        drawing = svg2rlg(str(src))
        if drawing is None:
            return False
        if fmt == "pdf":
            renderPDF.drawToFile(drawing, str(dst))
        elif fmt == "eps":
            renderPS.drawToFile(drawing, str(dst))
        else:
            return False
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        logger.info("svglib %s render başarısız: %s", fmt, e)
        return False


def export_pdf(src: Path, dst: Path) -> Path:
    for converter in (_cairosvg_convert, _inkscape_convert, _svglib_convert):
        if converter(Path(src), Path(dst), "pdf"):
            logger.info("PDF üretildi: %s (%s)", dst, converter.__name__)
            return Path(dst)
    raise RuntimeError("PDF render edilemedi (CairoSVG/Inkscape/svglib yok veya başarısız).")


def export_eps(src: Path, dst: Path) -> Path:
    for converter in (_cairosvg_convert, _inkscape_convert, _svglib_convert):
        if converter(Path(src), Path(dst), "eps"):
            logger.info("EPS üretildi: %s (%s)", dst, converter.__name__)
            return Path(dst)
    raise RuntimeError("EPS render edilemedi (CairoSVG/Inkscape/svglib yok veya başarısız).")


# ---------------------------------------------------------------------------
# DXF export (svgpathtools + ezdxf)
# ---------------------------------------------------------------------------
def _svg_height(src: Path) -> float:
    try:
        root = ET.parse(str(src)).getroot()
        vb = root.get("viewBox")
        if vb:
            parts = [float(x) for x in vb.replace(",", " ").split()]
            if len(parts) == 4:
                return parts[3]
        h = root.get("height")
        if h:
            return float("".join(ch for ch in h if (ch.isdigit() or ch == ".")))
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _svg_unit_scale(src: Path) -> float:
    """viewBox koordinatlarını width/height iç boyutuna taşıyan ölçek.

    Ön işleme izleme rasterini ölçekleyebilir (süperörnekleme/küçültme);
    pipeline width/height'ı kaynak boyuta çeker ama path koordinatları viewBox
    uzayında kalır. DXF fiziksel birim taşıdığından koordinatlar bu ölçekle
    kaynak piksel birimine indirgenir (500x300 girdinin 1000x600 birimlik DXF
    vermesi gerçek bir hataydı — PR incelemesi).
    """
    try:
        root = ET.parse(str(src)).getroot()
        vb = root.get("viewBox")
        w_attr = root.get("width")
        if not vb or not w_attr:
            return 1.0
        parts = [float(x) for x in vb.replace(",", " ").split()]
        w_px = float(str(w_attr).rstrip("px"))
        if len(parts) == 4 and parts[2] > 0 and w_px > 0:
            return w_px / parts[2]
    except Exception:  # noqa: BLE001
        pass
    return 1.0


def _is_finite_point(x: float, y: float) -> bool:
    return all(math.isfinite(v) for v in (x, y))


import re as _re

_XF_RE = _re.compile(r"(\w+)\s*\(([^)]*)\)")
_XF_NUM = _re.compile(r"[-+0-9.eE]+")


def _parse_transform(s: str | None) -> tuple[float, float, float, float, float, float]:
    """SVG transform string'ini affine matrise (a,b,c,d,e,f) çevirir.

    VTracer ``translate(x,y)`` kullanır; matrix/scale de desteklenir. Çoklu
    transform soldan sağa çarpılarak birleştirilir.
    """
    a, b, c, d, e, f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    if not s:
        return (a, b, c, d, e, f)
    for name, args in _XF_RE.findall(s):
        n = [float(x) for x in _XF_NUM.findall(args)]
        if name == "translate" and n:
            ta, tb, tc, td, te, tf = 1.0, 0.0, 0.0, 1.0, n[0], (n[1] if len(n) > 1 else 0.0)
        elif name == "scale" and n:
            sx = n[0]
            sy = n[1] if len(n) > 1 else n[0]
            ta, tb, tc, td, te, tf = sx, 0.0, 0.0, sy, 0.0, 0.0
        elif name == "matrix" and len(n) == 6:
            ta, tb, tc, td, te, tf = n
        else:
            continue
        # (a b c d e f) o (ta tb tc td te tf)  -> mevcut * yeni
        a, b, c, d, e, f = (
            a * ta + c * tb,
            b * ta + d * tb,
            a * tc + c * td,
            b * tc + d * td,
            a * te + c * tf + e,
            b * te + d * tf + f,
        )
    return (a, b, c, d, e, f)


def export_dxf(src: Path, dst: Path) -> Path:
    """SVG path'lerini DXF LWPOLYLINE'lara çevirir.

    * Eğriler nokta örneklemesiyle düzleştirilir.
    * Geometrik logolarda zaten düz çizgiler korunur.
    * Ardışık çakışan noktalar silinir, nan/inf noktalar atlanır.
    * Kapalı path'ler close=True olur.
    * Y ekseni CAD için ters çevrilir (SVG y-aşağı -> DXF y-yukarı).
    """
    try:
        import ezdxf  # noqa: PLC0415
        from svgpathtools import svg2paths2  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(f"DXF için gerekli kütüphane eksik: {e}") from e

    src = Path(src)
    unit = _svg_unit_scale(src)          # viewBox -> kaynak piksel birimi
    height = _svg_height(src) * unit

    try:
        paths, attributes, _svg_attr = svg2paths2(str(src))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"SVG path parse edilemedi: {e}") from e

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    polyline_count = 0

    for path, attr in zip(paths, attributes):
        if len(path) == 0:
            continue
        try:
            closed = path.isclosed()
        except Exception:  # noqa: BLE001
            closed = False

        # path'in transform'unu uygula (VTracer translate kullanır)
        xf = _parse_transform(attr.get("transform"))

        points: list[tuple[float, float]] = []
        for seg in path:
            seg_type = seg.__class__.__name__
            if seg_type == "Line":
                samples = [0.0, 1.0]
            else:
                # eğri: uzunluğa göre örnekle
                try:
                    seg_len = seg.length()
                except Exception:  # noqa: BLE001
                    seg_len = 10.0
                n = max(4, min(48, int(seg_len / 3.0)))
                samples = [i / n for i in range(n + 1)]

            for t in samples:
                try:
                    pt = seg.point(t)
                except Exception:  # noqa: BLE001
                    continue
                # yerel koordinatı transform ile kullanıcı uzayına taşı,
                # sonra kaynak piksel birimine ölçekle
                ux = (xf[0] * pt.real + xf[2] * pt.imag + xf[4]) * unit
                uy = (xf[1] * pt.real + xf[3] * pt.imag + xf[5]) * unit
                x = float(ux)
                y = float(height - uy) if height > 0 else float(-uy)
                if not _is_finite_point(x, y):
                    continue
                if points and abs(points[-1][0] - x) < 1e-6 and abs(points[-1][1] - y) < 1e-6:
                    continue
                points.append((x, y))

        if len(points) < 2:
            continue
        # kapalıysa baş==son tekrarını at
        if closed and len(points) > 2 and abs(points[0][0] - points[-1][0]) < 1e-6 and abs(points[0][1] - points[-1][1]) < 1e-6:
            points.pop()

        msp.add_lwpolyline(points, close=closed)
        polyline_count += 1

    if polyline_count == 0:
        raise RuntimeError("DXF'e yazılacak geçerli geometri bulunamadı.")

    doc.saveas(str(dst))
    logger.info("DXF üretildi: %s (%d polyline).", dst, polyline_count)
    return Path(dst)


# ---------------------------------------------------------------------------
# PNG export ("temizlenmiş" raster çıktı)
# ---------------------------------------------------------------------------
def export_png(src: Path, dst: Path, width: int | None = None, height: int | None = None) -> Path:
    """SVG'yi 'temizlenmiş' PNG olarak render eder (resvg -> fallback zinciri).

    Boyut verilmezse SVG'nin kendi boyutu kullanılır; en uzun kenar 4096 ile
    sınırlanır. Render backend'i yoksa RuntimeError (output_errors'a düşer).
    """
    from app.fidelity import render_svg_to_rgb  # noqa: PLC0415 (döngüsel import önlemi)
    from PIL import Image  # noqa: PLC0415

    src = Path(src)
    if not width or not height:
        try:
            root = ET.parse(str(src)).getroot()
            vb = root.get("viewBox")
            if vb:
                parts = [float(x) for x in vb.replace(",", " ").split()]
                width, height = int(parts[2]), int(parts[3])
            else:
                width = int(float("".join(ch for ch in (root.get("width") or "1024") if ch.isdigit() or ch == ".")))
                height = int(float("".join(ch for ch in (root.get("height") or "1024") if ch.isdigit() or ch == ".")))
        except Exception:  # noqa: BLE001
            width, height = 1024, 1024
    longest = max(width, height)
    if longest > 4096:
        scale = 4096.0 / longest
        width, height = max(1, int(width * scale)), max(1, int(height * scale))

    rgb = render_svg_to_rgb(src, int(width), int(height))
    if rgb is None:
        raise RuntimeError("PNG render edilemedi (render backend yok: resvg/pymupdf/cairosvg/svglib).")
    Image.fromarray(rgb).save(str(dst))
    logger.info("PNG üretildi: %s (%dx%d).", dst, width, height)
    return Path(dst)


# ---------------------------------------------------------------------------
# Toplu export
# ---------------------------------------------------------------------------
def export_all(
    best_svg: Path,
    job_dir: Path,
    job_id: str,
    candidate_id: str | None = None,
    formats: tuple[str, ...] = ("svg", "pdf", "eps", "dxf", "png"),
    png_size: tuple[int, int] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Tüm formatları üretir. ``(outputs, errors)`` döndürür.

    outputs: {format: dosya_yolu}, errors: {format: hata_mesajı}
    """
    job_dir = Path(job_dir)
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}

    # SVG önce üretilmeli (diğer formatların kaynağı)
    svg_dst = job_dir / f"{job_id}.svg"
    try:
        export_svg(Path(best_svg), svg_dst, candidate_id)
        outputs["svg"] = str(svg_dst)
    except Exception as e:  # noqa: BLE001
        errors["svg"] = str(e)
        shutil.copyfile(best_svg, svg_dst)
        outputs["svg"] = str(svg_dst)

    source_svg = Path(outputs["svg"])
    exporters = {
        "pdf": export_pdf,
        "eps": export_eps,
        "dxf": export_dxf,
    }
    for fmt in formats:
        if fmt == "svg":
            continue
        dst = job_dir / f"{job_id}.{fmt}"
        try:
            if fmt == "png":
                w, h = png_size if png_size else (None, None)
                export_png(source_svg, dst, width=w, height=h)
                outputs["png"] = str(dst)
                continue
            func = exporters.get(fmt)
            if not func:
                continue
            func(source_svg, dst)
            outputs[fmt] = str(dst)
        except Exception as e:  # noqa: BLE001
            errors[fmt] = str(e)
            logger.warning("%s export hatası: %s", fmt.upper(), e)

    return outputs, errors
