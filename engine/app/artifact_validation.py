"""Export artifact doğrulama ve ham-byte kimlik zinciri.

Bir dosyanın yalnız var/boş-değil olması onu indirilebilir yapmaz. Bu modül her
formatı job dizini sınırı, parse edilebilirlik ve ham SHA-256 üzerinden doğrular.
Görsel kalite verdict'i ayrı kalır; burada yalnız yapısal güvenlik ölçülür.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


def _result(fmt: str, path: Path | None = None) -> dict[str, Any]:
    return {
        "format": fmt,
        "exists": False,
        "byte_size": 0,
        "sha256": None,
        "structural_safe": False,
        "validation_level": "format_parse",
        "validation_codes": [],
        "validation_messages": [],
        "filename": path.name if path is not None else None,
    }


def _fail(rep: dict[str, Any], code: str, message: str) -> None:
    if code not in rep["validation_codes"]:
        rep["validation_codes"].append(code)
        rep["validation_messages"].append(message)


def resolve_job_artifact(path_value: Any, job_dir: Path) -> Path | None:
    """Output path'ini canonical job dizini içine çözer; dışarı taşmayı reddeder."""
    if path_value is None:
        return None
    base = Path(job_dir).resolve()
    raw = Path(path_value)
    candidate = (raw if raw.is_absolute() else base / raw).resolve()
    if candidate != base and base not in candidate.parents:
        return None
    return candidate


def _validate_png(data: bytes, rep: dict[str, Any], expected_size: tuple[int, int] | None) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            size = tuple(map(int, im.size))
            rep["dimensions"] = list(size)
            if im.format != "PNG":
                _fail(rep, "png_magic_mismatch", "Dosya gerçek PNG değil")
            if expected_size and size != tuple(expected_size):
                _fail(rep, "png_dimension_mismatch", "PNG beklenen boyutta değil")
    except Exception as e:  # noqa: BLE001
        _fail(rep, "png_parse_failed", f"PNG parse başarısız: {e}")


def _validate_pdf(data: bytes, rep: dict[str, Any]) -> None:
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            rep["page_count"] = int(doc.page_count)
            if doc.page_count < 1:
                _fail(rep, "pdf_no_pages", "PDF sayfa içermiyor")
            if getattr(doc, "embfile_names", lambda: [])():
                _fail(rep, "pdf_embedded_file", "PDF gömülü dosya içeriyor")
        finally:
            doc.close()
        # Exporter çıktısında aktif eylem beklenmez. Parser açsa dahi script,
        # launch veya otomatik action taşıyan belge indirmeye açılmaz.
        if re.search(
            rb"/(?:JavaScript|JS|OpenAction|AA|Launch|EmbeddedFile)\b", data,
        ):
            _fail(rep, "pdf_active_content", "PDF aktif/gömülü içerik taşıyor")
    except Exception as e:  # noqa: BLE001
        _fail(rep, "pdf_parse_failed", f"PDF parse başarısız: {e}")


def _validate_eps(data: bytes, rep: dict[str, Any]) -> None:
    # EPS, PostScript'tir; üretim exporter'ımız için header + finite BoundingBox
    # ve aktif binary/include bulunmaması güvenli yapısal minimumdur.
    try:
        text = data.decode("latin-1")
    except Exception as e:  # pragma: no cover - latin-1 her byte'ı çözer
        _fail(rep, "eps_decode_failed", f"EPS çözülemedi: {e}")
        return
    if not text.startswith("%!PS-Adobe"):
        _fail(rep, "eps_header_invalid", "EPS PostScript header geçersiz")
    match = re.search(
        r"^%%BoundingBox:\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$",
        text, flags=re.MULTILINE,
    )
    if not match:
        _fail(rep, "eps_bounding_box_missing", "EPS BoundingBox yok/geçersiz")
    elif not (float(match.group(3)) > float(match.group(1))
              and float(match.group(4)) > float(match.group(2))):
        _fail(rep, "eps_bounding_box_invalid", "EPS BoundingBox boyutu geçersiz")
    low = text.lower()
    if "%%beginbinary" in low or "%%begindata" in low:
        _fail(rep, "eps_unsafe_payload", "EPS beklenmeyen aktif/binary içerik taşıyor")
    gs = shutil.which("gs")
    if gs is None:
        _fail(rep, "eps_validator_unavailable", "EPS parser/renderer kullanılamıyor")
        return
    try:
        completed = subprocess.run(
            [gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=nullpage", "-"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=12,
        )
        rep["ghostscript_exit_code"] = int(completed.returncode)
        if completed.returncode != 0:
            _fail(rep, "eps_render_failed", "EPS güvenli renderer ile açılamadı")
    except subprocess.TimeoutExpired:
        _fail(rep, "eps_validation_timeout", "EPS doğrulama zaman aşımına uğradı")
    except Exception as e:  # noqa: BLE001
        _fail(rep, "eps_validation_failed", f"EPS doğrulanamadı: {type(e).__name__}")


def _finite_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if hasattr(value, "x") and hasattr(value, "y"):
        coords = [value.x, value.y]
        if hasattr(value, "z"):
            coords.append(value.z)
        return all(_finite_value(coord) for coord in coords)
    if isinstance(value, (list, tuple)):
        return all(_finite_value(item) for item in value)
    return True


def _validate_dxf(data: bytes, rep: dict[str, Any]) -> None:
    try:
        import ezdxf  # noqa: PLC0415

        text = data.decode("utf-8-sig")
        doc = ezdxf.read(io.StringIO(text))
        auditor = doc.audit()
        if auditor.has_errors:
            _fail(rep, "dxf_audit_failed", "DXF audit yapısal hata buldu")
        entities = list(doc.modelspace())
        rep["entity_count"] = len(entities)
        if not entities:
            _fail(rep, "dxf_no_entities", "DXF modelspace boş")
        forbidden = {"IMAGE", "WIPEOUT", "PDFUNDERLAY", "DGNUNDERLAY", "DWFUNDERLAY", "OLE2FRAME"}
        for entity in entities:
            if entity.dxftype() in forbidden:
                _fail(rep, "dxf_external_or_raster", "DXF dış/raster içerik taşıyor")
            if not all(_finite_value(value) for value in entity.dxfattribs().values()):
                _fail(rep, "dxf_nonfinite_geometry", "DXF non-finite geometri içeriyor")
    except Exception as e:  # noqa: BLE001
        _fail(rep, "dxf_parse_failed", f"DXF parse başarısız: {e}")


def validate_export_artifacts(
    outputs: dict[str, Any],
    job_dir: Path,
    *,
    svg_structure: dict[str, Any] | None = None,
    svg_structural_codes: list[str] | None = None,
    expected_png_size: tuple[int, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bütün exportları doğrular ve format -> immutable rapor döndürür."""
    artifacts: dict[str, dict[str, Any]] = {}
    for fmt in ("svg", "pdf", "eps", "dxf", "png"):
        path = resolve_job_artifact(outputs.get(fmt), job_dir)
        rep = _result(fmt, path)
        artifacts[fmt] = rep
        if path is None:
            _fail(rep, "path_outside_job_or_missing", "Artifact path'i job dizini dışında veya yok")
            continue
        try:
            if not path.is_file():
                _fail(rep, "artifact_missing", "Artifact dosyası yok")
                continue
            data = path.read_bytes()
        except Exception as e:  # noqa: BLE001
            _fail(rep, "artifact_read_failed", f"Artifact okunamadı: {e}")
            continue
        rep["exists"] = True
        rep["byte_size"] = len(data)
        rep["sha256"] = hashlib.sha256(data).hexdigest()
        if not data:
            _fail(rep, "artifact_empty", "Artifact boş")
            continue

        if fmt == "svg":
            if svg_structure is None:
                from app.final_artifact_evaluator import _structure_check  # noqa: PLC0415

                struct, _messages, parsed_codes, _root = _structure_check(data)
                struct["structural_safe"] = not parsed_codes
                codes = parsed_codes
            else:
                struct = svg_structure
                codes = list(svg_structural_codes or [])
            rep["exact_metrics"] = {
                key: struct.get(key) for key in (
                    "path_count", "node_count", "linear_gradient_count",
                    "radial_gradient_count", "mesh_gradient_count",
                    "gradient_definition_count", "gradient_reference_count",
                )
            }
            for code in codes:
                _fail(rep, code, f"SVG yapısal doğrulama hatası: {code}")
            if not struct.get("structural_safe"):
                _fail(rep, "svg_not_structurally_safe", "SVG yapısal olarak güvenli değil")
        elif fmt == "png":
            _validate_png(data, rep, expected_png_size)
        elif fmt == "pdf":
            _validate_pdf(data, rep)
        elif fmt == "eps":
            _validate_eps(data, rep)
        elif fmt == "dxf":
            _validate_dxf(data, rep)

        rep["structural_safe"] = not rep["validation_codes"]
    return artifacts


def downloadable_formats(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(fmt for fmt, rep in artifacts.items() if rep.get("structural_safe"))
