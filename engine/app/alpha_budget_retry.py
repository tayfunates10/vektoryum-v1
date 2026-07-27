"""Knockout adayının değerlendirme bütçesi için ölçüm-kapılı kompakt retry.

Normal exact/128-seviye üretim yolu otorite olarak kalır. Bu modül yalnız bu aday
başarıyla üretilip ``TransformJournal`` tarafından SADECE değerlendirme bütçesi
nedeniyle geri alındığında devreye girer. Kompakt adaylar kabul edilmiş parent'tan
yeniden kurulur; mevcut byte, artwork-kimliği, direct alpha ve
``FinalArtifactEvaluator`` kapıları değişmeden uygulanır. Her aday, production'daki
painter retry ile aynı şekilde taze fakat AYNI süre/karmaşıklık sınırlarına sahip bir
journal'da ölçülür. Eşik, tolerans veya ortam bütçesi artırılmaz.
"""
from __future__ import annotations

import copy
import hashlib
import os
import xml.etree.ElementTree as ET
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_COMPACT_ALPHA_LEVELS = (32, 16, 8)
_INSTALLED = False


@dataclass(frozen=True)
class _PendingKnockout:
    source_path: Path
    mode: str
    report: dict[str, Any]


_PENDING: ContextVar[dict[str, _PendingKnockout]] = ContextVar(
    "vektoryum_alpha_budget_retry_pending", default={}
)


def _path_key(path: Path) -> str:
    return str(Path(path).resolve())


def _register(path: Path, pending: _PendingKnockout) -> None:
    values = dict(_PENDING.get())
    values[_path_key(path)] = pending
    _PENDING.set(values)


def _take(path: Path) -> _PendingKnockout | None:
    values = dict(_PENDING.get())
    pending = values.pop(_path_key(path), None)
    _PENDING.set(values)
    return pending


def _quantize_alpha_levels(
    alpha: np.ndarray,
    level_count: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Mevcut alpha eşiklerini değiştirmeden açık seviye sayısına kuantize et."""
    if level_count < 2 or level_count > 256:
        raise ValueError("alpha seviye sayısı [2, 256] aralığında olmalı")
    steps = int(level_count) - 1
    indexes = np.rint(
        np.asarray(alpha, dtype=np.float32) * steps / 255.0
    ).astype(np.uint8)
    opacity = {
        int(value): int(value) / float(steps)
        for value in np.unique(indexes)
        if int(value) > 0
    }
    return indexes, opacity


def _compact_knockout_candidate(
    svg_path: Path,
    source_path: Path,
    mode: str,
    *,
    alpha_levels: int,
    preferred_clip_encoding: str,
) -> dict[str, Any]:
    """Tek kompakt adayı üret ve değişmemiş production kapılarıyla kanıtla."""
    from app.alpha_candidate_knockout import (
        _CLIP_GEOMETRY_ENCODINGS,
        _MAX_RECONSTRUCTION_SIDE,
        _SVG_NS,
        _XLINK_NS,
        _build_reconstruction_tree,
        _comparison_canvas_candidate,
        _local_name,
        _path_node_counts,
        _render_root,
        _sha256,
        _validate_reconstruction,
        _viewbox,
        _write_tree_to_temp,
    )
    from app.alpha_mask_budget import _journal_limits
    from app.alpha_preprocess import _rgba_from_source_at_size
    from app.source_truth import resize_rgba

    target = Path(svg_path)
    source = Path(source_path)
    before_sha = _sha256(target)
    before_size = target.stat().st_size

    # ElementTree aksi hâlde aynı namespace'i ns0 önekiyle yazabilir. Bu yalnız
    # serileştirme farkıdır fakat byte bütçesini gereksiz büyütmemesi için mevcut
    # knockout üreticisinin namespace sözleşmesi aynen korunur.
    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)
    original_root = ET.parse(target).getroot()
    if _local_name(str(original_root.tag)).lower() != "svg":
        raise RuntimeError("source_alpha_candidate_knockout_root_not_svg")
    parent_counts = _path_node_counts(original_root)

    _view_x, _view_y, view_width, view_height = _viewbox(original_root)
    scale = min(1.0, _MAX_RECONSTRUCTION_SIDE / float(max(view_width, view_height)))
    raster_width = max(1, int(round(view_width * scale)))
    raster_height = max(1, int(round(view_height * scale)))
    source_rgba = _rgba_from_source_at_size(source, (raster_width, raster_height))
    source_alpha = np.asarray(source_rgba[:, :, 3], dtype=np.uint8).copy()
    if bool(np.all(source_alpha == 255)):
        raise RuntimeError("source_alpha_candidate_knockout_opaque_source")
    if not np.any(source_alpha > 0):
        raise RuntimeError("source_alpha_candidate_knockout_empty_source")

    eval_scale = min(1.0, 256.0 / float(max(raster_width, raster_height)))
    eval_width = max(1, int(round(raster_width * eval_scale)))
    eval_height = max(1, int(round(raster_height * eval_scale)))
    source_eval = resize_rgba(source_rgba, eval_width, eval_height)
    collapsed = _render_root(copy.deepcopy(original_root), eval_width, eval_height)
    if collapsed is None:
        raise RuntimeError("source_alpha_candidate_knockout_parent_render_unmeasured")
    collapsed_coverage = float(collapsed[:, :, 3].astype(np.float32).mean() / 255.0)
    if collapsed_coverage < 0.98:
        raise RuntimeError(
            "source_alpha_candidate_knockout_parent_not_collapsed:"
            f"{collapsed_coverage:.6f}<0.98"
        )
    canvas = _comparison_canvas_candidate(
        original_root,
        source_eval,
        eval_width,
        eval_height,
    )
    quantized, opacity_by_level = _quantize_alpha_levels(source_alpha, alpha_levels)
    limits = _journal_limits(original_root, before_size)

    encodings = [preferred_clip_encoding]
    encodings.extend(
        encoding
        for encoding in _CLIP_GEOMETRY_ENCODINGS
        if encoding not in encodings
    )
    last_error: str | None = None
    for clip_encoding in encodings:
        candidate_root, geometry = _build_reconstruction_tree(
            original_root,
            canvas,
            quantized,
            opacity_by_level,
            clip_encoding=clip_encoding,
        )
        temporary = _write_tree_to_temp(candidate_root, target)
        try:
            after_size = temporary.stat().st_size
            if after_size > int(limits["byte_limit"]):
                last_error = (
                    "source_alpha_candidate_knockout_byte_budget_rejected:"
                    f"{after_size}>{limits['byte_limit']}"
                )
                continue
            with Image.open(source) as source_image:
                source_size = source_image.size
            source_rgba_full = _rgba_from_source_at_size(source, source_size)
            validation = _validate_reconstruction(
                temporary,
                source_rgba_full,
                mode,
                parent_counts,
            )
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return {
            "status": "accepted",
            "applied": True,
            "schema": "rfv3d3-candidate-geometry-knockout-compact-v1",
            "before_sha256": before_sha,
            "after_sha256": _sha256(target),
            "before_byte_size": int(before_size),
            "after_byte_size": int(target.stat().st_size),
            "threshold_image_class": {
                "geometric_logo": "geometric",
                "minimal_ai": "clean_logo",
                "flat_logo": "clean_logo",
                "logo_color": "clean_logo",
                "photo_poster": "photo",
            }.get(mode, "clean_logo"),
            "mask_encoding": "candidate_geometry_knockout",
            "reconstruction_alpha_encoding": f"quantized_{alpha_levels}",
            "compact_alpha_level_limit": int(alpha_levels),
            "candidate_canvas_knockout_count": 1,
            "candidate_geometry_preserved": True,
            "trace_rgb_bytes_preserved": True,
            "candidate_identity_preserved": True,
            "preflight_parent_path_count": int(parent_counts[0]),
            "preflight_parent_node_count": int(parent_counts[1]),
            "preflight_path_limit": int(limits["path_limit"]),
            "preflight_node_limit": int(limits["node_limit"]),
            "preflight_byte_limit": int(limits["byte_limit"]),
            **geometry,
            **validation,
        }
    raise RuntimeError(
        last_error
        or "source_alpha_candidate_knockout_no_admissible_compact_encoding"
    )


def _budget_only(stage: dict[str, Any]) -> bool:
    return (
        stage.get("status") == "budget_exhausted"
        and list(stage.get("reason_codes") or [])
        == ["evaluation_budget_exhausted"]
    )


def _new_journal_like(journal: Any) -> Any:
    """Aynı production sınırlarıyla bağımsız bir aday journal'ı oluştur."""
    return type(journal)(
        journal.baseline_path,
        journal.source_rgb,
        image_class=journal.image_class,
        required_metrics=set(journal.required_metrics),
        budget_seconds=journal.budget_seconds,
        stage_timeout_seconds=journal.stage_timeout_seconds,
        max_side=journal.max_side,
    )


def _adopt_journal_state(target: Any, source: Any) -> None:
    """Kabul edilen retry journal'ını çağıranın beklediği nesneye aktar."""
    target.__dict__.clear()
    target.__dict__.update(source.__dict__)


def _wrap_apply_source_alpha_mask(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    if getattr(original, "__vektoryum_alpha_budget_retry_context__", False):
        return original

    @wraps(original)
    def wrapped(svg_path: Path, source_path: Path, mode: str) -> dict[str, Any]:
        report = original(Path(svg_path), Path(source_path), mode)
        if (
            isinstance(report, dict)
            and report.get("status") == "accepted"
            and report.get("mask_encoding") == "candidate_geometry_knockout"
        ):
            _register(
                Path(svg_path),
                _PendingKnockout(Path(source_path), str(mode), report),
            )
        return report

    wrapped.__vektoryum_alpha_budget_retry_context__ = True
    return wrapped


def _wrap_consider_candidate(
    original: Callable[..., tuple[Path, dict[str, Any]]],
) -> Callable[..., tuple[Path, dict[str, Any]]]:
    if getattr(original, "__vektoryum_alpha_budget_retry__", False):
        return original

    @wraps(original)
    def wrapped(
        journal: Any,
        stage_id: str,
        parent_path: Path,
        candidate_path: Path,
        *,
        transform_report: Any = None,
    ) -> tuple[Path, dict[str, Any]]:
        accepted_path, stage = original(
            journal,
            stage_id,
            parent_path,
            candidate_path,
            transform_report=transform_report,
        )
        if stage_id != "source_alpha_vector_mask":
            return accepted_path, stage

        pending = _take(Path(candidate_path))
        if pending is None or not _budget_only(stage):
            return accepted_path, stage

        parent = Path(parent_path)
        candidate = Path(candidate_path)
        parent_data = parent.read_bytes()
        exact_candidate = candidate.read_bytes()
        exact_candidate_sha = hashlib.sha256(exact_candidate).hexdigest()
        preferred = str(
            pending.report.get("reconstruction_clip_encoding")
            or "rect-transform"
        )
        attempts: list[dict[str, Any]] = []

        for alpha_levels in _COMPACT_ALPHA_LEVELS:
            candidate.write_bytes(parent_data)
            try:
                compact_report = _compact_knockout_candidate(
                    candidate,
                    pending.source_path,
                    pending.mode,
                    alpha_levels=alpha_levels,
                    preferred_clip_encoding=preferred,
                )
            except RuntimeError as exc:
                message = str(exc)
                attempts.append({
                    "alpha_levels": int(alpha_levels),
                    "status": "candidate_rejected",
                    "reason": message,
                })
                # Byte maliyeti daha düşük seviyede yeniden denenebilir. Alpha,
                # evaluator, kimlik veya renderer kapısı reddiyse daha kaba adaya
                # geçmek güvenli değildir; fail-closed durulur.
                if "byte_budget" not in message:
                    break
                continue
            except Exception as exc:  # diagnostic retry üretimi ana hatayı maskelemez
                attempts.append({
                    "alpha_levels": int(alpha_levels),
                    "status": "candidate_error",
                    "reason": f"{type(exc).__name__}:{exc}",
                })
                break

            retry_journal = _new_journal_like(journal)
            retry_path, retry_stage = original(
                retry_journal,
                stage_id,
                parent,
                candidate,
                transform_report=compact_report,
            )
            attempts.append({
                "alpha_levels": int(alpha_levels),
                "status": str(retry_stage.get("status")),
                "reason_codes": list(retry_stage.get("reason_codes") or []),
                "evaluation_seconds": round(
                    float(retry_journal.evaluation_seconds),
                    6,
                ),
                "candidate_byte_size": int(candidate.stat().st_size),
                "rectangle_count": int(
                    compact_report.get("reconstruction_rectangle_count") or 0
                ),
            })
            if Path(retry_path) == candidate:
                compact_report["mask_fallback_reason"] = (
                    "journal_evaluation_budget_exhausted"
                )
                compact_report["mask_fallback_trigger"] = (
                    "evaluation_budget_exhausted"
                )
                compact_report["exact_candidate_sha256"] = exact_candidate_sha
                compact_report["compact_retry_attempts"] = attempts
                if isinstance(transform_report, dict):
                    transform_report.clear()
                    transform_report.update(compact_report)
                _adopt_journal_state(journal, retry_journal)
                return retry_path, retry_stage
            if not _budget_only(retry_stage):
                break

        # Hiçbir kompakt aday kabul edilmediyse ilk, kanıtlanmış exact/128 aday
        # baytı ve onun fail-closed budget sonucu korunur.
        candidate.write_bytes(exact_candidate)
        if isinstance(transform_report, dict):
            transform_report["compact_retry_attempts"] = attempts
        return accepted_path, stage

    wrapped.__vektoryum_alpha_budget_retry__ = True
    return wrapped


def install_alpha_budget_retry() -> None:
    """Public pipeline girişinde idempotent runtime bağlarını kur."""
    global _INSTALLED
    if _INSTALLED:
        return
    from app import alpha_svg_mask
    from app.transform_journal import TransformJournal

    alpha_svg_mask.apply_source_alpha_mask = _wrap_apply_source_alpha_mask(
        alpha_svg_mask.apply_source_alpha_mask
    )
    TransformJournal.consider_candidate = _wrap_consider_candidate(
        TransformJournal.consider_candidate
    )
    _INSTALLED = True
