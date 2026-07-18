"""Production SVG mutasyonları için ölçüm-kapılı byte transaction journal'ı.

Her mutator ayrı candidate baytı üretir. Yapı, topoloji, seam, görsel sadakat
ve complexity gate'leri gerilerse kabul edilmiş parent baytı atomik geri yazılır.
Global ağırlıklı skor hard invariant ihlalini telafi edemez.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app import source_truth as _source_truth


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".txn", delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(high, max(low, value))


def _measure_svg_bytes(
    data: bytes,
    source_rgb: np.ndarray,
    *,
    max_side: int = 512,
    required_metrics: set[str] | None = None,
    measure_alpha: bool = False,
    _allow_topology_refinement: bool = True,
) -> dict[str, Any]:
    """Stage gate için bounded structural + source-fidelity ölçümü."""
    from app.final_artifact_evaluator import (
        _classify,
        _derive_palette,
        _seam_ratio,
        _structure_check,
        _topology_signature,
    )
    from app.fidelity import _edge_f1, _ssim, render_svg_to_rgb

    struct, messages, codes, root = _structure_check(data)
    metric: dict[str, Any] = {
        "sha256": _sha(data),
        "byte_size": len(data),
        "structural_safe": not codes,
        "structural_failure_codes": list(codes),
        "structural_failures": list(messages),
        "path_count": int(struct.get("path_count") or 0),
        "node_count": int(struct.get("node_count") or 0),
        "gradient_definition_count": int(struct.get("gradient_definition_count") or 0),
        "required_unmeasured": sorted(required_metrics or ()),
    }
    render_data = data
    if codes:
        # Engine'in ilk ham SVG'si bazen viewBox yerine yalnız finite
        # width/height taşır. Bu artifact production-safe değildir; fakat
        # coordinate-contract dönüşümünün gerçekten gerileme olup olmadığını
        # ölçmek için immutable ölçüm kopyasına eşdeğer viewBox eklenebilir.
        repairable = set(codes) == {"viewbox_missing"} and root is not None
        if repairable:
            def _dimension(value: Any) -> float | None:
                match = re.fullmatch(
                    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:px)?\s*",
                    str(value or ""),
                )
                if not match:
                    return None
                parsed = float(match.group(1))
                return parsed if np.isfinite(parsed) and parsed > 0 else None

            width = _dimension(root.attrib.get("width"))
            height = _dimension(root.attrib.get("height"))
            if width is not None and height is not None:
                root.set("viewBox", f"0 0 {width:g} {height:g}")
                render_data = ET.tostring(root, encoding="utf-8")
                metric["measurement_repairs"] = ["viewbox_inferred_from_dimensions"]
            else:
                repairable = False
        if not repairable:
            return metric

    src = np.asarray(source_rgb, dtype=np.uint8)
    h0, w0 = src.shape[:2]
    if max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    else:
        h, w = h0, w0

    render_rgba = None
    with tempfile.TemporaryDirectory(prefix="vektoryum-stage-") as directory:
        path = Path(directory) / "candidate.svg"
        path.write_bytes(render_data)
        rnd = render_svg_to_rgb(path, w, h)
        if measure_alpha and "alpha_fidelity" in set(required_metrics or ()):
            render_rgba = _source_truth.render_svg_to_rgba(path, w, h)
    if rnd is None:
        metric["required_unmeasured"] = sorted(
            set(metric["required_unmeasured"]) | {"stage_render"}
        )
        return metric
    if rnd.shape[:2] != (h, w):
        rnd = cv2.resize(rnd, (w, h), interpolation=cv2.INTER_AREA)

    # TransformJournal'ın görevi final source-truth kararını yeniden vermek değil,
    # kabul edilmiş parent artifact'a göre her mutasyonun alpha düzlemini
    # koruduğunu kanıtlamaktır. Parent ve candidate aynı bounded RGBA renderer ile
    # ölçülür; ham ndarray yalnız özel cache anahtarında tutulur ve public journal
    # raporuna hiçbir zaman serileştirilmez. Renderer yoksa required metric açıkça
    # unmeasured kalır ve mevcut fail-closed karar yolu candidate'ı reddeder.
    if measure_alpha and "alpha_fidelity" in set(required_metrics or ()):
        if render_rgba is None:
            metric["alpha_fidelity_status"] = "unmeasured"
        else:
            if render_rgba.shape[:2] != (h, w):
                render_rgba = _source_truth.resize_rgba(render_rgba, w, h)
            render_alpha = np.asarray(render_rgba[:, :, 3], dtype=np.uint8).copy()
            metric["_render_alpha"] = render_alpha
            metric["alpha_fidelity_status"] = "measured"
            metric["alpha_coverage"] = float(render_alpha.astype(np.float32).mean() / 255.0)
            metric["alpha_sha256"] = hashlib.sha256(
                np.ascontiguousarray(render_alpha).tobytes()
            ).hexdigest()
            metric["required_unmeasured"] = [
                name for name in metric["required_unmeasured"]
                if name != "alpha_fidelity"
            ]

    ga = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(rnd, cv2.COLOR_RGB2GRAY)
    palette = _derive_palette(src)
    co = _classify(src, palette)
    cr = _classify(rnd, palette)
    min_area = max(6, round(0.00004 * w * h))
    ts_src = _topology_signature(co, len(palette), min_area)
    ts_rnd = _topology_signature(cr, len(palette), min_area)
    metric.update({
        "ssim": float(_ssim(ga, gb)),
        "edge_f1_1px": float(_edge_f1(gb, ga, tolerance=1)),
        "seam_ratio": float(_seam_ratio(src, rnd)),
        "component_delta": abs(ts_src["components"] - ts_rnd["components"]),
        "hole_delta": abs(ts_src["holes"] - ts_rnd["holes"]),
        "source_topology": ts_src,
        "render_topology": ts_rnd,
    })

    # 512px ölçüm, çok ince gerçek bileşenlerde AA örnekleme fazına bağlı
    # sahte component_delta üretebilir. Topoloji bir hard-veto olduğundan bu
    # farkı görmezden gelmek de, doğrudan veto etmek de güvenli değildir.
    # Yalnız coarse ölçüm bir topoloji farkı gördüğünde, kaynak 1024px veya
    # daha büyükse aynı artifact'ı ikinci ve bağımsız bir 1024px turda ölç.
    # Fine tur yalnız topoloji alanlarını doğrular; görsel eşikler coarse
    # bounded ölçümde kalır. Gerçek delik/bileşen kaybı fine turda da kalacağı
    # için veto edilmeye devam eder; LEGO ® gibi AA kaynaklı sahte fark ise
    # production dönüşümünü yanlışlıkla geri almaz.
    coarse_component_delta = int(metric["component_delta"])
    coarse_hole_delta = int(metric["hole_delta"])
    source_side = max(h0, w0)
    if (
        _allow_topology_refinement
        and (coarse_component_delta > 0 or coarse_hole_delta > 0)
        and source_side > max_side
        and max_side < 1024
    ):
        refined_side = min(1024, source_side)
        refined = _measure_svg_bytes(
            data,
            source_rgb,
            max_side=refined_side,
            required_metrics=required_metrics,
            measure_alpha=measure_alpha,
            _allow_topology_refinement=False,
        )
        if all(key in refined for key in (
            "component_delta", "hole_delta", "source_topology", "render_topology",
        )):
            metric["topology_refinement"] = {
                "coarse_max_side": max_side,
                "refined_max_side": refined_side,
                "coarse_component_delta": coarse_component_delta,
                "coarse_hole_delta": coarse_hole_delta,
                "refined_component_delta": int(refined["component_delta"]),
                "refined_hole_delta": int(refined["hole_delta"]),
            }
            metric["component_delta"] = int(refined["component_delta"])
            metric["hole_delta"] = int(refined["hole_delta"])
            metric["source_topology"] = refined["source_topology"]
            metric["render_topology"] = refined["render_topology"]
    return metric


class TransformJournal:
    """Bir seçilmiş SVG artifact zincirini transaction olarak yönetir."""

    schema_version = 1

    def __init__(
        self,
        baseline_path: Path,
        source_rgb: np.ndarray,
        *,
        image_class: str = "clean_logo",
        required_metrics: set[str] | None = None,
        budget_seconds: float | None = None,
        stage_timeout_seconds: float | None = None,
        max_side: int = 512,
    ) -> None:
        self.baseline_path = Path(baseline_path)
        baseline = self.baseline_path.read_bytes()
        self.baseline_sha256 = _sha(baseline)
        self.final_accepted_sha256 = self.baseline_sha256
        self.source_rgb = np.asarray(source_rgb, dtype=np.uint8)
        self.image_class = image_class
        self.required_metrics = set(required_metrics or ())
        # Alpha measurement is deliberately stage-scoped. The proven defect only
        # affects the mandatory coordinate-contract repair; opening alpha for every
        # downstream mutator would change previously fail-closed production scope.
        self._measurement_stage_id: str | None = None
        self.max_side = min(512, max(256, int(max_side)))
        self.budget_seconds = (
            budget_seconds if budget_seconds is not None
            else _env_float("VEKTORYUM_TRANSFORM_EVAL_BUDGET_S", 45.0, 5.0, 180.0)
        )
        self.stage_timeout_seconds = (
            stage_timeout_seconds if stage_timeout_seconds is not None
            else _env_float("VEKTORYUM_TRANSFORM_STAGE_TIMEOUT_S", 180.0, 5.0, 600.0)
        )
        self.started = time.perf_counter()
        self.evaluation_seconds = 0.0
        self.stages: list[dict[str, Any]] = []
        self._cache: dict[str, dict[str, Any]] = {}
        self.budget_exhausted = False

    def _elapsed(self) -> float:
        return self.evaluation_seconds

    def _wall_elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _measure(self, data: bytes) -> dict[str, Any]:
        sha = _sha(data)
        measure_alpha = self._measurement_stage_id == "restore_source_dimensions"
        cache_key = f"{sha}:alpha={int(measure_alpha)}"
        if cache_key not in self._cache:
            started = time.perf_counter()
            try:
                self._cache[cache_key] = _measure_svg_bytes(
                    data, self.source_rgb, max_side=self.max_side,
                    required_metrics=self.required_metrics,
                    measure_alpha=measure_alpha,
                )
            finally:
                self.evaluation_seconds += time.perf_counter() - started
        return self._cache[cache_key]

    @staticmethod
    def _deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
        keys = (
            "byte_size", "path_count", "node_count", "ssim", "edge_f1_1px",
            "seam_ratio", "component_delta", "hole_delta", "alpha_coverage",
        )
        result: dict[str, float] = {}
        for key in keys:
            if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float)):
                result[key] = float(after[key]) - float(before[key])
        return result

    @staticmethod
    def _alpha_comparison(
        before: dict[str, Any], after: dict[str, Any],
    ) -> dict[str, float] | None:
        parent_alpha = before.get("_render_alpha")
        candidate_alpha = after.get("_render_alpha")
        if not isinstance(parent_alpha, np.ndarray) or not isinstance(candidate_alpha, np.ndarray):
            return None
        if parent_alpha.shape != candidate_alpha.shape:
            return None
        return _source_truth.alpha_plane_metrics(parent_alpha, candidate_alpha)

    def _decide(self, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not after.get("structural_safe"):
            reasons.extend(after.get("structural_failure_codes") or ["structure_failed"])
        if after.get("required_unmeasured"):
            reasons.append("required_metric_unmeasured")
        if int(after.get("gradient_definition_count") or 0) < int(
            before.get("gradient_definition_count") or 0
        ):
            reasons.append("gradient_definition_loss")
        visual_keys = ("ssim", "edge_f1_1px", "seam_ratio", "component_delta", "hole_delta")
        if any(key not in after for key in visual_keys):
            reasons.append("stage_metrics_incomplete")

        if "alpha_fidelity" in self.required_metrics:
            alpha_comparison = self._alpha_comparison(before, after)
            if alpha_comparison is None:
                reasons.append("alpha_stage_metrics_incomplete")
            else:
                # Yeni eşik icat edilmez: final artifact evaluator'ın mevcut,
                # image-class bağlı alpha hard gate'leri parent/candidate koruma
                # karşılaştırmasında aynen yeniden kullanılır.
                from app.final_artifact_evaluator import _thresholds  # noqa: PLC0415

                alpha_thresholds = _thresholds(self.image_class, None)
                if float(alpha_comparison["alpha_iou"]) < alpha_thresholds["alpha_iou_min"]:
                    reasons.append("alpha_iou_regression")
                if float(alpha_comparison["alpha_mae"]) > alpha_thresholds["alpha_mae_max"]:
                    reasons.append("alpha_mae_regression")
        if reasons:
            return list(dict.fromkeys(reasons))

        before_has_visual = all(key in before for key in visual_keys)
        if before_has_visual:
            # Clean/artwork outputunda topoloji önceki kabul edilmiş artifact'tan
            # daha kötü olamaz. Photo sınıfında semantik renk topolojisi güvenilir
            # değildir; complexity ve render kapıları yine uygulanır.
            if self.image_class != "photo":
                if after["component_delta"] > before["component_delta"]:
                    reasons.append("topology_component_regression")
                if after["hole_delta"] > before["hole_delta"]:
                    reasons.append("topology_hole_regression")

            if after["ssim"] < before["ssim"] - 0.0005:
                reasons.append("ssim_regression")
            if after["edge_f1_1px"] < before["edge_f1_1px"] - 0.0015:
                reasons.append("edge_f1_regression")
            before_seam = float(before["seam_ratio"])
            if after["seam_ratio"] > max(before_seam + 0.0005, before_seam * 1.25):
                reasons.append("seam_regression")
        else:
            # Yapısal olarak eksik engine artifact'ı (çoğunlukla viewBox yok)
            # güvenli koordinat sözleşmesine onarılırken bilinmeyen baseline
            # metriği ASLA sıfır varsayılmaz. Candidate bağımsız, sıkı mutlak
            # source-fidelity kapılarını geçmelidir.
            absolute = {
                "geometric": (0.98, 0.985, 0.002),
                "clean_logo": (0.97, 0.98, 0.003),
                "lineart": (0.94, 0.97, 0.004),
                "photo": (0.75, 0.65, 0.025),
            }.get(self.image_class, (0.94, 0.95, 0.006))
            if self.image_class != "photo":
                if after["component_delta"] > 0:
                    reasons.append("topology_component_regression")
                if after["hole_delta"] > 0:
                    reasons.append("topology_hole_regression")
            if after["ssim"] < absolute[0]:
                reasons.append("ssim_absolute_gate")
            if after["edge_f1_1px"] < absolute[1]:
                reasons.append("edge_f1_absolute_gate")
            if after["seam_ratio"] > absolute[2]:
                reasons.append("seam_absolute_gate")

        # Hard complexity budget: iyileşen weighted skor bile patlamayı örtemez.
        bp = max(1, int(before.get("path_count") or 0))
        bn = max(1, int(before.get("node_count") or 0))
        bb = max(1, int(before.get("byte_size") or 0))
        if int(after.get("path_count") or 0) > max(bp * 4, bp + 500):
            reasons.append("path_complexity_explosion")
        if int(after.get("node_count") or 0) > max(bn * 4, bn + 2500):
            reasons.append("node_complexity_explosion")
        if int(after.get("byte_size") or 0) > max(bb * 3, bb + 250_000):
            reasons.append("byte_complexity_explosion")
        return list(dict.fromkeys(reasons))

    def _record(
        self,
        stage_id: str,
        parent_data: bytes,
        candidate_data: bytes,
        *,
        transform_report: Any = None,
        forced_status: str | None = None,
        forced_reasons: list[str] | None = None,
        exception_type: str | None = None,
        duration_ms: float = 0.0,
    ) -> tuple[bool, dict[str, Any]]:
        parent_sha = _sha(parent_data)
        candidate_sha = _sha(candidate_data)
        if candidate_sha == parent_sha and forced_status is None:
            stage = {
                "stage_id": stage_id,
                "transform_name": stage_id,
                "parent_sha256": parent_sha,
                "candidate_sha256": candidate_sha,
                "accepted_sha256": parent_sha,
                "input_byte_size": len(parent_data),
                "candidate_byte_size": len(candidate_data),
                "duration_ms": round(duration_ms, 3),
                "status": "no_op",
                "reason_codes": ["byte_identical"],
                "before_metrics": None,
                "after_metrics": None,
                "metric_deltas": {},
                "required_unmeasured": [],
                "exception_type": exception_type,
                "transform_report": transform_report,
            }
            self.stages.append(stage)
            self.final_accepted_sha256 = parent_sha
            return True, stage

        if forced_status is not None:
            accepted = forced_status == "accepted"
            before = self._cache.get(parent_sha)
            after = self._cache.get(candidate_sha)
            reasons = list(forced_reasons or [])
            status = forced_status
        else:
            previous_stage = self._measurement_stage_id
            self._measurement_stage_id = stage_id
            try:
                before = self._measure(parent_data)
                after = self._measure(candidate_data)
            finally:
                self._measurement_stage_id = previous_stage
            reasons = self._decide(before, after)
            accepted = not reasons
            status = "accepted" if accepted else "rolled_back"

        accepted_sha = candidate_sha if accepted else parent_sha
        alpha_comparison = self._alpha_comparison(before or {}, after or {})
        before_public = (
            _source_truth.public_metric_dict(before) if isinstance(before, dict) else before
        )
        after_public = (
            _source_truth.public_metric_dict(after) if isinstance(after, dict) else after
        )
        stage = {
            "stage_id": stage_id,
            "transform_name": stage_id,
            "parent_sha256": parent_sha,
            "candidate_sha256": candidate_sha,
            "accepted_sha256": accepted_sha,
            "input_byte_size": len(parent_data),
            "candidate_byte_size": len(candidate_data),
            "duration_ms": round(duration_ms, 3),
            "status": status,
            "reason_codes": reasons or ["metrics_non_regressing"],
            "before_metrics": before_public,
            "after_metrics": after_public,
            "alpha_comparison": alpha_comparison,
            "metric_deltas": self._deltas(before or {}, after or {}),
            "complexity_delta": {
                key: self._deltas(before or {}, after or {}).get(key)
                for key in ("byte_size", "path_count", "node_count")
            },
            "required_unmeasured": (after or {}).get("required_unmeasured", []),
            "exception_type": exception_type,
            "transform_report": transform_report,
        }
        self.stages.append(stage)
        self.final_accepted_sha256 = accepted_sha
        return accepted, stage

    def _budget_rollback(self, stage: dict[str, Any], parent_sha: str) -> None:
        """Ölçüm sırasında bütçe aşılırsa önceki kabul edilmiş SHA'ya dön."""
        self.budget_exhausted = True
        stage["status"] = "budget_exhausted"
        stage["accepted_sha256"] = parent_sha
        stage["reason_codes"] = ["evaluation_budget_exhausted"]
        self.final_accepted_sha256 = parent_sha

    def consider_candidate(
        self,
        stage_id: str,
        parent_path: Path,
        candidate_path: Path,
        *,
        transform_report: Any = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Out-of-place candidate'ı ölçer; kabul edilmezse parent path döner."""
        parent_path, candidate_path = Path(parent_path), Path(candidate_path)
        parent_data = parent_path.read_bytes()
        if self._elapsed() >= self.budget_seconds:
            self.budget_exhausted = True
            _accepted, stage = self._record(
                stage_id, parent_data, parent_data, transform_report=transform_report,
                forced_status="budget_exhausted", forced_reasons=["evaluation_budget_exhausted"],
            )
            return parent_path, stage
        try:
            candidate_data = candidate_path.read_bytes()
        except Exception as e:  # noqa: BLE001
            _accepted, stage = self._record(
                stage_id, parent_data, parent_data, transform_report=transform_report,
                forced_status="failed", forced_reasons=["candidate_read_failed", type(e).__name__],
            )
            return parent_path, stage
        start = time.perf_counter()
        accepted, stage = self._record(
            stage_id, parent_data, candidate_data, transform_report=transform_report,
        )
        stage["duration_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
        if self._elapsed() > self.budget_seconds:
            self._budget_rollback(stage, _sha(parent_data))
            accepted = False
        return (candidate_path if accepted else parent_path), stage

    def record_noop(
        self,
        stage_id: str,
        path: Path,
        *,
        reason_codes: list[str],
        transform_report: Any = None,
    ) -> dict[str, Any]:
        """Uygulanamayan/skipped mutator'u byte-identical no-op olarak kaydet."""
        data = Path(path).read_bytes()
        _accepted, stage = self._record(
            stage_id, data, data, transform_report=transform_report,
            forced_status="no_op", forced_reasons=reason_codes or ["not_applicable"],
        )
        return stage

    def run_in_place(
        self,
        stage_id: str,
        path: Path,
        transform: Callable[[Path], Any],
    ) -> tuple[bool, Any, dict[str, Any]]:
        """In-place API'li mutator'u kopyada çalıştırır; kabulde atomik yayınlar.

        Mutator hiçbir anda kabul edilmiş ``path`` baytlarına dokunmaz. Benzersiz,
        aynı-dizin bir SVG kopyası üzerinde çalışır; yalnız bütün gate'ler
        geçerse candidate ham baytları atomik olarak asıl path'e taşınır.
        """
        path = Path(path)
        parent_data = path.read_bytes()
        if self._elapsed() >= self.budget_seconds:
            self.budget_exhausted = True
            _accepted, stage = self._record(
                stage_id, parent_data, parent_data,
                forced_status="budget_exhausted", forced_reasons=["evaluation_budget_exhausted"],
            )
            return False, None, stage
        start = time.perf_counter()
        candidate_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.stem}.",
                suffix=".candidate.svg", delete=False,
            ) as handle:
                candidate_path = Path(handle.name)
                handle.write(parent_data)
                handle.flush()
                os.fsync(handle.fileno())
            transform_report = transform(candidate_path)
            candidate_data = candidate_path.read_bytes()
            transform_seconds = time.perf_counter() - start
            if transform_seconds > self.stage_timeout_seconds:
                self.budget_exhausted = True
                _accepted, stage = self._record(
                    stage_id, parent_data, candidate_data,
                    transform_report=transform_report,
                    forced_status="budget_exhausted",
                    forced_reasons=["transform_stage_timeout"],
                    duration_ms=transform_seconds * 1000.0,
                )
                return False, transform_report, stage
            accepted, stage = self._record(
                stage_id, parent_data, candidate_data, transform_report=transform_report,
            )
            stage["duration_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
            if self._elapsed() > self.budget_seconds:
                self._budget_rollback(stage, _sha(parent_data))
                accepted = False
            if accepted and candidate_data != parent_data:
                _atomic_write_bytes(path, candidate_data)
        except Exception as e:  # noqa: BLE001
            _accepted, stage = self._record(
                stage_id, parent_data, parent_data,
                transform_report={"exception_type": type(e).__name__},
                forced_status="failed", forced_reasons=["transform_exception"],
                exception_type=type(e).__name__,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            return False, None, stage
        finally:
            if candidate_path is not None:
                try:
                    candidate_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return accepted, transform_report, stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_sha256": self.baseline_sha256,
            "final_accepted_sha256": self.final_accepted_sha256,
            "stages": self.stages,
            "budget": {
                "seconds": self.budget_seconds,
                "elapsed_seconds": round(self._elapsed(), 4),
                "wall_seconds": round(self._wall_elapsed(), 4),
                "stage_timeout_seconds": self.stage_timeout_seconds,
                "max_side": self.max_side,
            },
            "budget_exhausted": self.budget_exhausted,
        }


def merge_journal_reports(*reports: dict[str, Any] | None) -> dict[str, Any] | None:
    valid = [report for report in reports if report]
    if not valid:
        return None
    stages = [stage for report in valid for stage in report.get("stages", [])]
    chain_codes: list[str] = []
    expected = valid[0].get("baseline_sha256")
    for stage in stages:
        if expected and stage.get("parent_sha256") != expected:
            chain_codes.append("stage_parent_hash_mismatch")
        expected = stage.get("accepted_sha256") or expected
    for left, right in zip(valid, valid[1:]):
        if left.get("final_accepted_sha256") != right.get("baseline_sha256"):
            chain_codes.append("journal_boundary_hash_mismatch")
    reported_final = valid[-1].get("final_accepted_sha256")
    if expected and reported_final and expected != reported_final:
        chain_codes.append("journal_final_hash_mismatch")

    return {
        "schema_version": max(int(r.get("schema_version", 1)) for r in valid),
        "baseline_sha256": valid[0].get("baseline_sha256"),
        "final_accepted_sha256": valid[-1].get("final_accepted_sha256"),
        "stages": stages,
        "budget": {
            "seconds": sum(float((r.get("budget") or {}).get("seconds", 0)) for r in valid),
            "elapsed_seconds": round(sum(
                float((r.get("budget") or {}).get("elapsed_seconds", 0)) for r in valid
            ), 4),
            "wall_seconds": round(sum(
                float((r.get("budget") or {}).get("wall_seconds", 0)) for r in valid
            ), 4),
            "stage_timeout_seconds": max(
                float((r.get("budget") or {}).get("stage_timeout_seconds", 0))
                for r in valid
            ),
            "max_side": max(
                int((r.get("budget") or {}).get("max_side", 0)) for r in valid
            ),
        },
        "budget_exhausted": any(bool(r.get("budget_exhausted")) for r in valid),
        "chain_valid": not chain_codes,
        "chain_failure_codes": list(dict.fromkeys(chain_codes)),
    }
