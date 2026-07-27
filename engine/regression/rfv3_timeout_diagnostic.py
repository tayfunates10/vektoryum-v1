"""RFV-3D3 tek-vaka production pipeline süre tanılama aracı.

Bu modül production davranışını değiştirmez. Yalnız açıkça çağrılan diagnostic
sürecinde ``sys.setprofile`` ile seçili Python giriş noktalarının sürelerini ölçer.
Ham corpus varlığı rapora yazılmaz; yalnız vaka kimliği, kaynak SHA-256 ve üretilen
SVG'lerin boyut/sayım özetleri tutulur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SCHEMA = "vektoryum-rfv3d3-timeout-diagnostic-v1"
DEFAULT_HEARTBEAT_SECONDS = 30.0
DEFAULT_NEAR_TIMEOUT_SECONDS = 3000.0
DEFAULT_TRACE_CAP = 2200


@dataclass
class _OpenCall:
    stage: str
    function: str
    started: float
    stage_root: bool
    metadata: dict[str, Any]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_path(value: object) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _svg_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file() or path.is_symlink():
        return {"svg_bytes": None, "path_count": None, "svg_sha256": None}
    try:
        data = path.read_bytes()
    except OSError:
        return {"svg_bytes": None, "path_count": None, "svg_sha256": None}
    return {
        "svg_bytes": len(data),
        "path_count": data.count(b"<path"),
        "svg_sha256": hashlib.sha256(data).hexdigest(),
    }


def _image_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file() or path.is_symlink():
        return {"input_bytes": None, "input_width": None, "input_height": None}
    width = height = None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # diagnostic metadata must never alter the measured call
        pass
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {"input_bytes": size, "input_width": width, "input_height": height}


def _classify_frame(frame: Any) -> tuple[str | None, str | None]:
    filename = str(frame.f_code.co_filename).replace("\\", "/")
    function = str(frame.f_code.co_name)

    if filename.endswith("/app/source_truth.py") and function == "render_svg_to_rgba":
        return "render", "app.source_truth.render_svg_to_rgba"
    if filename.endswith("/app/fidelity.py") and function == "render_svg_to_rgb":
        return "render", "app.fidelity.render_svg_to_rgb"

    if function in {"analyze_image_from_mem", "_analyze_image_from_mem_with_contract"}:
        return "analyzer", function
    if function == "preprocess_for_mode" or (
        filename.endswith("/app/alpha_preprocess.py") and "preprocess" in function
    ):
        return "preprocess", function
    if filename.endswith("/app/vector_engines.py") and function == "run_candidate":
        return "tracer_candidates", function
    if filename.endswith("/app/gradient_vectorize.py") and function == "vectorize_with_gradients":
        return "tracer_candidates", function
    if any(
        filename.endswith(suffix)
        for suffix in (
            "/app/shape_fitting.py",
            "/app/curve_fairing.py",
            "/app/geometry_cleanup.py",
        )
    ) and not function.startswith("_"):
        return "shape_fitting", function
    if filename.endswith("/app/scoring.py") and function == "score_vector_candidate":
        return "perceptual_scoring", function

    if filename.endswith("/app/alpha_candidate_direct.py"):
        return "alpha.direct", function
    if filename.endswith("/app/alpha_soft_ellipse.py"):
        return "alpha.soft_ellipse", function
    if filename.endswith("/app/alpha_candidate_painter.py"):
        return "alpha.painter", function
    if filename.endswith("/app/alpha_candidate_knockout.py"):
        return "alpha.knockout", function
    if any(
        filename.endswith(suffix)
        for suffix in (
            "/app/alpha_candidate_support.py",
            "/app/alpha_candidate_support_compact.py",
        )
    ):
        return "alpha.support", function
    if filename.endswith("/app/alpha_svg_mask.py") and function in {
        "apply_source_alpha_mask",
        "alpha_mask_finalized_pipeline",
    }:
        return "alpha.direct", function

    if (
        filename.endswith("/app/pipeline.py")
        and function
        in {
            "refine_best",
            "_refit_one",
            "_refit_candidates_preselect",
            "_apply_boundary_refit",
        }
    ) or any(
        filename.endswith(suffix)
        for suffix in (
            "/app/boundary_refit.py",
            "/app/color_refit.py",
            "/app/component_align.py",
            "/app/counter_merge.py",
            "/app/local_refine.py",
            "/app/cusp_refine.py",
        )
    ):
        return "refinement", function
    if filename.endswith("/app/edge_cleanup.py") and function == "apply_edge_cleanup":
        return "edge_cleanup", function
    if filename.endswith("/benchmark/pipeline_results.py") and function == "_exact_winner_metrics":
        return "export", function
    if filename.endswith("/app/final_artifact_evaluator.py") and function == "evaluate_final_svg":
        return "export", function
    return None, None


class DiagnosticTracker:
    """Seçili çağrıları kapsayıcı süreyle ölçer ve dayanıklı ilerleme yazar."""

    def __init__(
        self,
        *,
        output_dir: Path,
        case_id: str,
        source_sha256: str,
        engine_version: str,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        near_timeout_seconds: float = DEFAULT_NEAR_TIMEOUT_SECONDS,
        classifier: Callable[[Any], tuple[str | None, str | None]] = _classify_frame,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.case_id = case_id
        self.source_sha256 = source_sha256
        self.engine_version = engine_version
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.near_timeout_seconds = float(near_timeout_seconds)
        self.classifier = classifier
        self.started = time.perf_counter()
        self.status = "running"
        self.error: str | None = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.heartbeat_thread: threading.Thread | None = None
        self.open_calls: dict[int, _OpenCall] = {}
        self.stack: list[int] = []
        self.stage_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"call_count": 0, "total_seconds": 0.0, "longest_seconds": 0.0, "longest_call": None}
        )
        self.function_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"call_count": 0, "total_seconds": 0.0, "longest_seconds": 0.0}
        )
        self.renders: list[dict[str, Any]] = []
        self.tracers: list[dict[str, Any]] = []
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.near_timeout_written = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _event(self, event: str, **values: Any) -> None:
        payload = {
            "elapsed_seconds": round(self._elapsed(), 6),
            "event": event,
            **values,
        }
        self.events.append(payload)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()

    def _call_metadata(self, stage: str, frame: Any) -> dict[str, Any]:
        local = frame.f_locals
        if stage == "render":
            svg_path = _safe_path(local.get("svg_path") or local.get("path"))
            return {"svg_path": svg_path}
        if stage == "tracer_candidates":
            output_path = _safe_path(
                local.get("output_path")
                or local.get("svg_path")
                or local.get("output")
            )
            input_path = _safe_path(
                local.get("input_path")
                or local.get("image_path")
                or local.get("preprocessed_path")
                or local.get("source_path")
            )
            spec = local.get("spec") if isinstance(local.get("spec"), dict) else {}
            return {
                "engine": str(local.get("engine") or spec.get("engine") or "unknown"),
                "candidate": output_path.stem if output_path is not None else None,
                "output_path": output_path,
                "input_path": input_path,
                "trace_cap_effective": int(os.environ.get("VEKTORYUM_TRACE_CAP", str(DEFAULT_TRACE_CAP))),
                "vtracer_params": dict(spec.get("vtracer_params") or {}),
            }
        return {}

    def profile(self, frame: Any, event: str, arg: Any) -> None:
        frame_id = id(frame)
        if event == "call":
            stage, function = self.classifier(frame)
            if stage is None or function is None:
                return
            with self.lock:
                stage_root = not any(
                    self.open_calls[item].stage == stage
                    for item in self.stack
                    if item in self.open_calls
                )
                opened = _OpenCall(
                    stage=stage,
                    function=function,
                    started=time.perf_counter(),
                    stage_root=stage_root,
                    metadata=self._call_metadata(stage, frame),
                )
                self.open_calls[frame_id] = opened
                self.stack.append(frame_id)
                if stage_root:
                    self._event("stage_enter", stage=stage, function=function)
            return

        if event not in {"return", "exception"}:
            return
        with self.lock:
            opened = self.open_calls.pop(frame_id, None)
            if opened is None:
                return
            if frame_id in self.stack:
                self.stack.remove(frame_id)
            duration = max(0.0, time.perf_counter() - opened.started)
            function_key = f"{opened.stage}:{opened.function}"
            function_stat = self.function_stats[function_key]
            function_stat["call_count"] += 1
            function_stat["total_seconds"] += duration
            function_stat["longest_seconds"] = max(function_stat["longest_seconds"], duration)

            if opened.stage_root:
                stat = self.stage_stats[opened.stage]
                stat["call_count"] += 1
                stat["total_seconds"] += duration
                if duration >= stat["longest_seconds"]:
                    stat["longest_seconds"] = duration
                    stat["longest_call"] = opened.function
                self._event(
                    "stage_exit",
                    stage=opened.stage,
                    function=opened.function,
                    duration_seconds=round(duration, 6),
                    outcome="exception" if event == "exception" else "return",
                )

            if opened.stage == "render":
                metadata = _svg_metadata(opened.metadata.get("svg_path"))
                self.renders.append(
                    {
                        "function": opened.function,
                        "duration_seconds": round(duration, 6),
                        **metadata,
                    }
                )
            elif opened.stage == "tracer_candidates":
                output_metadata = _svg_metadata(opened.metadata.get("output_path"))
                input_metadata = _image_metadata(opened.metadata.get("input_path"))
                self.tracers.append(
                    {
                        "function": opened.function,
                        "engine": opened.metadata.get("engine"),
                        "candidate": opened.metadata.get("candidate"),
                        "duration_seconds": round(duration, 6),
                        "trace_cap_effective": opened.metadata.get("trace_cap_effective"),
                        "vtracer_params": opened.metadata.get("vtracer_params"),
                        **input_metadata,
                        **output_metadata,
                    }
                )

    def _active_call(self) -> dict[str, Any] | None:
        for frame_id in reversed(self.stack):
            opened = self.open_calls.get(frame_id)
            if opened is not None:
                return {
                    "stage": opened.stage,
                    "function": opened.function,
                    "active_seconds": round(time.perf_counter() - opened.started, 6),
                }
        return None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            renders = list(self.renders)
            tracers = list(self.tracers)
            longest_render = max(
                renders,
                key=lambda item: float(item["duration_seconds"]),
                default=None,
            )
            render_by_function: dict[str, dict[str, Any]] = {}
            for name in sorted({item["function"] for item in renders}):
                selected = [item for item in renders if item["function"] == name]
                render_by_function[name] = {
                    "call_count": len(selected),
                    "total_seconds": round(sum(float(item["duration_seconds"]) for item in selected), 6),
                    "longest_seconds": round(max(float(item["duration_seconds"]) for item in selected), 6),
                }
            stages = {
                name: {
                    "call_count": int(value["call_count"]),
                    "total_seconds": round(float(value["total_seconds"]), 6),
                    "longest_seconds": round(float(value["longest_seconds"]), 6),
                    "longest_call": value["longest_call"],
                }
                for name, value in sorted(self.stage_stats.items())
            }
            functions = {
                name: {
                    "call_count": int(value["call_count"]),
                    "total_seconds": round(float(value["total_seconds"]), 6),
                    "longest_seconds": round(float(value["longest_seconds"]), 6),
                }
                for name, value in sorted(self.function_stats.items())
            }
            return {
                "schema": SCHEMA,
                "status": self.status,
                "error": self.error,
                "case_id": self.case_id,
                "source_sha256": self.source_sha256,
                "engine_version": self.engine_version,
                "elapsed_seconds": round(self._elapsed(), 6),
                "active_call": self._active_call(),
                "last_events": list(self.events)[-10:],
                "stage_timing_semantics": "inclusive_root_call_wall_time",
                "stages": stages,
                "functions": functions,
                "renders": {
                    "call_count": len(renders),
                    "total_seconds": round(sum(float(item["duration_seconds"]) for item in renders), 6),
                    "longest": longest_render,
                    "by_function": render_by_function,
                },
                "tracers": {
                    "call_count": len(tracers),
                    "trace_cap_effective": int(os.environ.get("VEKTORYUM_TRACE_CAP", str(DEFAULT_TRACE_CAP))),
                    "calls": tracers,
                },
                "diagnostic_only": True,
                "production_behavior_changed": False,
                "raw_assets_in_artifact": False,
            }

    def write_snapshot(self, name: str = "diagnostic.json") -> None:
        _atomic_json(self.output_dir / name, self.snapshot())

    def _heartbeat_loop(self) -> None:
        next_tick = self.heartbeat_seconds
        while not self.stop_event.wait(max(0.1, next_tick - self._elapsed())):
            with self.lock:
                active = self._active_call()
                elapsed = self._elapsed()
                self._event("heartbeat", active_call=active)
                self.write_snapshot("progress.json")
                print(
                    "RFV3D3_DIAG "
                    + json.dumps(
                        {
                            "case_id": self.case_id,
                            "elapsed_seconds": round(elapsed, 3),
                            "active_call": active,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if elapsed >= self.near_timeout_seconds and not self.near_timeout_written:
                    self.near_timeout_written = True
                    self._event("near_timeout", active_call=active)
                    self.write_snapshot("near-timeout.json")
            next_tick += self.heartbeat_seconds

    def start(self) -> None:
        self._event("diagnostic_start")
        self.write_snapshot("progress.json")
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="rfv3d3-diagnostic-heartbeat",
            daemon=True,
        )
        self.heartbeat_thread.start()

    def finish(self, *, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.stop_event.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=2.0)
        self._event("diagnostic_finish", status=status, error=error)
        self.write_snapshot("diagnostic.json")
        self.write_snapshot("progress.json")


def run_diagnostic(
    *,
    corpus_root: Path,
    output_dir: Path,
    case_id: str,
    engine_version: str,
    heartbeat_seconds: float,
    near_timeout_seconds: float,
) -> int:
    from benchmark.pipeline_results import run_case
    from app.pipeline_entry import run_pipeline
    from engine.regression.rfv3_measurement_runner import load_qualification_cases

    cases = load_qualification_cases(Path(corpus_root))
    matches = [case for case in cases if case.case_id == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"diagnostic_case_not_found:{case_id}")
    case = matches[0]
    tracker = DiagnosticTracker(
        output_dir=Path(output_dir),
        case_id=case.case_id,
        source_sha256=case.source_sha256,
        engine_version=engine_version,
        heartbeat_seconds=heartbeat_seconds,
        near_timeout_seconds=near_timeout_seconds,
    )

    def _signal_handler(signum: int, _frame: Any) -> None:
        tracker._event("signal", signal=signum, active_call=tracker._active_call())
        tracker.write_snapshot("terminated.json")
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, _signal_handler)
    previous_int = signal.signal(signal.SIGINT, _signal_handler)
    tracker.start()
    sys.setprofile(tracker.profile)
    try:
        result = run_case(
            case,
            corpus_root=Path(corpus_root),
            work_root=Path(output_dir) / "work",
            pipeline=run_pipeline,
            engine_version=engine_version,
            trace_mode="auto",
            peak_rss_mb=None,
        )
    except BaseException as exc:
        sys.setprofile(None)
        message = f"{type(exc).__name__}: {exc}"
        tracker.finish(status="failure", error=message)
        raise
    else:
        sys.setprofile(None)
        _atomic_json(Path(output_dir) / "result-summary.json", result.to_dict())
        tracker.finish(status="success")
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--near-timeout-seconds", type=float, default=DEFAULT_NEAR_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run_diagnostic(
        corpus_root=args.corpus_root,
        output_dir=args.output,
        case_id=args.case_id,
        engine_version=args.engine_version,
        heartbeat_seconds=args.heartbeat_seconds,
        near_timeout_seconds=args.near_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
