"""Application package runtime bindings."""
from __future__ import annotations

from functools import wraps

# ``app.pipeline`` imports ``app.vector_engines`` immediately after package
# initialization, so loading that existing module here adds no new dependency.
from app import vector_engines as _vector_engines


def _lazy_graph_centerline(*args, **kwargs):
    from app.centerline_svg import vectorize_skeleton_graph_to_svg  # noqa: PLC0415

    return vectorize_skeleton_graph_to_svg(*args, **kwargs)


_vector_engines.vectorize_skeleton_to_svg = _lazy_graph_centerline

# AA-2: attach versioned metadata without changing the analyzer's heuristic
# recommendation. AA-3 can provide a request-scoped precomputed report so the
# auto gate and the core pipeline do not analyze the same pixels twice.
from app import analyzer as _analyzer

if not getattr(_analyzer.analyze_image_from_mem, "__vektoryum_contract_wrapped__", False):
    _original_analyze_image_from_mem = _analyzer.analyze_image_from_mem

    @wraps(_original_analyze_image_from_mem)
    def _analyze_image_from_mem_with_contract(image):
        from app.analyzer_decision_gate import consume_precomputed_analysis  # noqa: PLC0415

        precomputed = consume_precomputed_analysis()
        if precomputed is not None:
            return precomputed
        report = _original_analyze_image_from_mem(image)
        from app.analyzer_contracts import attach_analyzer_contract  # noqa: PLC0415

        return attach_analyzer_contract(report, image)

    _analyze_image_from_mem_with_contract.__vektoryum_contract_wrapped__ = True
    _analyzer.analyze_image_from_mem = _analyze_image_from_mem_with_contract


# AA-3: verify auto metadata before it selects preprocessing. The mature core
# pipeline receives an explicit verified mode, while manual requests pass through
# unchanged. ContextVar handoff keeps concurrent requests isolated.
from app import pipeline as _pipeline

if not getattr(_pipeline.run_pipeline, "__vektoryum_auto_gate_wrapped__", False):
    _original_run_pipeline = _pipeline.run_pipeline
    _analysis_entry = _analyzer.analyze_image_from_mem

    @wraps(_original_run_pipeline)
    def _run_pipeline_with_auto_gate(
        image,
        original_path,
        trace_mode,
        job_dir,
        refine=True,
        edge_cleanup=True,
    ):
        from app.analyzer_decision_gate import (  # noqa: PLC0415
            bind_precomputed_analysis,
            decide_trace_mode,
            reset_precomputed_analysis,
        )

        if trace_mode != "auto":
            result = _original_run_pipeline(
                image,
                original_path,
                trace_mode,
                job_dir,
                refine=refine,
                edge_cleanup=edge_cleanup,
            )
            decision = decide_trace_mode(result["analysis"], image, trace_mode)
            result["analysis"]["auto_decision"] = decision
            result["auto_decision"] = decision
            return result

        analysis = _analysis_entry(image)
        decision = decide_trace_mode(analysis, image, "auto")
        analysis["auto_decision"] = decision
        token = bind_precomputed_analysis(analysis)
        try:
            result = _original_run_pipeline(
                image,
                original_path,
                decision["execution_mode"],
                job_dir,
                refine=refine,
                edge_cleanup=edge_cleanup,
            )
        finally:
            reset_precomputed_analysis(token)

        result["analysis"] = analysis
        result["mode_used"] = decision["execution_mode"]
        result["auto_decision"] = decision
        result["mode_warning"] = (
            "Automatic mode confidence requires review; color-preserving mode was used."
            if decision["status"] == "needs_review"
            else None
        )
        if decision["status"] == "needs_review":
            from app.analyzer_runtime import register_job_auto_decision  # noqa: PLC0415

            register_job_auto_decision(job_dir, decision)
        return result

    _run_pipeline_with_auto_gate.__vektoryum_auto_gate_wrapped__ = True
    _pipeline.run_pipeline = _run_pipeline_with_auto_gate


# The final exported SVG is evaluated after the pipeline in another thread-pool
# call. Patch the evaluator once so an abstained auto request cannot be reported
# production-ready. Candidate/journal evaluations are unaffected because the
# registry only matches the exact <job_id>.svg export filename.
from app import final_artifact_evaluator as _final_artifact_evaluator

if not getattr(
    _final_artifact_evaluator.evaluate_final_svg,
    "__vektoryum_auto_review_wrapped__",
    False,
):
    _original_evaluate_final_svg = _final_artifact_evaluator.evaluate_final_svg

    @wraps(_original_evaluate_final_svg)
    def _evaluate_final_svg_with_auto_review(svg_path, *args, **kwargs):
        report = _original_evaluate_final_svg(svg_path, *args, **kwargs)
        from app.analyzer_runtime import take_final_svg_auto_decision  # noqa: PLC0415

        decision = take_final_svg_auto_decision(svg_path)
        if isinstance(decision, dict) and decision.get("status") == "needs_review":
            if report.verdict == "production_ready":
                report.verdict = "needs_review"
            if "analyzer_auto_review" not in report.soft_warning_codes:
                report.soft_warning_codes.append("analyzer_auto_review")
                report.soft_warnings.append(
                    "Automatic mode confidence requires review; color-preserving mode was used."
                )
        return report

    _evaluate_final_svg_with_auto_review.__vektoryum_auto_review_wrapped__ = True
    _final_artifact_evaluator.evaluate_final_svg = _evaluate_final_svg_with_auto_review


del _final_artifact_evaluator
del _pipeline
del _analyzer
del _vector_engines
