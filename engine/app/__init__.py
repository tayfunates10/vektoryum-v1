"""Application package runtime bindings."""
from __future__ import annotations

from functools import wraps

from app import vector_engines as _vector_engines


def _lazy_graph_centerline(*args, **kwargs):
    from app.centerline_svg import vectorize_skeleton_graph_to_svg  # noqa: PLC0415

    return vectorize_skeleton_graph_to_svg(*args, **kwargs)


_vector_engines.vectorize_skeleton_to_svg = _lazy_graph_centerline

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


# Alpha is staged before pipeline imports preprocess_for_mode by value. The
# selected SVG receives the single source-alpha truth only after every mutator.
from app import preprocess as _preprocess
from app.alpha_preprocess import wrap_gradient_vectorizer, wrap_preprocess_for_mode

_preprocess.preprocess_for_mode = wrap_preprocess_for_mode(
    _preprocess.preprocess_for_mode
)

from app import gradient_vectorize as _gradient_vectorize

_gradient_vectorize.vectorize_with_gradients = wrap_gradient_vectorizer(
    _gradient_vectorize.vectorize_with_gradients
)


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
            "Automatic mode confidence requires review."
            if decision["status"] == "needs_review"
            else None
        )
        if decision["status"] == "needs_review":
            from app.analyzer_runtime import register_job_auto_decision  # noqa: PLC0415

            register_job_auto_decision(job_dir, decision)
        return result

    _run_pipeline_with_auto_gate.__vektoryum_auto_gate_wrapped__ = True
    _pipeline.run_pipeline = _run_pipeline_with_auto_gate


from app import alpha_svg_mask as _alpha_svg_mask
from app.alpha_candidate_identity import (
    wrap_run_pipeline_preserving_candidate_identity,
)
from app import alpha_candidate_knockout as _alpha_candidate_knockout
from app.alpha_candidate_knockout import (
    make_candidate_geometry_knockout_fallback,
)
from app import alpha_candidate_support as _alpha_candidate_support
from app.alpha_candidate_support_compact import (
    build_compact_native_use_reconstruction_tree,
)
from app.alpha_candidate_validation import (
    validate_alpha_reconstruction_contract,
)

# Keep transaction code stable while binding the renderer-native compact encoder
# and the dual alpha-evaluator contract. Visual/topology/seam regressions remain
# fail-closed in the following real TransformJournal parent-delta stage.
_alpha_candidate_support._build_native_use_reconstruction_tree = (
    build_compact_native_use_reconstruction_tree
)
_alpha_candidate_knockout._validate_reconstruction = (
    validate_alpha_reconstruction_contract
)
_alpha_candidate_support._validate_reconstruction = (
    validate_alpha_reconstruction_contract
)
from app.alpha_candidate_support import (  # noqa: E402
    make_candidate_support_reconstruction_fallback,
)
from app.alpha_mask_adaptive import (
    make_adaptive_apply_source_alpha_mask,
    make_rect_fidelity_fallback,
)
from app.alpha_mask_budget import wrap_apply_source_alpha_mask

# The preflight computes the unchanged TransformJournal path/node/byte budgets.
# Rect encoding remains the default; compact paths are authorized when rect bytes
# do not fit, or after the exact rect mask render fails alpha fidelity and the same
# unchanged compact budgets independently admit a contour retry. An opaque trace
# canvas is then removed only when renderer probes prove its identity. If clipping
# the unchanged candidate paint still lacks source-edge support, the smallest
# same-color stroke painted behind the existing fill is measured on the renderer's
# native grid. Repeated rectangles use compact references so the unchanged byte
# budget remains authoritative. Rollback wraps every rejected representation.
_alpha_svg_mask.apply_source_alpha_mask = make_candidate_support_reconstruction_fallback(
    make_candidate_geometry_knockout_fallback(
        make_rect_fidelity_fallback(
            wrap_apply_source_alpha_mask(
                make_adaptive_apply_source_alpha_mask(
                    _alpha_svg_mask.apply_source_alpha_mask
                )
            )
        )
    )
)
_pipeline.run_pipeline = _alpha_svg_mask.wrap_run_pipeline_with_alpha_mask(
    _pipeline.run_pipeline
)
# Source-alpha finalization is a journaled artifact transform, not a new vector
# engine candidate. Preserve the selected candidate identity for API/regression
# consumers while the final SVG path and journal SHA point at the masked artifact.
_pipeline.run_pipeline = wrap_run_pipeline_preserving_candidate_identity(
    _pipeline.run_pipeline
)


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
                report.soft_warnings.append("Automatic mode confidence requires review.")
        return report

    _evaluate_final_svg_with_auto_review.__vektoryum_auto_review_wrapped__ = True
    _final_artifact_evaluator.evaluate_final_svg = _evaluate_final_svg_with_auto_review


del _final_artifact_evaluator
del _alpha_candidate_support
del _alpha_candidate_knockout
del _alpha_svg_mask
del _pipeline
del _gradient_vectorize
del _preprocess
del _analyzer
del _vector_engines
