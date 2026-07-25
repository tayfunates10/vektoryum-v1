"""Application package runtime bindings."""
from __future__ import annotations

from functools import wraps

from app import vector_engines as _vector_engines


def _lazy_graph_centerline(*args, **kwargs):
    from app.centerline_svg import vectorize_skeleton_graph_to_svg

    return vectorize_skeleton_graph_to_svg(*args, **kwargs)


_vector_engines.vectorize_skeleton_to_svg = _lazy_graph_centerline

from app import analyzer as _analyzer

if not getattr(_analyzer.analyze_image_from_mem, "__vektoryum_contract_wrapped__", False):
    _original_analyze_image_from_mem = _analyzer.analyze_image_from_mem

    @wraps(_original_analyze_image_from_mem)
    def _analyze_image_from_mem_with_contract(image):
        from app.analyzer_decision_gate import consume_precomputed_analysis

        precomputed = consume_precomputed_analysis()
        if precomputed is not None:
            return precomputed
        report = _original_analyze_image_from_mem(image)
        from app.analyzer_contracts import attach_analyzer_contract

        return attach_analyzer_contract(report, image)

    _analyze_image_from_mem_with_contract.__vektoryum_contract_wrapped__ = True
    _analyzer.analyze_image_from_mem = _analyze_image_from_mem_with_contract


# Alpha is staged before pipeline imports preprocess_for_mode by value. Production
# calls are additionally wrapped by a legacy-first context: alpha-specific RGB
# staging is enabled only after a measured source-alpha rejection.
from app import preprocess as _preprocess
from app.alpha_preprocess import wrap_gradient_vectorizer, wrap_preprocess_for_mode
from app.alpha_pipeline_retry import (
    wrap_gradient_with_alpha_context,
    wrap_preprocess_with_alpha_context,
    wrap_run_pipeline_with_alpha_retry,
)

_preprocess.preprocess_for_mode = wrap_preprocess_for_mode(
    _preprocess.preprocess_for_mode
)
_preprocess.preprocess_for_mode = wrap_preprocess_with_alpha_context(
    _preprocess.preprocess_for_mode
)

from app import gradient_vectorize as _gradient_vectorize

_gradient_vectorize.vectorize_with_gradients = wrap_gradient_vectorizer(
    _gradient_vectorize.vectorize_with_gradients
)
_gradient_vectorize.vectorize_with_gradients = wrap_gradient_with_alpha_context(
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
        from app.analyzer_decision_gate import (
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
            from app.analyzer_runtime import register_job_auto_decision

            register_job_auto_decision(job_dir, decision)
        return result

    _run_pipeline_with_auto_gate.__vektoryum_auto_gate_wrapped__ = True
    _pipeline.run_pipeline = _run_pipeline_with_auto_gate


from app import alpha_svg_mask as _alpha_svg_mask
from app.alpha_candidate_direct import make_direct_element_alpha_first
from app.alpha_candidate_identity import (
    wrap_run_pipeline_preserving_candidate_identity,
)
from app.alpha_parent_selection import wrap_run_pipeline_selecting_alpha_parent
from app import alpha_candidate_knockout as _alpha_candidate_knockout
from app.alpha_candidate_knockout import (
    make_candidate_geometry_knockout_fallback,
)
from app import alpha_candidate_support as _alpha_candidate_support
from app.alpha_candidate_support_compact import (
    _compact_direct_artifact,
    _union_polygon_points,
    build_compact_native_use_reconstruction_tree,
    make_compact_primitive_alpha_first,
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
from app.alpha_candidate_support import (
    make_candidate_support_reconstruction_fallback,
)
from app.alpha_mask_adaptive import (
    make_adaptive_apply_source_alpha_mask,
    make_rect_fidelity_fallback,
)
from app.alpha_mask_budget import wrap_apply_source_alpha_mask


def _compact_inert_empty_path_paint(original_apply):
    """Compact inert and reusable alpha markup without changing owned geometry."""
    if getattr(original_apply, "__vektoryum_empty_path_paint_compacted__", False):
        return original_apply

    @wraps(original_apply)
    def compacted(svg_path, source_path, mode):
        import hashlib
        import os
        import re
        import xml.etree.ElementTree as ET
        from pathlib import Path

        from PIL import Image

        from app.alpha_candidate_knockout import _path_node_counts
        from app.alpha_preprocess import _rgba_from_source_at_size

        def local_name(element):
            return str(element.tag).rsplit("}", 1)[-1].split(":")[-1].lower()

        target = Path(svg_path)
        source = Path(source_path)
        parent_root = ET.fromstring(target.read_bytes())
        parent_counts = _path_node_counts(parent_root)
        report = original_apply(target, source, mode)
        accepted_bytes = target.read_bytes()
        try:
            root = ET.fromstring(accepted_bytes)
            removed_empty_paint = 0
            compacted_rectangles = 0
            reused_underlay_paths = 0
            shortened_mask_ids = 0
            stripped_metadata = 0

            for element in root.iter():
                if (
                    local_name(element) == "path"
                    and not str(element.get("d", "")).strip()
                    and "fill" in element.attrib
                ):
                    element.attrib.pop("fill", None)
                    removed_empty_paint += 1

            used_ids = {
                str(element.get("id"))
                for element in root.iter()
                if element.get("id")
            }
            source_mask = next(
                (
                    element
                    for element in root.iter()
                    if local_name(element) == "mask"
                    and element.get("id") == "vektoryum-source-alpha"
                ),
                None,
            )
            if source_mask is not None and "a" not in used_ids:
                old_id = str(source_mask.get("id"))
                source_mask.set("id", "a")
                old_reference = f"url(#{old_id})"
                for element in root.iter():
                    if element.get("mask") == old_reference:
                        element.set("mask", "url(#a)")
                used_ids.discard(old_id)
                used_ids.add("a")
                shortened_mask_ids = 1

            viewbox = root.get("viewBox") or root.get("viewbox")
            dimensions = None
            if source_mask is not None and viewbox:
                try:
                    values = [
                        float(value)
                        for value in re.split(r"[\s,]+", viewbox.strip())
                        if value
                    ]
                    if (
                        len(values) == 4
                        and abs(values[0]) <= 1e-9
                        and abs(values[1]) <= 1e-9
                        and abs(values[2] - round(values[2])) <= 1e-9
                        and abs(values[3] - round(values[3])) <= 1e-9
                    ):
                        dimensions = (int(round(values[2])), int(round(values[3])))
                except (TypeError, ValueError):
                    dimensions = None

            if dimensions is not None:
                width, height = dimensions
                namespace = "http://www.w3.org/2000/svg"
                qname = lambda name: f"{{{namespace}}}{name}"
                for group in list(source_mask.iter()):
                    children = list(group)
                    if (
                        len(children) < 8
                        or any(local_name(child) != "rect" for child in children)
                    ):
                        continue
                    try:
                        rectangles = [
                            tuple(
                                int(str(child.get(name, "0")))
                                for name in ("x", "y", "width", "height")
                            )
                            for child in children
                        ]
                    except ValueError:
                        continue
                    points = _union_polygon_points(rectangles, width, height)
                    if not points:
                        continue
                    for child in children:
                        group.remove(child)
                    ET.SubElement(
                        group,
                        qname("polygon"),
                        {"points": points, "fill-rule": "evenodd"},
                    )
                    group.attrib.pop("data-vektoryum-alpha-level", None)
                    compacted_rectangles += len(rectangles)

            parent_by_child = {
                child: parent
                for parent in root.iter()
                for child in list(parent)
            }
            underlays = [
                element
                for element in root.iter()
                if local_name(element) == "g"
                and element.get("data-vektoryum-alpha-coverage-underlay")
            ]
            allowed_underlay_extra = {
                "stroke",
                "stroke-width",
                "stroke-linejoin",
                "stroke-linecap",
            }
            for underlay in underlays:
                owner = parent_by_child.get(underlay)
                underlay_paths = list(underlay)
                if (
                    owner is None
                    or not underlay_paths
                    or any(local_name(child) != "path" for child in underlay_paths)
                ):
                    continue
                main_paths = [
                    child
                    for child in list(owner)
                    if child is not underlay and local_name(child) == "path"
                ]
                matches = []
                claimed = set()
                valid = True
                for underlay_path in underlay_paths:
                    extra_names = set(underlay_path.attrib) - {
                        "d",
                        "fill",
                        "fill-rule",
                    }
                    if not extra_names.issubset(allowed_underlay_extra):
                        valid = False
                        break
                    candidates = [
                        path
                        for path in main_paths
                        if path not in claimed
                        and path.get("d") == underlay_path.get("d")
                        and path.get("fill") == underlay_path.get("fill")
                        and path.get("fill-rule") == underlay_path.get("fill-rule")
                    ]
                    if len(candidates) != 1:
                        valid = False
                        break
                    match = candidates[0]
                    claimed.add(match)
                    matches.append((underlay_path, match))
                if not valid:
                    continue

                replacements = []
                for underlay_path, match in matches:
                    identifier = match.get("id")
                    if not identifier:
                        candidate_id = "b"
                        while candidate_id in used_ids:
                            candidate_id += "a"
                        identifier = candidate_id
                        match.set("id", identifier)
                        used_ids.add(identifier)
                    attributes = {"href": f"#{identifier}"}
                    for name in sorted(allowed_underlay_extra):
                        value = underlay_path.get(name)
                        if value is not None:
                            attributes[name] = value
                    replacements.append(
                        ET.Element(
                            "{http://www.w3.org/2000/svg}use",
                            attributes,
                        )
                    )
                for index, child in enumerate(underlay_paths):
                    underlay.remove(child)
                    underlay.insert(index, replacements[index])
                reused_underlay_paths += len(replacements)

            if (
                removed_empty_paint == 0
                and compacted_rectangles == 0
                and reused_underlay_paths == 0
                and shortened_mask_ids == 0
            ):
                return report

            for element in root.iter():
                if "data-vektoryum-source-alpha" in element.attrib:
                    element.attrib.pop("data-vektoryum-source-alpha", None)
                    stripped_metadata += 1
                if element.get("transform") == "translate(0 0) scale(1 1)":
                    element.attrib.pop("transform", None)
                    stripped_metadata += 1
                if element.text is not None and not element.text.strip():
                    element.text = None
                if element.tail is not None and not element.tail.strip():
                    element.tail = None
            if root.get("version") == "1.1":
                root.attrib.pop("version", None)
                stripped_metadata += 1

            ET.register_namespace("", "http://www.w3.org/2000/svg")
            candidate = ET.tostring(
                root,
                encoding="utf-8",
                xml_declaration=False,
                short_empty_elements=True,
            )
            if len(candidate) >= len(accepted_bytes):
                return report

            temporary = target.parent / f".{target.name}.inert-alpha-compact.svg"
            temporary.write_bytes(candidate)
            try:
                with Image.open(source) as source_image:
                    source_size = source_image.size
                source_full = _rgba_from_source_at_size(source, source_size)
                validation = validate_alpha_reconstruction_contract(
                    temporary,
                    source_full,
                    mode,
                    parent_counts,
                )
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        except Exception:
            target.write_bytes(accepted_bytes)
            return report

        result = dict(report)
        result.update(validation)
        result["after_byte_size"] = int(target.stat().st_size)
        result["after_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        result["empty_path_paint_attributes_removed"] = int(removed_empty_paint)
        result["dense_mask_rectangles_compacted"] = int(compacted_rectangles)
        result["underlay_paths_reused"] = int(reused_underlay_paths)
        result["source_mask_ids_shortened"] = int(shortened_mask_ids)
        result["inert_alpha_metadata_removed"] = int(stripped_metadata)
        result["empty_path_artifact_compacted"] = True
        return result

    compacted.__vektoryum_empty_path_paint_compacted__ = True
    return compacted


# Bind the compact primitive before the guarded rect/path adapter. The final
# compactors are deliberately outermost so a direct rejection followed by a clip
# fallback is still compacted and revalidated before publication.
_adaptive_alpha_builder = make_compact_primitive_alpha_first(
    make_adaptive_apply_source_alpha_mask(_alpha_svg_mask.apply_source_alpha_mask)
)
_guarded_alpha_builder = make_candidate_support_reconstruction_fallback(
    make_candidate_geometry_knockout_fallback(
        make_rect_fidelity_fallback(
            wrap_apply_source_alpha_mask(_adaptive_alpha_builder)
        )
    )
)
_alpha_svg_mask.apply_source_alpha_mask = _compact_inert_empty_path_paint(
    _compact_direct_artifact(
        make_direct_element_alpha_first(_guarded_alpha_builder)
    )
)

# For low-level source alpha, trial at most four already-produced parents through
# the real guarded alpha transform. Parent reselection is active only during the
# bounded remediation pass; the legacy-first pass keeps the original winner.
_pipeline.run_pipeline = wrap_run_pipeline_selecting_alpha_parent(
    _pipeline.run_pipeline
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
# Outermost orchestration: return the untouched legacy result whenever it already
# satisfies alpha. Only a source-alpha fail-closed rejection authorizes one retry
# with alpha-specific preprocessing and alternate-parent selection enabled.
_pipeline.run_pipeline = wrap_run_pipeline_with_alpha_retry(
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
        from app.analyzer_runtime import take_final_svg_auto_decision

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
