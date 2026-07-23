from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

SOURCE_COMMIT = "8eb30de2c7256abe36fedaaddcb2d9f9655e4faf"
SOURCE_WORKFLOW = ".github/workflows/faz-3d4-paint-deficit-production.yml"


def _source_workflow() -> str:
    return subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_WORKFLOW}"],
        text=True,
    )


def _extract_cat(workflow: str, path: str) -> str:
    marker = f"cat > {path} <<'PY'\n"
    if marker not in workflow:
        raise RuntimeError(f"missing source heredoc: {path}")
    body = workflow.split(marker, 1)[1].split("\n          PY", 1)[0]
    return textwrap.dedent(body)


def _materialize_module_and_test(workflow: str) -> None:
    module_path = Path("engine/app/alpha_candidate_paint_deficit.py")
    test_path = Path("engine/test_alpha_painter_paint_deficit.py")
    module_path.write_text(_extract_cat(workflow, str(module_path)), encoding="utf-8")

    test_text = _extract_cat(workflow, str(test_path))
    old = '''tree2, report2 = build_paint_deficit_reconstruction_tree(
                  copy.deepcopy(root), list(copy.deepcopy(root))[0], source, "txn-fixed"
              )'''
    new = '''root2 = copy.deepcopy(root)
              canvas2 = list(root2)[0]
              tree2, report2 = build_paint_deficit_reconstruction_tree(
                  root2, canvas2, source, "txn-fixed"
              )'''
    if old not in test_text:
        raise RuntimeError("determinism test repair anchor missing")
    test_path.write_text(test_text.replace(old, new, 1), encoding="utf-8")


def _paint_deficit_function() -> list[str]:
    return r'''
    def _evaluate_paint_deficit() -> list[Any] | None:
        from app.alpha_candidate_paint_deficit import (  # noqa: PLC0415
            build_paint_deficit_reconstruction_tree,
        )

        label = "paint-deficit-q24"
        txn = alpha_transaction_id(parent_sha256, source_alpha_sha256, mode, label)
        entry: dict[str, Any] = {
            "stroke_width": 0.0,
            "encoding_label": label,
            "encoding_family": "paint_deficit",
            "exact_or_quantized": "paint_deficit",
            "source_alpha_level_count": int(source_level_count),
            "encoded_alpha_level_count": 24,
            "actual_serialized_bytes": None,
            "byte_limit": int(byte_limit),
            "projected_path_count": int(parent_counts[0]),
            "actual_path_count": None,
            "path_limit": int(limits["path_limit"]),
            "projected_node_count": int(parent_counts[1]),
            "actual_node_count": None,
            "node_limit": int(limits["node_limit"]),
            "preflight_status": None,
            "validation_started": False,
            "validation_stage": None,
            "status": None,
            "exact_error_code": "",
            "native_alpha_iou": None,
            "native_alpha_mae": None,
            "bounded_alpha_iou": None,
            "bounded_alpha_mae": None,
            "evaluator_alpha_iou": None,
            "evaluator_alpha_mae": None,
            "artwork_fingerprint_match": None,
            "journal_gate_started": False,
            "journal_passed": None,
            "journal_reason_codes": [],
        }
        try:
            probe_root, probe_geometry = build_paint_deficit_reconstruction_tree(
                original_root, canvas, grid_rgba, txn
            )
        except RuntimeError as exc:
            entry["preflight_status"] = "not_constructed"
            entry["validation_stage"] = "paint_deficit_geometry"
            entry["status"] = "geometry_rejected"
            entry["exact_error_code"] = str(exc)
            attempts.append(entry)
            return None

        probe_temp = _write_tree_to_temp(probe_root, target)
        probe_size = int(probe_temp.stat().st_size)
        entry["actual_serialized_bytes"] = probe_size
        entry["encoded_alpha_level_count"] = int(
            probe_geometry.get("encoded_alpha_level_count", 24)
        )
        if probe_size > byte_limit:
            entry["preflight_status"] = "over_budget"
            entry["status"] = "byte_rejected"
            entry["exact_error_code"] = (
                "source_alpha_candidate_painter_byte_budget_rejected:"
                f"{label}:{probe_size}>{byte_limit}"
            )
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            return None

        entry["preflight_status"] = "within_budget"
        entry["validation_started"] = True
        parent_artwork_fp = artwork_fingerprint(
            original_root, txn, excluded_from_parent
        )
        assessment = _assess_painter_candidate(
            probe_temp,
            source_rgba_full,
            grid_alpha,
            mode,
            parent_counts,
            transaction_id=txn,
            parent_artwork_fingerprint=parent_artwork_fp,
        )
        for field in (
            "validation_stage",
            "status",
            "exact_error_code",
            "native_alpha_iou",
            "native_alpha_mae",
            "bounded_alpha_iou",
            "bounded_alpha_mae",
            "evaluator_alpha_iou",
            "evaluator_alpha_mae",
            "artwork_fingerprint_match",
            "actual_path_count",
            "actual_node_count",
        ):
            entry[field] = assessment[field]
        if assessment["status"] != "accepted":
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            return None

        entry["journal_gate_started"] = True
        journal_passed, journal_codes = _run_painter_geometry_journal(
            parent_journal_path,
            probe_temp,
            journal_source_rgb,
            journal_image_class,
            assessment["report"],
        )
        entry["journal_passed"] = bool(journal_passed)
        entry["journal_reason_codes"] = list(journal_codes)
        if not journal_passed:
            entry["status"] = "geometry_rejected"
            entry["validation_stage"] = "journal_geometry"
            entry["exact_error_code"] = (
                "source_alpha_candidate_painter_journal_geometry_rejected:"
                + ",".join(journal_codes)
            )
            attempts.append(entry)
            probe_temp.unlink(missing_ok=True)
            return None

        attempts.append(entry)
        return [
            probe_size,
            int(assessment["actual_path_count"] or 0),
            int(assessment["actual_node_count"] or 0),
            0,
            probe_temp,
            probe_geometry,
            dict(assessment["report"] or {}),
            label,
            0.0,
        ]
'''.strip("\n").splitlines()


def _integrate_painter() -> None:
    path = Path("engine/app/alpha_candidate_painter.py")
    lines = path.read_text(encoding="utf-8").splitlines()

    validated_quantized = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == 'or _validated("quantized")'
    )
    if not any('_validated("paint_deficit")' in line for line in lines):
        indent = lines[validated_quantized][
            : len(lines[validated_quantized]) - len(lines[validated_quantized].lstrip())
        ]
        lines.insert(
            validated_quantized + 1,
            indent + 'or _validated("paint_deficit")',
        )

    smallest_quantized = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == 'or _smallest_byte_rejected("quantized")'
    )
    if not any(
        '_smallest_byte_rejected("paint_deficit")' in line for line in lines
    ):
        indent = lines[smallest_quantized][
            : len(lines[smallest_quantized]) - len(lines[smallest_quantized].lstrip())
        ]
        lines.insert(
            smallest_quantized + 1,
            indent + 'or _smallest_byte_rejected("paint_deficit")',
        )

    phase_start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("    def _evaluate_phase(")
    )
    try_index = next(
        index
        for index in range(phase_start + 1, len(lines) - 1)
        if lines[index] == "    try:" and "Kademe 1" in lines[index + 1]
    )
    if not any(
        line.startswith("    def _evaluate_paint_deficit(") for line in lines
    ):
        lines[try_index:try_index] = _paint_deficit_function() + [""]

    quantized_call = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == "winner = _evaluate_phase(quantized_specs)"
    )
    nearby = lines[quantized_call + 1 : quantized_call + 4]
    if not any("_evaluate_paint_deficit()" in line for line in nearby):
        lines[quantized_call + 1 : quantized_call + 1] = [
            "        if winner is None:",
            "            winner = _evaluate_paint_deficit()",
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    workflow = _source_workflow()
    _materialize_module_and_test(workflow)
    _integrate_painter()


if __name__ == "__main__":
    main()
