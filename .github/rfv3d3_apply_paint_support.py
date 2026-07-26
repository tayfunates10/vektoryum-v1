from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def update_painter() -> None:
    path = Path("engine/app/alpha_candidate_painter.py")
    text = path.read_text()

    helper_anchor = '''_ALPHA_PLANE_FAILURE_CODES = {"alpha_iou_below_min", "alpha_mae_above_max"}\n\n\ndef _painter_loops(\n'''
    helper = '''_ALPHA_PLANE_FAILURE_CODES = {"alpha_iou_below_min", "alpha_mae_above_max"}\n\n\ndef _source_alpha_paint_support_proof(\n    root: ET.Element,\n    comparison_canvas: ET.Element | None,\n    source_alpha: np.ndarray,\n    width: int,\n    height: int,\n) -> dict[str, Any]:\n    """Prove every source-alpha component intersects real non-canvas paint.\n\n    A full-canvas comparison background can reproduce an arbitrary source alpha\n    mask even when the selected artwork is elsewhere.  It is therefore excluded\n    from support.  The established alpha>=128 binary boundary and exact component\n    intersection are used; no quality tolerance or guessed distance is introduced.\n    """\n    from app.alpha_candidate_knockout import (  # noqa: PLC0415\n        _RENDERABLE_TAGS,\n        _local_name,\n        _probe_root,\n        _render_root,\n    )\n\n    plane = np.asarray(source_alpha, dtype=np.uint8)\n    binary = (plane >= 128).astype(np.uint8)\n    component_total, labels, _stats, _centroids = cv2.connectedComponentsWithStats(\n        binary, connectivity=8\n    )\n    source_component_count = int(component_total - 1)\n    if source_component_count <= 0:\n        raise RuntimeError(\n            "source_alpha_candidate_painter_paint_support_unmeasured:"\n            "no_alpha_component_at_128"\n        )\n\n    support = np.zeros((height, width), dtype=bool)\n    renderable_count = 0\n    for child in list(root):\n        if child is comparison_canvas:\n            continue\n        if _local_name(str(child.tag)).lower() not in _RENDERABLE_TAGS:\n            continue\n        rendered = _render_root(_probe_root(root, child), width, height)\n        if rendered is None:\n            raise RuntimeError(\n                "source_alpha_candidate_painter_paint_support_unmeasured:"\n                "non_canvas_render_failed"\n            )\n        renderable_count += 1\n        support |= np.asarray(rendered[:, :, 3], dtype=np.uint8) >= 128\n\n    unsupported = 0\n    for component_id in range(1, component_total):\n        component = labels == component_id\n        if not bool(np.any(component & support)):\n            unsupported += 1\n\n    proof = {\n        "source_alpha_paint_support_verified": unsupported == 0,\n        "source_alpha_component_count": source_component_count,\n        "source_alpha_unsupported_component_count": int(unsupported),\n        "source_alpha_non_canvas_renderable_count": int(renderable_count),\n        "source_alpha_paint_support_binary_alpha": 128,\n        "source_alpha_paint_support_authority": (\n            "exact_component_overlap_with_non_canvas_rendered_paint"\n        ),\n    }\n    if unsupported:\n        raise RuntimeError(\n            "source_alpha_candidate_painter_paint_support_missing:"\n            f"{unsupported}/{source_component_count}"\n        )\n    return proof\n\n\ndef _painter_loops(\n'''
    text = replace_once(text, helper_anchor, helper, label="paint-support helper anchor")

    classify_anchor = '''    if background_status == "ambiguous":\n        raise RuntimeError("source_alpha_candidate_painter_background_ambiguous")\n    limits = _journal_limits(original_root, before_size)\n'''
    classify_replacement = '''    if background_status == "ambiguous":\n        raise RuntimeError("source_alpha_candidate_painter_background_ambiguous")\n    paint_support_proof = _source_alpha_paint_support_proof(\n        original_root, canvas, grid_alpha, grid_width, grid_height\n    )\n    limits = _journal_limits(original_root, before_size)\n'''
    text = replace_once(
        text,
        classify_anchor,
        classify_replacement,
        label="paint-support preflight anchor",
    )

    first_assessment = '''                assessment = _assess_painter_candidate(\n                    probe_temp,\n                    source_rgba_full,\n                    grid_alpha,\n                    mode,\n                    parent_counts,\n                    transaction_id=txn,\n                    parent_artwork_fingerprint=parent_artwork_fp,\n                    parent_alpha_report=parent_alpha_report,\n                )\n                for field in (\n'''
    first_replacement = '''                assessment = _assess_painter_candidate(\n                    probe_temp,\n                    source_rgba_full,\n                    grid_alpha,\n                    mode,\n                    parent_counts,\n                    transaction_id=txn,\n                    parent_artwork_fingerprint=parent_artwork_fp,\n                    parent_alpha_report=parent_alpha_report,\n                )\n                if isinstance(assessment.get("report"), dict):\n                    assessment["report"].update(paint_support_proof)\n                for field in (\n'''
    text = replace_once(
        text,
        first_assessment,
        first_replacement,
        label="standard painter report proof",
    )

    second_assessment = '''            assessment = _assess_painter_candidate(\n                probe_temp,\n                source_rgba_full,\n                grid_alpha,\n                mode,\n                parent_counts,\n                transaction_id=txn,\n                parent_artwork_fingerprint=parent_artwork_fp,\n                parent_alpha_report=parent_alpha_report,\n            )\n            for field in (\n'''
    second_replacement = '''            assessment = _assess_painter_candidate(\n                probe_temp,\n                source_rgba_full,\n                grid_alpha,\n                mode,\n                parent_counts,\n                transaction_id=txn,\n                parent_artwork_fingerprint=parent_artwork_fp,\n                parent_alpha_report=parent_alpha_report,\n            )\n            if isinstance(assessment.get("report"), dict):\n                assessment["report"].update(paint_support_proof)\n            for field in (\n'''
    text = replace_once(
        text,
        second_assessment,
        second_replacement,
        label="paint-deficit report proof",
    )

    path.write_text(text)


def update_transform_journal() -> None:
    path = Path("engine/app/transform_journal.py")
    text = path.read_text()
    old = '''        if (\n            transform_report.get("schema")\n            == "rfv3d2-candidate-painter-reconstruction-v1"\n            and transform_report.get("trace_rgb_bytes_preserved") is not True\n        ):\n            return False\n'''
    new = '''        painter_schema = (\n            transform_report.get("schema")\n            == "rfv3d2-candidate-painter-reconstruction-v1"\n        )\n        if painter_schema:\n            if transform_report.get("trace_rgb_bytes_preserved") is not True:\n                return False\n            if transform_report.get("source_alpha_paint_support_verified") is not True:\n                return False\n            if transform_report.get("source_alpha_paint_support_authority") != (\n                "exact_component_overlap_with_non_canvas_rendered_paint"\n            ):\n                return False\n            try:\n                component_count = int(\n                    transform_report["source_alpha_component_count"]\n                )\n                unsupported_count = int(\n                    transform_report["source_alpha_unsupported_component_count"]\n                )\n                renderable_count = int(\n                    transform_report["source_alpha_non_canvas_renderable_count"]\n                )\n            except (KeyError, TypeError, ValueError):\n                return False\n            if (\n                component_count <= 0\n                or unsupported_count != 0\n                or renderable_count <= 0\n            ):\n                return False\n'''
    path.write_text(replace_once(text, old, new, label="journal painter proof"))


def update_source_alpha_tests() -> None:
    path = Path("engine/test_rfv3d3_source_alpha_journal.py")
    text = path.read_text()
    old = '''    if schema == "rfv3d2-candidate-painter-reconstruction-v1":\n        report["trace_rgb_bytes_preserved"] = True\n    return report\n'''
    new = '''    if schema == "rfv3d2-candidate-painter-reconstruction-v1":\n        report.update({\n            "trace_rgb_bytes_preserved": True,\n            "source_alpha_paint_support_verified": True,\n            "source_alpha_component_count": 1,\n            "source_alpha_unsupported_component_count": 0,\n            "source_alpha_non_canvas_renderable_count": 1,\n            "source_alpha_paint_support_binary_alpha": 128,\n            "source_alpha_paint_support_authority": (\n                "exact_component_overlap_with_non_canvas_rendered_paint"\n            ),\n        })\n    return report\n'''
    text = replace_once(text, old, new, label="test report painter proof")

    anchor = '''def test_unverified_source_alpha_keeps_existing_journal_vetoes(\n'''
    test = '''@pytest.mark.parametrize("proof_value", [None, False])\ndef test_painter_support_proof_cannot_be_omitted(\n    tmp_path: Path,\n    monkeypatch: pytest.MonkeyPatch,\n    proof_value: bool | None,\n) -> None:\n    import app.transform_journal as transform_journal\n    from app.transform_journal import TransformJournal\n\n    parent = tmp_path / "parent.svg"\n    candidate = tmp_path / "candidate.svg"\n    parent.write_bytes(b"<svg>parent</svg>")\n    candidate.write_bytes(b"<svg>candidate</svg>")\n\n    def measure(data: bytes, _source: np.ndarray, **_kwargs):\n        return _metric(data, candidate=b"candidate" in data)\n\n    monkeypatch.setattr(transform_journal, "_measure_svg_bytes", measure)\n    report = _report(schema="rfv3d2-candidate-painter-reconstruction-v1")\n    if proof_value is None:\n        report.pop("source_alpha_paint_support_verified")\n    else:\n        report["source_alpha_paint_support_verified"] = proof_value\n    journal = TransformJournal(parent, np.zeros((8, 8, 3), dtype=np.uint8))\n    accepted, stage = journal.consider_candidate(\n        "source_alpha_painter_candidate",\n        parent,\n        candidate,\n        transform_report=report,\n    )\n    assert accepted == parent\n    assert stage["source_alpha_contract_verified"] is False\n    assert "seam_regression" in stage["reason_codes"]\n\n\ndef test_unverified_source_alpha_keeps_existing_journal_vetoes(\n'''
    text = replace_once(text, anchor, test, label="painter support fail-closed test")
    path.write_text(text)


def update_rfv3d2_workflow() -> None:
    path = Path(".github/workflows/real-world-fidelity-rfv3d2-alpha-mask-fix.yml")
    text = path.read_text()

    text = replace_once(
        text,
        '      - "engine/app/alpha_candidate_painter.py"\n',
        '      - "engine/app/alpha_candidate_painter.py"\n      - "engine/app/transform_journal.py"\n',
        label="RFV3D2 trigger production path",
    )
    text = replace_once(
        text,
        '      - "engine/test_alpha_candidate_painter.py"\n',
        '      - "engine/test_alpha_candidate_painter.py"\n      - "engine/test_rfv3d3_source_alpha_journal.py"\n',
        label="RFV3D2 trigger test path",
    )
    text = replace_once(
        text,
        '          engine/app/alpha_candidate_painter.py\n',
        '          engine/app/alpha_candidate_painter.py\n          engine/app/transform_journal.py\n',
        label="RFV3D2 compile production path",
    )
    text = replace_once(
        text,
        '          engine/test_alpha_candidate_painter.py\n',
        '          engine/test_alpha_candidate_painter.py\n          engine/test_rfv3d3_source_alpha_journal.py\n',
        label="RFV3D2 compile test path",
    )

    step_anchor = '''          cat alpha-mask-tests.log\n          exit "${status}"\n\n      - name: Upload alpha-mask test diagnostics\n'''
    step_replacement = '''          cat alpha-mask-tests.log\n          exit "${status}"\n\n      - name: Run RFV-3D3 source-alpha journal contract\n        run: >-\n          pytest -q\n          engine/test_rfv3d3_source_alpha_journal.py\n          engine/test_alpha_candidate_validation.py\n\n      - name: Upload alpha-mask test diagnostics\n'''
    text = replace_once(
        text,
        step_anchor,
        step_replacement,
        label="RFV3D2 source-alpha test step",
    )

    text = replace_once(
        text,
        '|engine/app/alpha_candidate_painter.py|engine/app/alpha_candidate_paint_deficit.py|',
        '|engine/app/alpha_candidate_painter.py|engine/app/transform_journal.py|engine/app/alpha_candidate_paint_deficit.py|',
        label="RFV3D2 production scope allowlist",
    )
    text = replace_once(
        text,
        '|engine/test_alpha_candidate_painter.py|engine/test_alpha_parent_selection.py|',
        '|engine/test_alpha_candidate_painter.py|engine/test_rfv3d3_source_alpha_journal.py|engine/test_alpha_parent_selection.py|',
        label="RFV3D2 test scope allowlist",
    )
    text = replace_once(
        text,
        '          engine/app/source_truth.py\n          engine/app/transform_journal.py\n          engine/benchmark\n',
        '          engine/app/source_truth.py\n          engine/benchmark\n',
        label="RFV3D2 forbidden journal exception",
    )
    path.write_text(text)


if __name__ == "__main__":
    update_painter()
    update_transform_journal()
    update_source_alpha_tests()
    update_rfv3d2_workflow()
