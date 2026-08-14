from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    assert count == 1, f"{label}: expected 1 anchor, found {count}"
    return text.replace(old, new, 1)


path = Path("engine/app/alpha_candidate_paint_deficit.py")
text = path.read_text(encoding="utf-8")
text = replace_once(text, "    ROLE_ARTWORK_CONTAINER,\n    ROLE_MASK_APPLICATION,\n", "    ROLE_ARTWORK_CONTAINER,\n    ROLE_CANVAS_UNDERPAINT,\n    ROLE_MASK_APPLICATION,\n", "identity import")
fn_start = text.index("def build_paint_deficit_reconstruction_tree(\n")
prefix, fn = text[:fn_start], text[fn_start:]
fn = replace_once(fn, '    mask_encoding: str = "polygon",\n    levels: int = _ALPHA_LEVELS,\n) -> tuple[ET.Element, dict[str, Any]]:\n', '    mask_encoding: str = "polygon",\n    levels: int = _ALPHA_LEVELS,\n    retain_canvas_under_mask: bool = False,\n) -> tuple[ET.Element, dict[str, Any]]:\n', "builder signature")
fn = replace_once(fn, "        if child is target_canvas:\n            continue\n        _strip_content_alpha(child)\n        paint.append(child)\n        artwork_count += 1\n", "        if child is target_canvas:\n            if not retain_canvas_under_mask:\n                continue\n            tag_transform_node(child, ROLE_CANVAS_UNDERPAINT, transaction_id)\n            child.set(\n                \"data-vektoryum-candidate-geometry-underpaint\",\n                \"comparison-canvas-v1\",\n            )\n            _strip_content_alpha(child)\n            paint.append(child)\n            continue\n        _strip_content_alpha(child)\n        paint.append(child)\n        artwork_count += 1\n", "builder canvas knockout")
fn = replace_once(fn, '        "comparison_canvas_knocked_out": bool(target_canvas is not None),\n        "comparison_canvas_retained_under_mask": False,\n', '        "comparison_canvas_knocked_out": bool(target_canvas is not None and not retain_canvas_under_mask),\n        "comparison_canvas_retained_under_mask": bool(target_canvas is not None and retain_canvas_under_mask),\n', "builder report flags")
path.write_text(prefix + fn, encoding="utf-8")

path = Path("engine/app/alpha_candidate_painter.py")
text = path.read_text(encoding="utf-8")
helper_marker = "\n\ndef apply_candidate_painter_reconstruction(\n"
assert text.count(helper_marker) == 1
helper = r'''


def _paint_deficit_underpaint_retry_eligible(
    paint_deficit_attempts: list[dict[str, Any]],
    canvas_present: bool,
) -> bool:
    """Route only legacy native-alpha support loss to terminal underpaint retry."""
    if not canvas_present:
        return False
    relevant = [
        entry for entry in paint_deficit_attempts
        if entry.get("encoding_family") == "paint_deficit"
        and entry.get("exact_or_quantized") == "paint_deficit"
    ]
    if not relevant or any(entry.get("status") == "accepted" for entry in relevant):
        return False
    return any(
        entry.get("status") == "native_alpha_rejected"
        and entry.get("validation_stage") == "native_alpha"
        for entry in relevant
    )
'''
text = text.replace(helper_marker, helper + helper_marker, 1)
apply_start = text.index("def apply_candidate_painter_reconstruction(\n")
apply_prefix, apply_fn = text[:apply_start], text[apply_start:]
pd_start = apply_fn.index("    def _evaluate_paint_deficit() -> list[Any] | None:\n")
pd_end = apply_fn.index("\n    try:\n", pd_start)
before_pd, pd, after_pd = apply_fn[:pd_start], apply_fn[pd_start:pd_end], apply_fn[pd_end:]
pd = replace_once(pd, "        def _try(\n            mask_encoding: str,\n            label: str,\n            levels: int = _ALPHA_LEVELS,\n        ) -> list[Any] | None:\n", "        def _try(\n            mask_encoding: str,\n            label: str,\n            levels: int = _ALPHA_LEVELS,\n            retain_canvas_under_mask: bool = False,\n        ) -> list[Any] | None:\n", "try signature")
pd = replace_once(pd, '                "encoding_family": "paint_deficit",\n                "exact_or_quantized": "paint_deficit",\n                "source_alpha_level_count": int(source_level_count),\n', '                "encoding_family": "paint_deficit",\n                "exact_or_quantized": "paint_deficit",\n                "paint_support_mode": ("canvas_underpaint" if retain_canvas_under_mask else "deficit_only"),\n                "source_alpha_level_count": int(source_level_count),\n', "ledger support mode")
pd = replace_once(pd, "                        mask_encoding=mask_encoding,\n                        levels=levels,\n                    )\n", "                        mask_encoding=mask_encoding,\n                        levels=levels,\n                        retain_canvas_under_mask=retain_canvas_under_mask,\n                    )\n", "builder call")
pd = replace_once(pd, "            for stat_key in (\n                \"paint_deficit_pixel_count\",\n                \"source_component_count\",\n                \"anchored_source_component_count\",\n                \"detached_source_component_count\",\n            ):\n                if stat_key in probe_geometry:\n                    entry[stat_key] = int(probe_geometry[stat_key])\n", "            for stat_key in (\n                \"paint_deficit_pixel_count\",\n                \"source_component_count\",\n                \"anchored_source_component_count\",\n                \"detached_source_component_count\",\n            ):\n                if stat_key in probe_geometry:\n                    entry[stat_key] = int(probe_geometry[stat_key])\n            for flag_key in (\n                \"comparison_canvas_knocked_out\",\n                \"comparison_canvas_retained_under_mask\",\n            ):\n                if flag_key in probe_geometry:\n                    entry[flag_key] = bool(probe_geometry[flag_key])\n", "ledger flags")
pd = replace_once(pd, '        winner = _try("polygon", "paint-deficit-q24")\n', '        paint_deficit_attempt_start = len(attempts)\n        winner = _try("polygon", "paint-deficit-q24")\n', "attempt start")
return_marker = "        return winner\n"
return_pos = pd.rfind(return_marker)
assert return_pos >= 0
terminal = r'''
        legacy_paint_attempts = attempts[paint_deficit_attempt_start:]
        if winner is None and _paint_deficit_underpaint_retry_eligible(
            legacy_paint_attempts, canvas is not None
        ):
            winner = _try("polygon", "paint-deficit-q24-underpaint", retain_canvas_under_mask=True)
            if winner is None:
                winner = _try("cumulative", "paint-deficit-cumulative-underpaint", retain_canvas_under_mask=True)
            if winner is None:
                for coarse_levels in _PAINT_DEFICIT_CUMULATIVE_LEVELS:
                    winner = _try(
                        "cumulative",
                        f"paint-deficit-cumulative-q{coarse_levels}-underpaint",
                        levels=coarse_levels,
                        retain_canvas_under_mask=True,
                    )
                    if winner is not None:
                        break
'''
pd = pd[:return_pos] + terminal + pd[return_pos:]
path.write_text(apply_prefix + before_pd + pd + after_pd, encoding="utf-8")

Path("engine/test_alpha_painter_public05_support_fallback.py").write_text(r'''from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
import numpy as np
from app.alpha_candidate_knockout import _render_root
from app.alpha_candidate_paint_deficit import build_paint_deficit_reconstruction_tree
from app.alpha_candidate_painter import _paint_deficit_underpaint_retry_eligible

SVG_NS = "http://www.w3.org/2000/svg"
def qname(name: str) -> str: return f"{{{SVG_NS}}}{name}"

def fixture():
    root = ET.Element(qname("svg"), {"viewBox":"0 0 4 4","width":"4","height":"4"})
    canvas = ET.SubElement(root,qname("rect"),{"x":"0","y":"0","width":"4","height":"4","fill":"white"})
    ET.SubElement(root,qname("rect"),{"x":"0","y":"0","width":"2","height":"4","fill":"black"})
    ET.SubElement(root,qname("rect"),{"x":"2","y":"0","width":"1","height":"4","fill":"white"})
    source=np.full((4,4,4),255,dtype=np.uint8); source[:,:3,:3]=0
    return root,canvas,source

class BuilderTests(unittest.TestCase):
    def test_legacy_default_still_knocks_out_canvas(self):
        root,canvas,source=fixture(); tree,report=build_paint_deficit_reconstruction_tree(root,canvas,source,"legacy",mask_encoding="cumulative")
        rendered=_render_root(tree,4,4); self.assertIsNotNone(rendered); assert rendered is not None
        self.assertTrue(report["comparison_canvas_knocked_out"]); self.assertFalse(report["comparison_canvas_retained_under_mask"])
        self.assertTrue(np.all(rendered[:,3,3]==0))
    def test_terminal_underpaint_restores_near_white_alpha_support(self):
        root,canvas,source=fixture(); tree,report=build_paint_deficit_reconstruction_tree(root,canvas,source,"underpaint",mask_encoding="cumulative",retain_canvas_under_mask=True)
        rendered=_render_root(tree,4,4); self.assertIsNotNone(rendered); assert rendered is not None
        self.assertFalse(report["comparison_canvas_knocked_out"]); self.assertTrue(report["comparison_canvas_retained_under_mask"])
        self.assertTrue(np.all(rendered[:,:,3]==255)); self.assertTrue(np.all(rendered[:,3,:3]>240))

class GateTests(unittest.TestCase):
    def entry(self,status,stage=None): return {"encoding_family":"paint_deficit","exact_or_quantized":"paint_deficit","status":status,"validation_stage":stage}
    def test_native_alpha_support_failure_is_eligible(self): self.assertTrue(_paint_deficit_underpaint_retry_eligible([self.entry("byte_rejected"),self.entry("native_alpha_rejected","native_alpha")],True))
    def test_byte_only_public15_class_does_not_trigger(self): self.assertFalse(_paint_deficit_underpaint_retry_eligible([self.entry("byte_rejected"),self.entry("byte_rejected")],True))
    def test_missing_canvas_does_not_trigger(self): self.assertFalse(_paint_deficit_underpaint_retry_eligible([self.entry("native_alpha_rejected","native_alpha")],False))
    def test_existing_acceptance_never_retries(self): self.assertFalse(_paint_deficit_underpaint_retry_eligible([self.entry("native_alpha_rejected","native_alpha"),self.entry("accepted","accepted")],True))
    def test_evaluator_only_failure_does_not_trigger(self): self.assertFalse(_paint_deficit_underpaint_retry_eligible([self.entry("evaluator_rejected","evaluator")],True))
''', encoding="utf-8")
