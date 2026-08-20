from __future__ import annotations

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
