from __future__ import annotations
import unittest
import numpy as np
from engine.regression.residual_topology_metrics import extend_near_zero_contract, measure_label_topology

class TopologyHardStopTests(unittest.TestCase):
    @staticmethod
    def boundary():
        src=np.zeros((64,64),dtype=np.uint8); src[8:56,8:32]=1; src[8:56,32:56]=2; return src

    def test_exact_shared_boundary_passes(self):
        src=self.boundary(); m=measure_label_topology(src,src.copy()); s=m["shared_boundary"]
        self.assertTrue(m["complete"]); self.assertEqual(m["adjacency_loss_count"],0); self.assertEqual(s["max_gap_ratio"],0.0); self.assertEqual(s["max_double_line_ratio"],0.0); self.assertEqual(s["max_overlap_ratio"],0.0); self.assertEqual(s["max_drift_p95_px"],0.0)

    def test_gap_is_hard_stop_signal(self):
        src=self.boundary(); r=src.copy(); r[8:56,31:33]=0; m=measure_label_topology(src,r)
        self.assertGreater(m["shared_boundary"]["max_gap_ratio"],0.0); self.assertGreater(m["adjacency_loss_count"],0)

    def test_double_line_is_hard_stop_signal(self):
        src=self.boundary(); r=src.copy(); r[8:56,30]=2; r[8:56,31]=1; r[8:56,32]=2; m=measure_label_topology(src,r)
        self.assertGreater(m["shared_boundary"]["max_double_line_ratio"],0.0)

    def test_overlap_is_hard_stop_signal(self):
        src=self.boundary(); r=src.copy(); r[8:12,32]=1; m=measure_label_topology(src,r)
        self.assertGreater(m["shared_boundary"]["max_overlap_ratio"],0.01)

    def test_boundary_drift_is_hard_stop_signal(self):
        src=self.boundary(); r=src.copy(); r[8:56,32:34]=1; m=measure_label_topology(src,r)
        self.assertGreater(m["shared_boundary"]["max_drift_p95_px"],0.75)

    def test_adjacent_region_separation_is_hard_stop_signal(self):
        src=self.boundary(); r=src.copy(); r[8:56,31:34]=0; m=measure_label_topology(src,r)
        self.assertGreater(m["adjacency_loss_count"],0)

    def test_missing_topology_metric_fails_closed_with_deterministic_code(self):
        base={"checks":[],"blocker_codes":[],"ready":True,"production_gate":False}
        result=extend_near_zero_contract(base,{},None)
        self.assertFalse(result["ready"]); self.assertIn("topology_metric_missing",result["blocker_codes"])

    def test_missing_shared_boundary_metric_fails_closed_with_deterministic_code(self):
        residual={"topology":{"complete":False,"hole_loss_count":0,"ring_break_count":0,"adjacency_loss_count":0,"shared_boundary":{"applicable":True,"max_gap_ratio":0.0,"max_double_line_ratio":0.0,"max_overlap_ratio":0.0,"max_drift_p95_px":None,"min_matched_transition_ratio":1.0}}}
        base={"checks":[],"blocker_codes":[],"ready":True,"production_gate":False}
        result=extend_near_zero_contract(base,residual,None)
        self.assertFalse(result["ready"]); self.assertIn("shared_boundary_metric_missing",result["blocker_codes"])

    def test_hole_loss_is_hard_stop_signal(self):
        yy,xx=np.ogrid[:64,:64]; src=np.zeros((64,64),dtype=np.uint8); ring=((xx-32)**2+(yy-32)**2<20**2)&((xx-32)**2+(yy-32)**2>9**2); src[ring]=1; r=src.copy(); r[(xx-32)**2+(yy-32)**2<=9**2]=1
        self.assertGreater(measure_label_topology(src,r)["hole_loss_count"],0)

    def test_ring_break_is_hard_stop_signal(self):
        yy,xx=np.ogrid[:64,:64]; src=np.zeros((64,64),dtype=np.uint8); ring=((xx-32)**2+(yy-32)**2<20**2)&((xx-32)**2+(yy-32)**2>9**2); src[ring]=1; r=src.copy(); r[:,31:33]=0
        self.assertGreater(measure_label_topology(src,r)["ring_break_count"],0)

if __name__=="__main__": unittest.main()
