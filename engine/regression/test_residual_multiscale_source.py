from __future__ import annotations
import inspect, unittest
import numpy as np
from engine.regression import output_quality_residual_suite as suite
from engine.regression import residual_measurement_contract as measurement
from engine.regression import residual_multiscale_source as source
from engine.regression.residual_semantic_geometry import measure_alpha_support_geometry

class AnalyticMultiScaleContractTests(unittest.TestCase):
    def test_all_twelve_builders_verify_at_five_scales(self):
        self.assertEqual(len(suite._BUILDER_BY_CASE),12)
        for case_id,builder in suite._BUILDER_BY_CASE.items():
            with self.subTest(case_id=case_id):
                contract=source.validate_analytic_scale_contract(builder)
                self.assertTrue(contract["verified"],contract["failures"])
                self.assertEqual([x["size"] for x in contract["levels"]],[64,128,256,512,1024])
                self.assertEqual(len({x["geometry_sha256"] for x in contract["levels"]}),1)
                self.assertTrue(all(x["shape"]==[x["size"],x["size"],4] for x in contract["levels"]))

    def test_analytic_256_matches_native_fixture_bytes(self):
        for case_id,builder in suite._BUILDER_BY_CASE.items():
            with self.subTest(case_id=case_id):
                native=np.asarray(builder(256).convert("RGBA"),dtype=np.uint8)
                analytic=np.asarray(source.build_analytic_scene(builder,256).image.convert("RGBA"),dtype=np.uint8)
                self.assertTrue(np.array_equal(native,analytic),case_id)

    def test_source_path_contains_no_raster_resize(self):
        text=inspect.getsource(source)
        self.assertNotIn("cv2.resize",text)
        self.assertNotIn(".resize(",text)
        self.assertNotIn("INTER_AREA",text)
        self.assertNotIn("INTER_LANCZOS",text)
        self.assertNotIn("INTER_NEAREST",text)

    def test_monoline_and_shared_boundary_topology_stable(self):
        for case_id in ("qa-monoline","qa-shared-boundary"):
            c=source.validate_analytic_scale_contract(suite._BUILDER_BY_CASE[case_id])
            self.assertTrue(c["verified"],c)
            signatures=[level["topology"] for level in c["levels"]]
            self.assertTrue(all(item==signatures[0] for item in signatures[1:]))

    def test_unknown_builder_fails_closed(self):
        def unknown(size): return None
        with self.assertRaises(ValueError): source.build_analytic_scene(unknown,64)

    def test_lowres_tone_collapse_is_not_source_reconstructed_into_a_pass(self):
        builder=suite._BUILDER_BY_CASE["qa-lowres-badge"]
        scene=source.build_analytic_scene(builder,256)
        source_rgba=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8)
        rendered=source_rgba.copy()
        rendered[scene.labels==2,:3]=255
        result=measurement.measure_residual_error(source_rgba,rendered)
        self.assertLess(result["source_component_recall"],1.0)
        self.assertEqual(result["semantic_geometry"]["overrides"],[])
        self.assertEqual(result["semantic_geometry"]["source_match"],"exact_verified_analytic_rgba")

    def test_soft_alpha_uses_hard_shapes_plus_alpha_support_not_shadow_tone_bands(self):
        builder=suite._BUILDER_BY_CASE["qa-soft-alpha-shadow"]
        scene=source.build_analytic_scene(builder,256)
        rgba=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8)
        result=measurement.measure_residual_error(rgba,rgba.copy())
        self.assertEqual(result["source_component_recall"],1.0)
        self.assertEqual(result["render_component_precision"],1.0)
        self.assertEqual(result["min_component_iou"],1.0)
        self.assertEqual(result["boundary_p95_px"],0.0)
        self.assertFalse(result["topology"]["shared_boundary"]["applicable"])
        self.assertEqual(result["semantic_geometry"]["excluded_source_labels"],[1])

    def test_soft_alpha_missing_hard_shape_stays_detectable(self):
        builder=suite._BUILDER_BY_CASE["qa-soft-alpha-shadow"]
        scene=source.build_analytic_scene(builder,256)
        source_rgba=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8)
        rendered=source_rgba.copy()
        rendered[scene.labels==3]=0
        result=measurement.measure_residual_error(source_rgba,rendered)
        self.assertLess(result["source_component_recall"],1.0)
        self.assertLess(result["min_component_iou"],0.95)

    def test_alpha_support_identity_is_clean_and_has_no_fake_shared_boundary(self):
        rgba=np.zeros((64,64,4),dtype=np.uint8)
        rgba[8:56,8:56,3]=180
        measured=measure_alpha_support_geometry(rgba,rgba.copy())
        self.assertEqual(measured["boundary"]["boundary_p95_px"],0.0)
        self.assertEqual(measured["topology"]["hole_loss_count"],0)
        self.assertEqual(measured["topology"]["ring_break_count"],0)
        self.assertFalse(measured["topology"]["shared_boundary"]["applicable"])

    def test_alpha_support_hole_loss_is_a_real_positive_control(self):
        source_rgba=np.zeros((64,64,4),dtype=np.uint8)
        render_rgba=np.zeros_like(source_rgba)
        yy,xx=np.ogrid[:64,:64]
        ring=((xx-32)**2+(yy-32)**2<20**2)&((xx-32)**2+(yy-32)**2>9**2)
        source_rgba[ring,3]=255
        render_rgba[(xx-32)**2+(yy-32)**2<20**2,3]=255
        measured=measure_alpha_support_geometry(source_rgba,render_rgba)
        self.assertGreater(measured["topology"]["hole_loss_count"],0)


    def test_continuous_analytic_palette_uses_semantic_shared_boundary(self):
        builder=suite._BUILDER_BY_CASE["qa-hard-stop-gradient"]
        scene=source.build_analytic_scene(builder,256)
        rgba=np.asarray(scene.image.convert("RGBA"),dtype=np.uint8)
        result=measurement.measure_residual_error(rgba,rgba.copy())
        self.assertEqual(result["semantic_geometry"]["palette_mode"],"continuous_analytic")
        self.assertEqual(result["source_component_recall"],1.0)
        self.assertEqual(result["render_component_precision"],1.0)
        self.assertEqual(result["min_component_iou"],1.0)
        self.assertEqual(result["boundary_p95_px"],0.0)
        self.assertEqual(result["topology"]["adjacency_loss_count"],0)
        shared=result["topology"]["shared_boundary"]
        self.assertTrue(shared["applicable"])
        self.assertEqual(shared["max_gap_ratio"],0.0)
        self.assertEqual(shared["max_overlap_ratio"],0.0)
        self.assertEqual(shared["min_matched_transition_ratio"],1.0)

if __name__=="__main__": unittest.main()
