from __future__ import annotations
import inspect, unittest
import numpy as np
from engine.regression import output_quality_residual_suite as suite
from engine.regression import residual_multiscale_source as source

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

if __name__=="__main__": unittest.main()
