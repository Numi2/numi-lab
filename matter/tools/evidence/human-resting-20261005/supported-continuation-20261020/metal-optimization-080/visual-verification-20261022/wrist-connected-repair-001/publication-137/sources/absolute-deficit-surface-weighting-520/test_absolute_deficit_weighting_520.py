import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("absolute_deficit_weighting_520",HERE/"absolute_deficit_weighting_520.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class AbsoluteDeficitWeighting520Tests(unittest.TestCase):
    def test_weight_over_h_is_constant_and_family_mass_is_preserved(self):
        h=np.array([1e-4,.02,1.5],dtype=np.float64)
        original=np.array([.2,.2,.1],dtype=np.float64)
        result={"h":h,"weights":original,"C":np.array([[1.],[2.],[3.]]),
                "G":np.array([[4.]]),"lower":np.array([5.]),
                "metadata":[{"kind":"face","normalized_weight":float(w)} for w in original],
                "diagnostics":{"kept":True}}
        out=mod.reweight_surface_rows(result,total_weight=.5)
        np.testing.assert_allclose(out["weights"].sum(),.5,rtol=0,atol=1e-15)
        np.testing.assert_allclose(out["weights"]/h,np.full(3,.5/h.sum()),rtol=1e-14,atol=0)
        np.testing.assert_array_equal(out["h"],h)
        np.testing.assert_array_equal(out["C"],result["C"])
        np.testing.assert_array_equal(out["G"],result["G"])
        np.testing.assert_array_equal(out["lower"],result["lower"])
        self.assertEqual(out["diagnostics"],result["diagnostics"])
        np.testing.assert_array_equal([m["normalized_weight"] for m in out["metadata"]],out["weights"])
        np.testing.assert_array_equal(result["weights"],original)

    def test_rejects_zero_or_mismatched_rows_and_wrong_mass(self):
        base={"h":np.array([.1]),"weights":np.array([.5])}
        for h,w,mass in [(np.array([0.]),np.array([.5]),.5),
                         (np.array([.1,.2]),np.array([.5]),.5),
                         (np.array([.1]),np.array([.4]),.5)]:
            with self.assertRaises(ValueError):
                mod.reweight_surface_rows({"h":h,"weights":w},total_weight=mass)

    def test_point_prefix_is_independent(self):
        point_h=np.array([.006,16.64]); point_w=np.array([.25,.25])
        surface=mod.reweight_surface_rows({"h":np.array([.001,.2]),"weights":np.array([.25,.25])},total_weight=.5)
        combined_h=np.concatenate((point_h,np.array([.001,.2])))
        combined_w=np.concatenate((point_w,surface["weights"]))
        np.testing.assert_array_equal(combined_h[:2],point_h)
        np.testing.assert_array_equal(combined_w[:2],point_w)
        self.assertAlmostEqual(float(combined_w[2:].sum()),.5)
        self.assertAlmostEqual(float(combined_w.sum()),1.0)

if __name__=="__main__":unittest.main()
