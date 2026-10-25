import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
import unittest

ROOT=Path("/Users/n/Work/numi-human-connected-surface-069")
MOD=ROOT/"embedded_self_separator_078.py"
spec=importlib.util.spec_from_file_location("separator078",MOD)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SeparatorTests(unittest.TestCase):
    def test_shared_identity_zero_row_is_omitted_and_baseline_passes(self):
        A=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        B=np.array([[0.,0.,0.],[-1.,0.,0.],[0.,-1.,0.]])
        idsA=np.array([0,1,2]);idsB=np.array([0,3,4])
        X=np.vstack([A,B[1:]])
        W=sparse.eye(5,format="csr")
        rows,rec,prov=m.pair_rows(A,B,idsA,idsB,W[idsA],W[idsB],X,A,B,0)
        self.assertEqual(rec["status"],"ok")
        self.assertEqual(rec["shared_vertex_ids"],[0])
        self.assertEqual(len(rows),8)
        shared=[p for p in prov if p["status"]=="omitted_shared_identity_zero_row"]
        self.assertEqual(len(shared),1)
        self.assertEqual(shared[0]["target"],0.0)
        mat=sparse.vstack(rows,format="csr")
        targets=np.array([p["target"] for p in prov if p["status"]=="constraint"])
        slack=np.asarray(mat@X.reshape(-1)-targets)
        self.assertGreaterEqual(float(slack.min()),-1e-12)

    def test_nonshared_positive_margin_and_candidate_violation(self):
        A=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        B=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        idsA=np.array([0,1,2]);idsB=np.array([3,4,5])
        X=np.vstack([A,B])
        W=sparse.eye(6,format="csr")
        candA=A.copy();candB=B.copy();candB[:,2]=.4
        rows,rec,prov=m.pair_rows(A,B,idsA,idsB,W[idsA],W[idsB],X,candA,candB,1)
        self.assertEqual(len(rows),9)
        self.assertGreater(rec["margin_m"],0.)
        self.assertAlmostEqual(rec["margin_m"],.5,places=12)
        self.assertGreater(rec["candidate_margin_violation_m"],0.)
        mat=sparse.vstack(rows,format="csr")
        targets=np.array([p["target"] for p in prov if p["status"]=="constraint"])
        self.assertGreaterEqual(float(np.min(mat@X.reshape(-1)-targets)),-1e-12)

    def test_emitted_actual_constraints_baseline_and_candidate_witnesses(self):
        out=ROOT/"E/embedded-self-separator-078/run-002"
        A=sparse.load_npz(out/"separator-inequalities.npz")
        z=np.load(out/"separator-targets.npz")
        report=__import__("json").loads((out/"report.json").read_text())
        base=np.load(m.BASE)["current_points_f32"]
        np.load(m.CAND)["points_world_f32"]
        self.assertEqual(A.shape,(25,9320*3))
        self.assertEqual(len(z["target"]),25)
        self.assertGreaterEqual(float(np.min(A@base.reshape(-1)-z["target"])),-1e-12)
        self.assertEqual(len(report["pairs"]),3)
        self.assertTrue(all(p["candidate_margin_violation_m"]>0 for p in report["pairs"]))
        self.assertEqual(report["omitted_zero_rows"],2)
        self.assertTrue(all(p["status"]=="omitted_shared_identity_zero_row"
                            for p in report["row_provenance"] if p["status"]!="constraint"))

if __name__=="__main__":unittest.main()

