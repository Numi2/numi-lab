import unittest, subprocess, sys, json
from pathlib import Path
import numpy as np
from scipy import sparse
from fine_point_basis_405 import enrich_point_basis
from surface_witness_basis_408 import enrich_surface_basis
class Tests(unittest.TestCase):
 def setUp(self):
  self.scope=np.arange(6,dtype=np.int64); self.anchors=np.array([0],dtype=np.int64)
  self.B=sparse.csr_matrix(np.array([[0,0],[1,0],[.5,.5],[0,1],[.25,.75],[.7,.3]]))
 def test_union_and_parent_span(self):
  M,r=enrich_surface_basis(self.B,self.scope,self.anchors,np.array([2]),np.array([[0,3,4],[3,4,4]]))
  self.assertEqual(r['fine_output_node_origin_flags_408'],[1,2,2]);self.assertEqual(r['fine_output_node_rows'],[2,3,4]); self.assertEqual((M[:,:2]-self.B).nnz,0)
  self.assertEqual(M[0].nnz,0);self.assertEqual(r['requested_unique_union_rows'],4)
 def test_same_as_explicit_union(self):
  M,_=enrich_surface_basis(self.B,self.scope,self.anchors,np.array([2]),np.array([[0,3,4]]))
  N,_=enrich_point_basis(self.B,self.scope,self.anchors,np.array([0,2,3,4]))
  self.assertEqual((M-N).nnz,0)
 def test_no_surface_keeps_point_path(self):
  M,_=enrich_surface_basis(self.B,self.scope,self.anchors,np.array([2]),np.empty((0,3),dtype=np.int64))
  N,_=enrich_point_basis(self.B,self.scope,self.anchors,np.array([2]))
  self.assertEqual((M-N).nnz,0)
 def test_existing_output_constraints_stay_on_output(self):
  M,_=enrich_surface_basis(self.B,self.scope,self.anchors,np.array([2]),np.array([[1,3,4]]))
  rng=np.random.default_rng(408);A=rng.normal(size=(17,6));q=rng.normal(size=M.shape[1]);d=M@q
  np.testing.assert_allclose((A@M)@q,A@d,rtol=1e-14,atol=1e-14)
 def test_interleaved_prior_point_column(self):
  M,r=enrich_surface_basis(self.B,self.scope,self.anchors,np.array([4]),np.array([[1,2,3]]))
  N,_=enrich_point_basis(self.B,self.scope,self.anchors,np.array([4]))
  row=r['fine_output_node_rows'].index(4)+self.B.shape[1]
  self.assertEqual((M[:,:self.B.shape[1]]-self.B).nnz,0)
  self.assertEqual((M[:,row]-N[:,-1]).nnz,0)
 def test_invalid_surface_rows(self):
  for x in [np.array([1,2,3]),np.array([[1.,2.,3.]]),np.array([[-1,2,3]]),np.array([[1,2,7]])]:
   with self.subTest(x=x.tolist()):
    with self.assertRaises(ValueError):enrich_surface_basis(self.B,self.scope,self.anchors,np.array([2]),x)
 def test_invalid_point_rows(self):
  for x in [np.array([-1]),np.array([2.]),np.array([[2]])]:
   with self.subTest(x=x.tolist()):
    with self.assertRaises(ValueError):enrich_surface_basis(self.B,self.scope,self.anchors,x,np.array([[1,2,3]]))
 def test_isolated_identity_skip_report_is_json_serializable(self):
  B=sparse.csr_matrix(np.array([[0.,0.],[1.,0.],[0.,.5],[0.,.5]]))
  M,r=enrich_surface_basis(B,np.arange(4),np.array([0]),np.array([1]),np.array([[1,2,3]]))
  self.assertEqual(r["excluded_isolated_identity_rows"],[1])
  self.assertEqual(r["fine_output_node_rows"],[2,3])
  self.assertIs(type(r["excluded_isolated_identity_rows"][0]),int)
  json.dumps(r)
 def test_cli_requires_fine_point(self):
  p=Path(__file__).with_name('propose_joint_clearance_witnesses_408.py')
  cp=subprocess.run([sys.executable,str(p),'--fine-surface-controls',*[v for k in ['--base-candidate','--raw-trial','--self-pairs','--area-reference-npz','--point-target-npz','--point-target-report','--face-target-npz','--face-target-report','--output'] for v in [k,'unused']]],capture_output=True,text=True)
  self.assertNotEqual(cp.returncode,0);self.assertIn('fine-surface-controls requires fine-point-controls',cp.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
